import time
import json
from pathlib import Path
from collections import deque, OrderedDict
from concurrent.futures import ThreadPoolExecutor

import cv2
import depthai as dai
import numpy as np

from utils.roboflow import RoboflowUploader

BLOB_PATH = "models/mobilenet-ssd_openvino_2021.4_6shave.blob"
LABELS = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def make_pipeline():
    # Pipeline
    pipeline = dai.Pipeline()

    # Camera
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setPreviewSize(300, 300)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    camRgb.setPreviewKeepAspectRatio(False)
    camRgb.setInterleaved(False)
    camRgb.setFps(40)

    # Detector
    nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    nn.setConfidenceThreshold(0.5)
    nn.setBlobPath(BLOB_PATH)
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)

    # Image output
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")

    # Detection output
    nnOut = pipeline.create(dai.node.XLinkOut)
    nnOut.setStreamName("nn")

    # Link elements
    nn.passthrough.link(xoutRgb.input)
    camRgb.preview.link(nn.input)  # RGB buffer
    nn.out.link(nnOut.input)

    return pipeline


def get_config():

    with open("config.json") as f:
        config = json.loads(f.read())

    UPLOAD_THR = config["upload_threshold"]
    DATASET = config["dataset"]
    API_KEY = config["api_key"]

    return (DATASET, API_KEY, UPLOAD_THR)


def parse_dets(detections, confidence_thr=0.8):

    labels = [LABELS[d.label] for d in detections if d.confidence > confidence_thr]

    bboxes = [
        [300 * d.xmin, 300 * d.ymin, 300 * d.xmax, 300 * d.ymax]
        for d in detections
        if d.confidence > confidence_thr
    ]

    return labels, bboxes


# nn data (bounding box locations) are in <0..1> range - they need to be normalized with frame width/height
def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


def overlay_boxes(frame, detections):

    # Overlay on a copy of image to keep the original
    frame = frame.copy()
    BLUE = (255, 0, 0)

    for detection in detections:
        bbox = frameNorm(
            frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
        )
        cv2.putText(
            frame,
            LABELS[detection.label],
            (bbox[0] + 10, bbox[1] + 20),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            BLUE,
        )
        cv2.putText(
            frame,
            f"{int(detection.confidence * 100)}%",
            (bbox[0] + 10, bbox[1] + 40),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            BLUE,
        )
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), BLUE, 2)

    return frame


def upload_all(uploader, frame: np.ndarray, labels: list, bboxes: list, fname: str):
    """
    Uploads `frame` as an image to Roboflow and saves it under `fname`.jpg
    Then, upload annotations  with corresponding `bboxes` and `frame`
    """

    # Upload image frame. Retreive Roboflow's image_id
    img_id = uploader.upload_image(frame, fname)

    # Annotate the image we just uploaded
    uploader.upload_annotation(img_id, fname=fname, labels=labels, bboxes=bboxes)


def get_last_synced_pair(rgb_deque, dets_deque):
    # Returns (frame, dets) with the highest available seq_n or (None, None) if no mach found.

    # rgb_deque sorted by seq_n
    rgb_deque_s = sorted(rgb_deque, key=lambda x: x[1], reverse=True)

    # Dict mapping seq_n: dets
    seq2dets = {seq_n: det for (det, seq_n) in dets_deque}

    # ODict mapping {seq_n: (frame, dets)}. Ignores seq_n without dets
    seq2frames_dets = OrderedDict(
        (
            (seq_n, (frame, seq2dets.get(seq_n)))
            for frame, seq_n in rgb_deque_s
            if seq2dets.get(seq_n) is not None
        )
    )

    # Return matches if any exist
    if len(seq2frames_dets) > 0:
        frame, dets = list(seq2frames_dets.values())[0]
    else:
        frame, dets = None, None

    return frame, dets


if __name__ == "__main__":

    # Parse config
    (DATASET, API_KEY, UPLOAD_THR) = get_config()

    # Initialize variables
    frame = None
    detections = []
    WHITE = (255, 255, 255)

    # Queues for detections and frames. Used for syncing frame<->detections pairs.
    rgb_deque = deque(maxlen=10)
    det_deque = deque(maxlen=10)

    # Wrapper around Roboflow upload/annotate API
    uploader = RoboflowUploader(dataset_name=DATASET, api_key=API_KEY)

    # Executor to handle uploads asynchronously
    # For real-time uploads at ~10Hz we spawn 40 threads
    executor = ThreadPoolExecutor(max_workers=40)

    # DAI pipeline
    pipeline = make_pipeline()

    with dai.Device(pipeline) as device:

        queue_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        queue_dets = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

        # self.nnSyncSeq = min(map(lambda packet: packet.getSequenceNum(), qDet.values()))

        while True:

            rgb_msg = queue_rgb.get()  # instance of depthai.ImgFrame
            det_msg = queue_dets.get()  # instance of depthai.ImgDetections

            # Obtain sequence numbers to sync frames
            rgb_seq = rgb_msg.getSequenceNum()
            det_seq = det_msg.getSequenceNum()

            # Get frame and dets
            frame = rgb_msg.getCvFrame()  # np.ndarray / BGR CV Mat
            dets = det_msg.detections  # list of depthai.ImgDetection

            # Put (object, seq_n) tuples in a queue
            rgb_deque.append((frame, rgb_seq))
            det_deque.append((dets, det_seq))

            frame, dets = get_last_synced_pair(rgb_deque, det_deque)

            # Display results
            frame_with_boxes = overlay_boxes(frame, dets)
            cv2.imshow("Roboflow Demo", frame_with_boxes)

            # Handle user input
            key = cv2.waitKey(1)

            if key == ord("q"):
                # q -> exit
                exit()
            elif key == 13:
                # Enter -> upload to Roboflow
                labels, bboxes = parse_dets(dets, confidence_thr=UPLOAD_THR)
                print("Uploading grabbed frame!")
                executor.submit(
                    upload_all, uploader, frame, labels, bboxes, int(1000 * time.time())
                )
