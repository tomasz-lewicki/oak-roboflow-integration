import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

from roboflow import RoboflowUploader

BLOB_PATH = "mobilenet-ssd_openvino_2021.4_6shave.blob"
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
    camRgb.preview.link(nn.input)
    nn.out.link(nnOut.input)

    return pipeline


def parse_dets(detections, confidence_thr=0.9):
    dets_out = [
        [300 * d.xmin, 300 * d.ymin, 300 * d.xmax, 300 * d.ymax]
        for d in detections
        if d.confidence > confidence_thr
    ]
    return dets_out


# nn data (bounding box locations) are in <0..1> range - they need to be normalized with frame width/height
def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


def displayFrame(name, frame, detections):
    color = (255, 0, 0)
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
            color,
        )
        cv2.putText(
            frame,
            f"{int(detection.confidence * 100)}%",
            (bbox[0] + 10, bbox[1] + 40),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            color,
        )
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    # Show the frame
    cv2.imshow(name, frame)


def mainloop():

    pipeline = make_pipeline()

    with dai.Device(pipeline) as device:

        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

        frame = None
        counter = 0
        detections = []
        startTime = time.monotonic()
        WHITE = (255, 255, 255)

        while True:

            inRgb = qRgb.get()
            inDet = qDet.get()

            if inRgb is not None:
                frame = inRgb.getCvFrame()
                cv2.putText(
                    frame,
                    "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                    (2, frame.shape[0] - 4),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.4,
                    WHITE,
                )

            if inDet is not None:
                detections = inDet.detections
                bboxes = parse_dets(detections, confidence_thr=0)
                print(bboxes)

            # If the frame is available, draw bounding boxes on it and show the frame
            if frame is not None:

                fname = "image" + str(int(1000 * time.time()))

                # displayFrame("rgb", frame, detections)

                img_id = uploader.upload_image(frame, fname)

                uploader.upload_annotation(img_id, fname, ["helmet", "helmet"], bboxes)

            if cv2.waitKey(1) == ord("q"):
                break


if __name__ == "__main__":

    uploader = RoboflowUploader(
        dataset_name="oak-dataset",
        api_key="vkIkZac3CXvp0RZ31B3f"
    )

    mainloop()
