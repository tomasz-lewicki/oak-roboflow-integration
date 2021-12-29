import base64
import io
import time

import numpy as np
import requests
from PIL import Image

from annotation import make_voc_annotations


class Config:
    dataset_name = "oak-d-dataset"
    api_key = "vkIkZac3CXvp0RZ31B3f"


def upload(unique_id):
    # Load Image with PIL

    arr = (np.random.random((500, 500, 3)) * 255).astype(np.uint8)
    image = Image.fromarray(arr)

    # Convert to JPEG Buffer
    buffered = io.BytesIO()
    image.save(buffered, quality=90, format="JPEG")

    # Base 64 Encode
    img_str = base64.b64encode(buffered.getvalue())
    img_str = img_str.decode("ascii")

    # Construct the URL
    upload_url = "".join(
        [
            f"https://api.roboflow.com/dataset/{Config.dataset_name}/upload",
            f"?api_key={Config.api_key}",
            f"&name={unique_id}.jpg",  # Epoch timestamp in miliseconds e.g. 1640677054993.jpg
            "&split=train",
        ]
    )

    # POST to the API
    r = requests.post(
        upload_url,
        data=img_str,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    # Output result
    print(r.json())

    return r.json()["id"]


def annotate(image_id, unique_id):
    # keep track of image_id to associate with an image
    # time.sleep(10)

    annotation_str = make_voc_annotations(
        ["helmet", "helmet"], [[179, 85, 231, 144], [112, 145, 135, 175]]
    )

    # Construct the URL
    upload_url = "".join(
        [
            f"https://api.roboflow.com/dataset/oak-d-dataset/annotate/{image_id}",
            f"?api_key=vkIkZac3CXvp0RZ31B3f",
            f"&name={unique_id}.xml",
        ]
    )

    print(upload_url)
    # POST to the API
    r = requests.post(
        upload_url, data=annotation_str, headers={"Content-Type": "text/plain"}
    )

    # Output result
    print(r.json())


if __name__ == "__main__":

    start = time.perf_counter()
    unique_id = int(1000 * time.time())
    img_id = upload(unique_id)
    annotate(img_id, unique_id)
    print(time.perf_counter() - start)
