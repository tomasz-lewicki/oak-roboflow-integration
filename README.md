# OAK + Roboflow Demo
An app creating [Roboflow](roboflow.com) dataset using detections from an [OAK-1](https://store.opencv.ai/products/oak-1) camera.

## Demo
Live preview shows MobileNet SSD detections. After pressing `enter` the app grabs frames and uploads them to Roboflow dataset.

https://user-images.githubusercontent.com/26127866/147658296-23be4621-d37a-4fd6-a169-3ea414ffa636.mp4

## Getting Started

1. If it's your first project with OAK, follow this [first steps guide](https://docs.luxonis.com/en/latest/pages/tutorials/first_steps/#first-steps-with-depthai).

2. Clone repository. Install requirements.
```bash
git clone https://github.com/tomek-l/oak-roboflow-integration.git
cd oak-roboflow-integration
pip3 install -r requirements.txt
```

3. Setup Roboflow account
- Get API key ([app.roboflow.com](https://app.roboflow.com/) -> `settings` -> `workspaces` -> `Roboflow API` -> Copy private API key)
- Create new (empty) project at [app.roboflow.com](https://app.roboflow.com/) -> Copy project (a.k.a. dataset) name.
- Paste to [config.json](https://github.com/tomek-l/oak-roboflow-integration/blob/master/config.json), similarly as in the example below.

```json
{
    "dataset": "oak-dataset2",
    "api_key": "vkIkZac3CXvp0RZ31B3f",
    "upload_threshold": 0.8
}
```

4. Run the code!
```python
python3 main.py
```