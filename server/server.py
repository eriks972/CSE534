from flask import Flask, request, jsonify
from ultralytics import YOLO
import time
import base64
import numpy as np
import cv2

app = Flask(__name__)

models = {
    "yolov8n": YOLO("yolov8n.pt"),
    "yolov8s": YOLO("yolov8s.pt"),
    "yolov8m": YOLO("yolov8m.pt")
}

@app.route("/process", methods=["POST"])
def process():
    data = request.json

    image_data = data["image"]
    model_name = data.get("model", "yolov8n")

    model = models[model_name]

    img_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    start = time.time()
    results = model(img)
    end = time.time()

    boxes = results[0].boxes
    detections = len(boxes)
    avg_confidence = float(boxes.conf.mean()) if detections > 0 else 0.0

    return jsonify({
        "model": model_name,
        "server_inference_time": end - start,
        "detections": detections,
        "avg_confidence": avg_confidence
    })

if __name__ == "__main__":
    app.run(port=5000)