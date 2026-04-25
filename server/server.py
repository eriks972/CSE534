from flask import Flask, request, jsonify
from ultralytics import YOLO
import time
import base64
import numpy as np
import cv2

app = Flask(__name__)
model = YOLO("yolov8n.pt")

@app.route("/process", methods=["POST"])
def process():
    
    time.sleep(0.05)  # simulate inference time (50ms)

    start = time.time()

    data = request.json["image"]

    # decode image
    img_bytes = base64.b64decode(data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # run model
    results = model(img)

    end = time.time()

    return jsonify({
        "latency": end - start,
        "detections": len(results[0].boxes)
    })

if __name__ == "__main__":
    app.run(port=5000)