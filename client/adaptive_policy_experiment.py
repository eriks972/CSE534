import os
import time
import json
import base64
import requests
from ultralytics import YOLO

SERVER_URL = "http://127.0.0.1:5000/process"
IMAGE_DIR = "../dataset/images"
RESULTS_FILE = "../results/adaptive_policy_results.json"

delays = [0, 0.05, 0.1, 0.2]

models = {
    "yolov8n": YOLO("yolov8n.pt"),
    "yolov8s": YOLO("yolov8s.pt"),
    "yolov8m": YOLO("yolov8m.pt")
}

images = sorted([
    img for img in os.listdir(IMAGE_DIR)
    if img.lower().endswith((".jpg", ".jpeg", ".png"))
])

if not images:
    raise ValueError("No images found.")

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def edge_inference(img_path, model_name):
    model = models[model_name]

    start = time.time()
    result = model(img_path)
    end = time.time()

    boxes = result[0].boxes
    detections = len(boxes)
    confidence = float(boxes.conf.mean()) if detections > 0 else 0.0

    return end - start, detections, confidence

def cloud_inference(img_path, model_name, delay):
    start = time.time()

    time.sleep(delay)

    img_encoded = encode_image(img_path)

    response = requests.post(SERVER_URL, json={
        "image": img_encoded,
        "model": model_name
    })

    response.raise_for_status()
    data = response.json()

    end = time.time()

    return end - start, data["detections"], data.get("avg_confidence", 0.0)

def adaptive_policy(delay):
    if delay < 0.05:
        return "cloud", "yolov8m"
    elif delay < 0.1:
        return "cloud", "yolov8s"
    else:
        return "edge", "yolov8n"

# warm up models
warmup_path = os.path.join(IMAGE_DIR, images[0])
for model in models.values():
    model(warmup_path)

results = []

for delay in delays:
    adaptive_times = []
    adaptive_detections = []
    adaptive_confidences = []

    edge_n_times = []
    cloud_m_times = []

    decisions = {}

    print(f"\nRunning delay: {int(delay * 1000)} ms")

    for img in images:
        img_path = os.path.join(IMAGE_DIR, img)

        location, model_name = adaptive_policy(delay)
        decision_key = f"{location}_{model_name}"
        decisions[decision_key] = decisions.get(decision_key, 0) + 1

        if location == "edge":
            latency, detections, confidence = edge_inference(img_path, model_name)
        else:
            latency, detections, confidence = cloud_inference(img_path, model_name, delay)

        adaptive_times.append(latency)
        adaptive_detections.append(detections)
        adaptive_confidences.append(confidence)

        edge_n_latency, _, _ = edge_inference(img_path, "yolov8n")
        cloud_m_latency, _, _ = cloud_inference(img_path, "yolov8m", delay)

        edge_n_times.append(edge_n_latency)
        cloud_m_times.append(cloud_m_latency)

    results.append({
        "delay": delay,
        "delay_ms": int(delay * 1000),
        "num_images": len(images),

        "adaptive_avg_latency": sum(adaptive_times) / len(adaptive_times),
        "adaptive_avg_detections": sum(adaptive_detections) / len(adaptive_detections),
        "adaptive_avg_confidence": sum(adaptive_confidences) / len(adaptive_confidences),

        "edge_yolov8n_avg_latency": sum(edge_n_times) / len(edge_n_times),
        "cloud_yolov8m_avg_latency": sum(cloud_m_times) / len(cloud_m_times),

        "decisions": decisions
    })

print(json.dumps(results, indent=4))

with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=4)