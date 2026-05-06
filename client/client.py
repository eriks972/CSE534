import cv2
import requests
import time
import base64
import os
from ultralytics import YOLO
import json

SERVER_URL = "http://127.0.0.1:5000/process"
IMAGE_DIR = "../dataset/images"
RESULTS_FILE = "../results/results.json"

model = YOLO("yolov8n.pt")


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def edge_inference(img_path):
    start = time.time()
    model(img_path)
    end = time.time()
    return end - start

def hybrid_inference(img_path, delay, threshold=0.1):
    if delay < threshold:
        latency = cloud_inference(img_path, delay)
        decision = "cloud"
    else:
        latency = edge_inference(img_path)
        decision = "edge"

    return latency, decision

def cloud_inference(img_path, delay=0):
    start = time.time()

    time.sleep(delay)  # simulate network latency before request

    img_encoded = encode_image(img_path)
    response = requests.post(SERVER_URL, json={"image": img_encoded})
    response.raise_for_status()

    end = time.time()
    return end - start

def resize_image(path, size):
    img = cv2.imread(path)
    resized = cv2.resize(img, size)

    temp_path = "temp.jpg"
    cv2.imwrite(temp_path, resized)

    return temp_path

# only keep image files
images = sorted([
    img for img in os.listdir(IMAGE_DIR)
    if img.lower().endswith((".jpg", ".jpeg", ".png"))
])

if not images:
    raise ValueError(f"No images found in {IMAGE_DIR}")

# warm up model using first available image
warmup_path = os.path.join(IMAGE_DIR, images[0])
model(warmup_path)

results = []
delays = [0, 0.05, 0.1, 0.2]  # seconds = 0ms, 50ms, 100ms, 200ms
sizes = {
    "small": (320, 320),
    "medium": (640, 640),
    "large": (1280, 1280)
}

for size_name, size in sizes.items():
    print(f"\nRunning size: {size_name}")

    results = []
    delays = [0, 0.05, 0.1, 0.2]

    for delay in delays:
        edge_times = []
        cloud_times = []
        hybrid_times = []

        hybrid_edge_count = 0
        hybrid_cloud_count = 0

        print(f"\nRunning delay: {int(delay * 1000)} ms")

        for img in images:
            path = os.path.join(IMAGE_DIR, img)

            edge_time = edge_inference(path)
            cloud_time = cloud_inference(path, delay)
            hybrid_time, decision = hybrid_inference(path, delay)

            edge_times.append(edge_time)
            cloud_times.append(cloud_time)
            hybrid_times.append(hybrid_time)

            if decision == "edge":
                hybrid_edge_count += 1
            else:
                hybrid_cloud_count += 1

        results.append({
            "delay": delay,
            "delay_ms": int(delay * 1000),
            "num_images": len(images),

            "edge_avg": sum(edge_times) / len(edge_times),
            "cloud_avg": sum(cloud_times) / len(cloud_times),
            "hybrid_avg": sum(hybrid_times) / len(hybrid_times),

            "hybrid_edge_count": hybrid_edge_count,
            "hybrid_cloud_count": hybrid_cloud_count
        })

print("\nFinal Results:")
print(json.dumps(results, indent=4))

with open("../results/hybrid_results.json", "w") as f:
    json.dump(results, f, indent=4)
    
with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=4)