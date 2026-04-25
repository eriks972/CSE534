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


def cloud_inference(img_path, delay=0):
    start = time.time()

    time.sleep(delay)  # simulate network latency before request

    img_encoded = encode_image(img_path)
    response = requests.post(SERVER_URL, json={"image": img_encoded})
    response.raise_for_status()

    end = time.time()
    return end - start


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

for delay in delays:
    edge_times = []
    cloud_times = []

    print(f"Running delay: {int(delay * 1000)} ms")

    for img in images:
        path = os.path.join(IMAGE_DIR, img)

        edge_time = edge_inference(path)
        cloud_time = cloud_inference(path, delay)

        edge_times.append(edge_time)
        cloud_times.append(cloud_time)

        print(
            f"{img} | Edge: {edge_time:.4f}s | "
            f"Cloud ({int(delay * 1000)}ms): {cloud_time:.4f}s"
        )

    results.append({
        "delay": delay,
        "delay_ms": int(delay * 1000),
        "num_images": len(images),
        "edge_avg": sum(edge_times) / len(edge_times),
        "cloud_avg": sum(cloud_times) / len(cloud_times)
    })

print("\nFinal Results:")
print(json.dumps(results, indent=4))

with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=4)