import os
import time
import json
from ultralytics import YOLO

IMAGE_DIR = "../dataset/images"
RESULTS_FILE = "../results/model_variant_results.json"

MODEL_VARIANTS = [
    "yolov8n.pt",  # nano: fastest
    "yolov8s.pt",  # small: more accurate/slower
    "yolov8m.pt"   # medium: heavier
]

images = sorted([
    img for img in os.listdir(IMAGE_DIR)
    if img.lower().endswith((".jpg", ".jpeg", ".png"))
])

if not images:
    raise ValueError(f"No images found in {IMAGE_DIR}")

# Optional: limit while testing
# images = images[:20]

all_results = []

for model_name in MODEL_VARIANTS:
    print(f"\nRunning model: {model_name}")

    model = YOLO(model_name)

    # warmup
    warmup_path = os.path.join(IMAGE_DIR, images[0])
    model(warmup_path)

    inference_times = []
    detection_counts = []
    confidence_scores = []

    for img in images:
        img_path = os.path.join(IMAGE_DIR, img)

        start = time.time()
        results = model(img_path)
        end = time.time()

        latency = end - start
        boxes = results[0].boxes

        num_detections = len(boxes)

        if num_detections > 0:
            avg_confidence = float(boxes.conf.mean())
        else:
            avg_confidence = 0.0

        inference_times.append(latency)
        detection_counts.append(num_detections)
        confidence_scores.append(avg_confidence)

        print(
            f"{img} | {model_name} | "
            f"Latency: {latency:.4f}s | "
            f"Detections: {num_detections} | "
            f"Avg Conf: {avg_confidence:.3f}"
        )

    model_result = {
        "model": model_name,
        "num_images": len(images),
        "avg_latency": sum(inference_times) / len(inference_times),
        "avg_detections": sum(detection_counts) / len(detection_counts),
        "avg_confidence": sum(confidence_scores) / len(confidence_scores),
        "min_latency": min(inference_times),
        "max_latency": max(inference_times)
    }

    all_results.append(model_result)

print("\nFinal Model Variant Results:")
print(json.dumps(all_results, indent=4))

with open(RESULTS_FILE, "w") as f:
    json.dump(all_results, f, indent=4)