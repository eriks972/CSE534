import json
import matplotlib.pyplot as plt

with open("model_variant_results.json", "r") as f:
    results = json.load(f)

models = [r["model"] for r in results]
latencies = [r["avg_latency"] for r in results]
detections = [r["avg_detections"] for r in results]
confidences = [r["avg_confidence"] for r in results]

plt.figure()
plt.bar(models, latencies)
plt.xlabel("YOLO Model Variant")
plt.ylabel("Average Latency (seconds)")
plt.title("Latency Across YOLO Model Variants")
plt.savefig("model_variant_latency.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure()
plt.bar(models, detections)
plt.xlabel("YOLO Model Variant")
plt.ylabel("Average Number of Detections")
plt.title("Detection Count Across YOLO Model Variants")
plt.savefig("model_variant_detections.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure()
plt.bar(models, confidences)
plt.xlabel("YOLO Model Variant")
plt.ylabel("Average Confidence")
plt.title("Detection Confidence Across YOLO Model Variants")
plt.savefig("model_variant_confidence.png", dpi=300, bbox_inches="tight")
plt.show()