import json
import matplotlib.pyplot as plt

with open("adaptive_policy_results.json", "r") as f:
    results = json.load(f)

delays = [r["delay_ms"] for r in results]

adaptive = [r["adaptive_avg_latency"] for r in results]
edge_n = [r["edge_yolov8n_avg_latency"] for r in results]
cloud_m = [r["cloud_yolov8m_avg_latency"] for r in results]

detections = [r["adaptive_avg_detections"] for r in results]
confidence = [r["adaptive_avg_confidence"] for r in results]

plt.figure()
plt.plot(delays, adaptive, marker="o", label="Adaptive Policy")
plt.plot(delays, edge_n, marker="o", label="Edge YOLOv8n")
plt.plot(delays, cloud_m, marker="o", label="Cloud YOLOv8m")
plt.xlabel("Network Delay (ms)")
plt.ylabel("Average Latency (seconds)")
plt.title("Adaptive Model and Location Selection")
plt.legend()
plt.grid()
plt.savefig("adaptive_policy_latency.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure()
plt.plot(delays, detections, marker="o", label="Adaptive Avg Detections")
plt.xlabel("Network Delay (ms)")
plt.ylabel("Average Detections")
plt.title("Adaptive Policy Detection Output")
plt.grid()
plt.savefig("adaptive_policy_detections.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure()
plt.plot(delays, confidence, marker="o", label="Adaptive Avg Confidence")
plt.xlabel("Network Delay (ms)")
plt.ylabel("Average Confidence")
plt.title("Adaptive Policy Detection Confidence")
plt.grid()
plt.savefig("adaptive_policy_confidence.png", dpi=300, bbox_inches="tight")
plt.show()