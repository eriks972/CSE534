import json
import matplotlib.pyplot as plt

with open("results.json", "r") as f:
    results = json.load(f)

delays = [r["delay"] * 1000 for r in results]  # convert to ms
edge = [r["edge_avg"] for r in results]
cloud = [r["cloud_avg"] for r in results]

plt.figure()

plt.plot(delays, edge, marker='o', label="Edge")
plt.plot(delays, cloud, marker='o', label="Cloud")

plt.xlabel("Network Delay (ms)")
plt.ylabel("Latency (seconds)")
plt.title("Edge vs Cloud Latency under Network Delay")

plt.legend()
plt.grid()

plt.show()