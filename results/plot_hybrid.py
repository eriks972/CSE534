import json
import matplotlib.pyplot as plt

with open("hybrid_results.json", "r") as f:
    results = json.load(f)

delays = [r["delay_ms"] for r in results]
edge = [r["edge_avg"] for r in results]
cloud = [r["cloud_avg"] for r in results]
hybrid = [r["hybrid_avg"] for r in results]

plt.figure()

plt.plot(delays, edge, marker="o", label="Edge-only")
plt.plot(delays, cloud, marker="o", label="Cloud-only")
plt.plot(delays, hybrid, marker="o", label="Hybrid Policy")

plt.xlabel("Network Delay (ms)")
plt.ylabel("Average Latency (seconds)")
plt.title("Edge vs Cloud vs Hybrid Decision Policy")
plt.legend()
plt.grid()

plt.savefig("hybrid_policy_plot.png", dpi=300, bbox_inches="tight")
plt.show()