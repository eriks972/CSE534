import json
import matplotlib.pyplot as plt

with open("results.json", "r") as f:
    data = json.load(f)

sizes = ["small", "medium", "large"]

for size in sizes:
    delays = []
    cloud = []

    for entry in data:
        if entry["size"] == size:
            delays.append(entry["delay"] * 1000)
            cloud.append(entry["cloud_avg"])

    plt.plot(delays, cloud, marker='o', label=f"Cloud ({size})")

plt.xlabel("Delay (ms)")
plt.ylabel("Latency (s)")
plt.title("Impact of Image Size on Cloud Latency")

plt.legend()
plt.grid()
plt.show()