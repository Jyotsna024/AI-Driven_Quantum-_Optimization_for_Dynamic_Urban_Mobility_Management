import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/peak_s0_varied.csv")
df = df[df["is_warmup"] == False].copy()

plt.figure(figsize=(11, 6))

for junction, group in df.groupby("junction_short"):
    group = group.groupby("timestamp_s")["queue_length_m"].mean().reset_index()
    group = group.sort_values("timestamp_s")
    plt.plot(group["timestamp_s"], group["queue_length_m"], label=junction)

plt.title("Queue Length Over Simulation Time - Peak Scenario")
plt.xlabel("Simulation Time (seconds)")
plt.ylabel("Average Queue Length (metres)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/graphs/queue_over_time.png", dpi=300)
plt.close()

print("Queue time-series graph generated successfully!")
