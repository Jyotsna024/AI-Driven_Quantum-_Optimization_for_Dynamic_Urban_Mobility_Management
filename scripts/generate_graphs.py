import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("results/graphs", exist_ok=True)

df = pd.read_csv("data/raw/peak_s0_varied.csv")
df = df[df["is_warmup"] == False].copy()

metrics = {
    "waiting_time_s": ("Average Recorded Waiting Time", "Seconds"),
    "queue_length_m": ("Average Queue Length", "Metres"),
    "mean_speed_mps": ("Average Vehicle Speed", "m/s"),
    "mean_delay_s": ("Average Recorded Delay", "Seconds"),
    "throughput": ("Average Recorded Throughput", "Vehicles per interval")
}

labels = (
    df[["junction_id", "junction_short"]]
    .drop_duplicates()
    .set_index("junction_id")["junction_short"]
    .to_dict()
)

for column, (title, ylabel) in metrics.items():
    values = df.groupby("junction_id")[column].mean()
    values.index = [labels.get(x, x) for x in values.index]

    ax = values.plot(kind="bar", figsize=(10, 5))
    ax.set_title(title + " - Peak Scenario")
    ax.set_xlabel("Junction")
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(f"results/graphs/{column}.png", dpi=300)
    plt.close()

plt.figure(figsize=(11, 6))

for junction, group in df.groupby("junction_short"):
    group = group.groupby("timestamp")["queue_length_m"].mean().reset_index()
    group = group.sort_values("timestamp")
    plt.plot(group["timestamp"], group["queue_length_m"], label=junction)

plt.title("Queue Length Over Simulation Time - Peak Scenario")
plt.xlabel("Simulation Time (seconds)")
plt.ylabel("Queue Length (metres)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/graphs/queue_over_time.png", dpi=300)
plt.close()

print("GENERATED GRAPHS:")
for file in sorted(os.listdir("results/graphs")):
    print(file)
