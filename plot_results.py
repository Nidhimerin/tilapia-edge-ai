import sys
import pandas as pd
import matplotlib.pyplot as plt

csv_path, png_path = sys.argv[1], sys.argv[2]
df = pd.read_csv(csv_path)

fig, ax1 = plt.subplots(figsize=(9, 4))
ax1.plot(df["time_s"], df["fish_count"], color="tab:blue", label="Fish count")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Detected fish count", color="tab:blue")

ax2 = ax1.twinx()
ax2.plot(df["time_s"], df["glare_score"], color="tab:red", label="Glare score")
ax2.set_ylabel("Glare score (saturated pixel fraction)", color="tab:red")

plt.title("Fish count and glare over time")
plt.tight_layout()
plt.savefig(png_path, dpi=150)