#!/usr/bin/env python3
"""
plot_pi_evidence.py - summary stats and plots from the Pi logs.

Run on your PC after copying the Pi evidence folder:
  python plot_pi_evidence.py pi_evidence

Reads  : health.csv, counts.csv, benchmark.csv (inside the folder given)
Writes : health_plot.png, latency_vs_temp.png, pi_summary.txt (same folder)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "pi_evidence")
lines = []


def say(s=""):
    print(s)
    lines.append(s)


# ---------------- health log ----------------
h = pd.read_csv(folder / "health.csv", on_bad_lines="skip")
h["timestamp"] = pd.to_datetime(h["timestamp"], errors="coerce")
h = h.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
for c in ["cpu_temp_c", "cpu_pct", "cpu_mhz", "ram_pct", "uptime_s"]:
    h[c] = pd.to_numeric(h[c], errors="coerce")
h["thr"] = h["throttled_hex"].apply(lambda x: int(str(x), 16) if pd.notna(x) else np.nan)

gap = h["timestamp"].diff().dt.total_seconds()
logged_h = gap[gap < 60].sum() / 3600
reboots = int((h["uptime_s"].diff() < 0).sum())

say("== Health log ==")
say(f"rows: {len(h)}  span: {h.timestamp.min()} -> {h.timestamp.max()}")
say(f"continuously logged hours (gaps <60 s): {logged_h:.1f}")
say(f"gaps >60 s: {int((gap > 60).sum())}  (largest {gap.max()/3600:.1f} h)")
say(f"reboots detected (uptime reset): {reboots}")
say(f"temp C  min/mean/max: {h.cpu_temp_c.min():.1f} / {h.cpu_temp_c.mean():.1f} / {h.cpu_temp_c.max():.1f}")
say(f"cpu %   mean/max: {h.cpu_pct.mean():.1f} / {h.cpu_pct.max():.1f}   ram % max: {h.ram_pct.max():.1f}")
say("throttled_hex counts:")
for k, v in h["throttled_hex"].value_counts().items():
    say(f"   {k}: {v} rows ({100*v/len(h):.1f}%)")
flags = {0x1: "under-voltage now", 0x2: "arm freq capped now", 0x4: "throttled now",
         0x8: "soft temp limit now", 0x10000: "under-voltage occurred",
         0x20000: "freq cap occurred", 0x40000: "throttling occurred",
         0x80000: "soft temp limit occurred"}
anyflag = h["thr"].fillna(0).astype(int)
for bit, name in flags.items():
    n = int(((anyflag & bit) > 0).sum())
    if n:
        say(f"   bit {hex(bit)} ({name}): {n} rows")

fig, ax = plt.subplots(3, 1, sharex=True, figsize=(11, 7))
ax[0].plot(h.timestamp, h.cpu_temp_c, lw=0.8)
ax[0].axhline(80, ls="--", c="r", lw=0.8)   # Pi 4 soft-limit reference
ax[0].set_ylabel("CPU temp (C)")
ax[1].plot(h.timestamp, h.cpu_pct, lw=0.8, label="CPU %")
ax[1].plot(h.timestamp, h.ram_pct, lw=0.8, label="RAM %")
ax[1].legend(loc="upper right"); ax[1].set_ylabel("%")
ax[2].plot(h.timestamp, h.cpu_mhz, lw=0.8)
ax[2].set_ylabel("CPU MHz")
throttled_now = (anyflag & 0xF) > 0
if throttled_now.any():
    for a in ax:
        a.fill_between(h.timestamp, 0, 1, where=throttled_now, transform=a.get_xaxis_transform(),
                       color="red", alpha=0.2)
fig.suptitle("Raspberry Pi 4 health log (red shading = throttled/under-voltage now)")
plt.tight_layout()
plt.savefig(folder / "health_plot.png", dpi=130)
plt.close()

# ---------------- benchmark ----------------
say("\n== Benchmark (benchmark.csv) ==")
b = pd.read_csv(folder / "benchmark.csv")
say(b.to_string(index=False))

# ---------------- latency vs temperature ----------------
c = pd.read_csv(folder / "counts.csv")
c = c.dropna(subset=["infer_ms", "cpu_temp_c"])
say("\n== Inference latency (counts.csv) ==")
for name, g in c.groupby("video"):
    say(f"{name}: n={len(g)}  infer_ms median={g.infer_ms.median():.0f}  "
        f"p95={g.infer_ms.quantile(.95):.0f}  temp {g.cpu_temp_c.min():.0f}-{g.cpu_temp_c.max():.0f} C")
r = np.corrcoef(c.cpu_temp_c, c.infer_ms)[0, 1]
say(f"correlation(temp, infer_ms) = {r:.2f}  (n={len(c)})")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(c.cpu_temp_c, c.infer_ms, s=6, alpha=0.5)
ax[0].set_xlabel("CPU temp (C)"); ax[0].set_ylabel("inference (ms)")
ax[0].set_title("Latency vs temperature")
for name, g in c.groupby("video"):
    ax[1].plot(g.video_time_s, g.infer_ms.rolling(15, min_periods=1).median(), label=name)
ax[1].set_xlabel("video time (s)"); ax[1].set_ylabel("inference, rolling median (ms)")
ax[1].legend(); ax[1].set_title("Latency over a run")
plt.tight_layout()
plt.savefig(folder / "latency_vs_temp.png", dpi=130)
plt.close()

(folder / "pi_summary.txt").write_text("\n".join(lines))
say("\nwrote health_plot.png, latency_vs_temp.png, pi_summary.txt")