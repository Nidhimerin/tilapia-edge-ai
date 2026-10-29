import sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt

src, dst, feed = sys.argv[1], sys.argv[2], float(sys.argv[3])
df = pd.read_csv(src)
base = df.time_s < 600

# Shake rule, fixed before looking at results: median + 6 MAD of baseline global_shift
g = df.global_shift
med = g[base].median()
mad = (g[base] - med).abs().median() * 1.4826
thr = med + 6 * mad
shake = g > thr
step = df.time_s.diff().median()
pad = int(round(1.0 / step))                      # 1 s either side
shake = shake.rolling(2 * pad + 1, center=True, min_periods=1).max().astype(bool)
print(f"threshold {thr:.3f}; flagged {shake.sum()} of {len(df)} rows "
      f"({shake[base].sum()} in baseline)")
print("flagged spans (s):")
edges = shake.astype(int).diff().fillna(0)
starts = df.time_s[edges == 1].tolist(); ends = df.time_s[edges == -1].tolist()
print(list(zip(starts, ends)))

for c in ["diff_near", "diff_away", "flow_near", "flow_away"]:
    s = df[c].where(~shake)
    df[c + "_z"] = (s - s[base].mean()) / s[base].std()

d = df[(df.time_s > feed - 60) & (df.time_s < feed + 180)]
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
for c in ["diff_near_z", "diff_away_z"]: ax[0].plot(d.time_s, d[c], label=c)
for c in ["flow_near_z", "flow_away_z"]: ax[1].plot(d.time_s, d[c], label=c)
for a in ax: a.axvline(feed, ls="--", c="gray"); a.legend()
plt.savefig(dst.replace(".csv", ".png"), dpi=120)
df.assign(shake=shake).to_csv(dst, index=False)