import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

MODELS = ["baseline", "finetuned", "finetuned_v2"]
counts = pd.read_csv("outputs/compare/counts.csv")
glare = pd.read_csv("outputs/glare_scores.csv")
manual = pd.read_csv("glare_manual.csv")

df = counts.merge(glare, on="image").merge(manual, on="image")
print("rows after merge:", len(df), "(expect 6)")
for m in MODELS:
    df["err_" + m] = df[m] - df["hand"]
    df["abs_" + m] = df["err_" + m].abs()
df["frame"] = df["image"].str.extract(r"__(\d+)\.")[0]
Path("outputs").mkdir(exist_ok=True)
df.to_csv("outputs/glare_vs_error.csv", index=False)

def spearman(a, b):
    return a.rank().corr(b.rank())

print("\nGlare vs absolute error (n=%d, indicative only)" % len(df))
for g in ["p99", "frac_ge240", "glare_rating"]:
    for m in ["finetuned", "finetuned_v2"]:
        print(f"{g:13s} vs {m:13s} pearson {df[g].corr(df['abs_' + m]):+.2f}  spearman {spearman(df[g], df['abs_' + m]):+.2f}")

print("\nAgreement of automatic scores with manual rating (spearman):")
for g in ["p99", "frac_ge240", "frac_ge220"]:
    print(f"{g:11s} {spearman(df[g], df['glare_rating']):+.2f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for m, c in [("finetuned", "tab:orange"), ("finetuned_v2", "tab:green")]:
    ax[0].scatter(df["p99"], df["abs_" + m], label=m, color=c, s=60)
    ax[1].scatter(df["glare_rating"], df["abs_" + m], label=m, color=c, s=60)
for _, r in df.iterrows():
    ax[0].annotate(r["frame"], (r["p99"], r["abs_finetuned_v2"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
ax[0].set_xlabel("Glare score (99th percentile brightness)")
ax[1].set_xlabel("Manual glare rating (0-2)")
ax[1].set_xticks([0, 1, 2])
for a in ax:
    a.set_ylabel("Absolute count error (fish)")
    a.grid(alpha=0.3)
ax[0].legend()
fig.suptitle("Glare vs counting error - preliminary, n=6 frames, 1 clip")
fig.savefig("outputs/glare_vs_error.png", dpi=150, bbox_inches="tight")
print("\nSaved outputs/glare_vs_error.csv and outputs/glare_vs_error.png")


d = pd.read_csv("outputs/glare_vs_error.csv")
print("hand count vs abs error (v2): spearman", round(d["hand"].rank().corr(d["abs_finetuned_v2"].rank()), 2))
print(d[["frame", "hand", "finetuned_v2", "abs_finetuned_v2"]])