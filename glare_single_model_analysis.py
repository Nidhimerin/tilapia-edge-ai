import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

counts = pd.read_csv("outputs/compare/single_model_counts.csv")
glare = pd.read_csv("outputs/glare_scores.csv")
manual = pd.read_csv("glare_manual.csv")

df = counts.merge(glare, on="image", validate="one_to_one")
df = df.merge(manual, on="image", validate="one_to_one")

if len(df) != 6:
    raise ValueError(f"Expected 6 matched frames, got {len(df)}")

df["frame"] = df["image"].str.extract(r"__(\d+)\.")[0]
df["abs_error"] = (
    df["best_3_prediction"] - df["hand"]
).abs()

out = Path("outputs")
out.mkdir(exist_ok=True)
df.to_csv(out / "glare_vs_error_single_model.csv", index=False)

def spearman(x, y):
    return x.rank().corr(y.rank())

print("\nPer-frame results:")
print(df[[
    "frame", "hand", "best_3_prediction", "abs_error",
    "glare_rating", "frac_ge240", "p99"
]].to_string(index=False))

print("\nPreliminary correlations (n=6; descriptive only):")
for col in ["glare_rating", "frac_ge240", "frac_ge220", "p99"]:
    print(
        f"{col:14s} Spearman vs absolute error: "
        f"{spearman(df[col], df['abs_error']):+.2f}"
    )

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

ax[0].scatter(df["p99"], df["abs_error"], s=65)
ax[1].scatter(df["glare_rating"], df["abs_error"], s=65)

for _, r in df.iterrows():
    ax[0].annotate(r["frame"], (r["p99"], r["abs_error"]),
                   xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax[1].annotate(r["frame"], (r["glare_rating"], r["abs_error"]),
                   xytext=(4, 4), textcoords="offset points", fontsize=8)

ax[0].set_xlabel("99th-percentile grayscale brightness (p99)")
ax[1].set_xlabel("Manual glare rating (0–2)")
ax[1].set_xticks([0, 1, 2])

for a in ax:
    a.set_ylabel("Absolute counting error (fish)")
    a.grid(alpha=0.3)

fig.suptitle("Glare vs fish-counting error — one model, 6 frames")
fig.tight_layout()
fig.savefig(out / "glare_vs_error_single_model.png",
            dpi=160, bbox_inches="tight")

print("\nSaved:")
print("outputs/glare_vs_error_single_model.csv")
print("outputs/glare_vs_error_single_model.png")
