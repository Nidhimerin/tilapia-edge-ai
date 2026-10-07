import csv, glob, os, subprocess, sys
import pandas as pd

os.makedirs("outputs", exist_ok=True)
videos = sorted(glob.glob("footage/videos/*.mp4") + glob.glob("footage/videos/*.mov"))
print(f"Found {len(videos)} videos")

for v in videos:
    name = os.path.splitext(os.path.basename(v))[0]
    subprocess.run([
        sys.executable, "run_demo.py", "--video", v,
        "--out_video", f"outputs/{name}_annotated.mp4",
        "--out_csv", f"outputs/{name}_counts.csv",
        "--stride", "10",
    ], check=True)

rows = []
for v in videos:
    name = os.path.splitext(os.path.basename(v))[0]
    df = pd.read_csv(f"outputs/{name}_counts.csv")
    rows.append({
        "clip": name,
        "frames_sampled": len(df),
        "mean_glare": round(df["glare_score"].mean(), 4),
        "mean_count": round(df["fish_count"].mean(), 1),
        "max_count": int(df["fish_count"].max()),
    })

summary = pd.DataFrame(rows).sort_values("mean_glare")
summary.to_csv("outputs/clip_summary.csv", index=False)
print(summary.to_string(index=False))