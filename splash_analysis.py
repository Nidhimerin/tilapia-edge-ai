import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

VIDEO = "feeding_videos/pi_feeding_test.mp4"
FEED_EVENTS = [670, 680]  # 11:10 and 11:20
SAMPLE_FPS = 2
OUT = Path("out_splash_analysis")
OUT.mkdir(exist_ok=True)

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise SystemExit(f"Cannot open video: {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
if fps <= 0:
    raise SystemExit("Invalid video FPS.")

# Coordinates for original 1280x720 video.
# Region A: near food-drop / water-return pipe.
# Region B: away from the outlet.
regions = {
    "near_outlet": (180, 260, 480, 650),
    "away_from_outlet": (520, 250, 1180, 650),
}

step = max(1, round(fps / SAMPLE_FPS))
previous = {}
rows = []
frame_index = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    if frame_index % step == 0:
        time_s = frame_index / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        record = {"time_s": time_s}

        for name, (x1, y1, x2, y2) in regions.items():
            water = gray[y1:y2, x1:x2]

            if name in previous:
                diff = cv2.absdiff(water, previous[name])
                record[name] = float(np.mean(diff > 20))

            previous[name] = water.copy()

        if len(record) == 3:
            rows.append(record)

    frame_index += 1

cap.release()

if not rows:
    raise SystemExit("No measurements generated.")

times = np.array([r["time_s"] for r in rows])
near = np.array([r["near_outlet"] for r in rows])
away = np.array([r["away_from_outlet"] for r in rows])

csv_path = OUT / "splash_scores.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["time_s", "near_outlet", "away_from_outlet"]
    )
    writer.writeheader()
    writer.writerows(rows)

# Plot raw scores and a 3-second moving average.
window = max(1, round(SAMPLE_FPS * 3))
kernel = np.ones(window) / window
near_smooth = np.convolve(near, kernel, mode="same")
away_smooth = np.convolve(away, kernel, mode="same")

plt.figure(figsize=(13, 6))
plt.plot(times / 60, near_smooth, label="Near outlet (smoothed)")
plt.plot(times / 60, away_smooth, label="Away from outlet (smoothed)")

for i, event in enumerate(FEED_EVENTS, start=1):
    plt.axvline(event / 60, linestyle="--",
                label=f"Feeding event {i} ({event // 60}:{event % 60:02d})")

plt.xlabel("Time (minutes)")
plt.ylabel("Changed-pixel fraction")
plt.title("Pi Camera: Surface Disturbance Around Feeding")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "splash_curve_regions.png", dpi=160)
plt.close()

print(f"Measurements: {len(rows)}")
print(f"CSV: {csv_path}")
print(f"Plot: {OUT / 'splash_curve_regions.png'}")

# Compare 10 seconds before and after each event.
# These windows overlap because the events are only 10 seconds apart.
for i, event in enumerate(FEED_EVENTS, start=1):
    before_mask = (times >= event - 10) & (times < event)
    after_mask = (times >= event) & (times < event + 10)

    print(f"\nEvent {i} at {event}s:")
    for name, values in [
        ("near outlet", near),
        ("away from outlet", away),
    ]:
        before = values[before_mask]
        after = values[after_mask]

        if len(before) and len(after):
            print(
                f"  {name}: before={before.mean():.4f}, "
                f"after={after.mean():.4f}, "
                f"change={after.mean() - before.mean():+.4f}"
            )
os.startfile(str((OUT / "splash_curve_regions.png").resolve()))