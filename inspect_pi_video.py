
import cv2
from pathlib import Path

video = Path("feeding_videos/pi_feeding_test.mp4")
output = Path("pi_inspection_frames")
output.mkdir(exist_ok=True)

cap = cv2.VideoCapture(str(video))

if not cap.isOpened():
    raise SystemExit(f"Cannot open video: {video}")

# Assumes feed first touches the water at 11 min 10 sec.
times = [640, 655, 665, 670, 675, 685, 700]

for seconds in times:
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read()

    if ok:
        filename = output / f"frame_{seconds}s.jpg"
        cv2.imwrite(str(filename), frame)
        print(f"Saved {filename}")
    else:
        print(f"Could not read frame at {seconds}s")

cap.release()
