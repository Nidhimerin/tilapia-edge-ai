import argparse
import time
from pathlib import Path
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="models/best_finetuned.pt")
ap.add_argument("--frames", default="frames_test")
ap.add_argument("--imgsz", type=int, default=640)
a = ap.parse_args()

m = YOLO(a.weights)

imgs = [
    str(p)
    for p in Path(a.frames).rglob("*")
    if p.suffix.lower() in {".jpg", ".png", ".jpeg"}
]

m.predict(imgs[0], imgsz=a.imgsz, verbose=False)

t = []

for p in imgs:
    s = time.perf_counter()
    m.predict(p, imgsz=a.imgsz, verbose=False)
    t.append(time.perf_counter() - s)

print(
    f"{len(t)} frames  "
    f"mean {1000 * sum(t) / len(t):.1f} ms  "
    f"FPS {len(t) / sum(t):.2f}"
)

try:
    print(
        "CPU temp C:",
        int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000
    )
except OSError:
    print("CPU temp: not available (not a Pi)")