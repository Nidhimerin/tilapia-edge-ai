import csv, argparse
from pathlib import Path
import cv2, numpy as np

def glare_metrics(img, roi=None):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if roi:
        x1, y1, x2, y2 = roi
        g = g[y1:y2, x1:x2]

    return {
        "frac_ge240": float((g >= 240).mean()),
        "frac_ge220": float((g >= 220).mean()),
        "p99": float(np.percentile(g, 99))
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="frames_test")
    ap.add_argument("--roi", nargs=4, type=int,
                    help="x1 y1 x2 y2 of the water area")
    a = ap.parse_args()

    Path("outputs").mkdir(exist_ok=True)

    with open("outputs/glare_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "frac_ge240", "frac_ge220", "p99"])

        for p in sorted(Path(a.frames).rglob("*")):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                m = glare_metrics(cv2.imread(str(p)), a.roi)
                w.writerow([
                    p.name,
                    f"{m['frac_ge240']:.5f}",
                    f"{m['frac_ge220']:.5f}",
                    f"{m['p99']:.1f}"
                ])

    print("wrote outputs/glare_scores.csv")