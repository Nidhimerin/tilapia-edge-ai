# compare_models.py - baseline vs fine-tuned models on held-out test frames
# usage: python compare_models.py [frames_dir] [hand_counts_csv] [conf]
import sys, csv
from pathlib import Path
from ultralytics import YOLO

frames_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("frames_test")
hand_csv   = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/hand_counts.csv")
conf       = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15

models = {
    "baseline":     "models/best.pt",
    "finetuned":    "models/best (1).pt",
    "finetuned_v2": "models/best (2).pt",
}

imgs = sorted(p for p in frames_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
if not imgs:
    sys.exit(f"No images in {frames_dir}")
for w in models.values():
    if not Path(w).exists():
        sys.exit(f"Missing weights file: {w}")

out = Path("outputs/compare"); out.mkdir(parents=True, exist_ok=True)

counts = {p.name: {} for p in imgs}
for name, w in models.items():
    print("running", name)
    m = YOLO(w)
    save_dir = out / name; save_dir.mkdir(exist_ok=True)
    for p in imgs:
        r = m.predict(str(p), conf=conf, iou=0.4, imgsz=640, verbose=False)[0]
        counts[p.name][name] = len(r.boxes)
        r.save(filename=str(save_dir / p.name))   # boxed image

hand = {}
if hand_csv.exists():
    with open(hand_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["count"].strip() != "":
                hand[Path(row["image"]).name] = float(row["count"])
else:
    print(f"(no {hand_csv} yet - writing predicted counts only)")

with open(out / "counts.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["image", *models, "hand"])
    for n, c in counts.items():
        wr.writerow([n, *[c[k] for k in models], hand.get(n, "")])

print(f"\n{len(imgs)} images, conf={conf}")
for name in models:
    pairs = [(counts[n][name], hand[n]) for n in counts if n in hand]
    if not pairs:
        print(f"{name}: mean count {sum(c[name] for c in counts.values())/len(counts):.1f} (no hand counts)")
        continue
    mae = sum(abs(a - b) for a, b in pairs) / len(pairs)
    bias = sum(a - b for a, b in pairs) / len(pairs)
    pct = [abs(a - b) / b * 100 for a, b in pairs if b > 0]
    print(f"{name}: n={len(pairs)}  MAE={mae:.2f}  bias={bias:+.2f}  "
          f"mean %error={sum(pct)/max(len(pct),1):.1f}%")
print(f"\nBoxed images in {out}, table in {out}\\counts.csv")