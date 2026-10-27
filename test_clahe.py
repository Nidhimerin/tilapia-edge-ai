import csv
from pathlib import Path
import cv2
import pandas as pd
from ultralytics import YOLO

frames = Path("/Users/nileena/Desktop/Fish_Labels/frames_test")
model = YOLO("best (3).pt")
hand = pd.read_csv("data/hand_counts.csv")
hand["image"] = hand["image"].map(lambda x: Path(x).name)
truth = dict(zip(hand["image"], hand["count"]))

out = Path("outputs/clahe")
out.mkdir(parents=True, exist_ok=True)

rows = []
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

for p in sorted(frames.rglob("*")):
    if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue

    img = cv2.imread(str(p))
    if img is None:
        print("Skipping unreadable image:", p)
        continue

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    enhanced = cv2.merge([clahe.apply(l), a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    cv2.imwrite(str(out / p.name), enhanced)

    result = model.predict(
        enhanced, conf=0.15, iou=0.4, imgsz=640, verbose=False
    )[0]
    predicted = len(result.boxes)
    actual = truth.get(p.name)

    rows.append({
        "image": p.name,
        "hand": actual,
        "clahe_prediction": predicted,
        "clahe_absolute_error":
            abs(predicted - actual) if actual is not None else ""
    })

df = pd.DataFrame(rows)
df.to_csv(out / "clahe_counts.csv", index=False)

valid = df.dropna(subset=["hand"])
valid["clahe_absolute_error"] = pd.to_numeric(
    valid["clahe_absolute_error"]
)
print(valid.to_string(index=False))

if len(valid):
    print(f"\nCLAHE MAE: {valid['clahe_absolute_error'].mean():.2f} fish")
    print("Frames evaluated:", len(valid))

print("\nProcessed images:", out)
print("Counts CSV:", out / "clahe_counts.csv")
