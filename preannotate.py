import glob, os
from ultralytics import YOLO

model = YOLO("models/best.pt")
for img in sorted(glob.glob("frames_train/*.jpg")):
    r = model.predict(img, conf=0.15, iou=0.4, imgsz=640, verbose=False)[0]
    lines = []
    for b in r.boxes:
        x, y, w, h = b.xywhn[0].tolist()
        lines.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    open(img.replace(".jpg", ".txt"), "w").write("\n".join(lines))
print("done")