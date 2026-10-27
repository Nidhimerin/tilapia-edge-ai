import cv2
from pathlib import Path

image_path = Path(
    "/Users/nileena/Desktop/Fish_Labels/frames_test/"
    "VID-20260908-WA0065__292.jpg"
)

img = cv2.imread(str(image_path))
if img is None:
    raise SystemExit(f"Cannot open image: {image_path}")

H, W = img.shape[:2]
scale = min(1200 / W, 700 / H, 1.0)
small = cv2.resize(img, None, fx=scale, fy=scale)

x, y, w, h = cv2.selectROI(
    "Select water region; press ENTER to confirm",
    small,
    showCrosshair=True
)
cv2.destroyAllWindows()

if w == 0 or h == 0:
    raise SystemExit("No region selected. Run again.")

x1 = round(x / scale)
y1 = round(y / scale)
x2 = round((x + w) / scale)
y2 = round((y + h) / scale)

print(f"ROI coordinates: {x1} {y1} {x2} {y2}")
print(f"Image dimensions: {W} x {H}")
print(f"Use with glare_score.py: --roi {x1} {y1} {x2} {y2}")
