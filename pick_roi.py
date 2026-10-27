import sys, cv2

if len(sys.argv) < 2:
    print("Usage: python pick_roi.py <video_path>")
    sys.exit(1)

video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Cannot open video: {video_path}")
    sys.exit(1)

cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
ok, frame = cap.read()
if not ok:
    print("Could not read frame at index 30.")
    sys.exit(1)

H, W = frame.shape[:2]
scale = min(1200 / W, 700 / H, 1.0)
small = cv2.resize(frame, None, fx=scale, fy=scale)

x, y, w, h = cv2.selectROI("Drag a box over the water, press ENTER", small, showCrosshair=True)
cv2.destroyAllWindows()

if w == 0 or h == 0:
    print("No box selected, run again.")
else:
    x1, y1 = int(round(x / scale)), int(round(y / scale))
    x2, y2 = int(round((x + w) / scale)), int(round((y + h) / scale))
    print(f"--roi {x1},{y1},{x2},{y2}")
