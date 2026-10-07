import argparse, csv, cv2
from ultralytics import YOLO, YOLOWorld

def glare_score(frame, thr=240):
    """Fraction of near-saturated pixels (whole frame for now; restrict to water ROI later)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float((gray >= thr).mean())

def main(a):
    if "world" in a.weights:
        model = YOLOWorld(a.weights)
        model.set_classes(["fish"])
    else:
        model = YOLO(a.weights)

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(a.out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps / a.stride, (w, h))

    with open(a.out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "time_s", "fish_count", "glare_score"])
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % a.stride == 0:
                r = model.predict(frame, conf=a.conf, iou=a.iou, imgsz=640, verbose=False)[0]
                wr.writerow([i, round(i / fps, 2), len(r.boxes), round(glare_score(frame), 4)])
                out.write(r.plot())
            i += 1
    cap.release(); out.release()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--weights", default="models/best.pt")  # or yolov8s-worldv2.pt
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--iou", type=float, default=0.4)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--out_video", default="out_annotated.mp4")
    p.add_argument("--out_csv", default="out_counts.csv")
    main(p.parse_args())