import argparse, csv, os, time
from pathlib import Path
import cv2
import numpy as np
import ncnn

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--model", default="/home/piadmin/pi_deploy/models/best (2)_ncnn_model")
ap.add_argument("--every", type=float, default=1.0, help="analyse one frame every N seconds of video")
ap.add_argument("--max-sec", type=float, default=0, help="stop after N seconds of video (0 = whole video)")
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--nms", type=float, default=0.45)
a = ap.parse_args()

LOGS = Path("/home/piadmin/tank/logs")
LOGS.mkdir(parents=True, exist_ok=True)
SIZE = 640

def cpu_temp():
    try:
        return int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000
    except OSError:
        return -1

# Load model (same calls as your test script)
net = ncnn.Net()
net.opt.num_threads = 4
model_dir = Path(a.model)
assert net.load_param(str(model_dir / "model.ncnn.param")) == 0, "param load failed"
assert net.load_model(str(model_dir / "model.ncnn.bin")) == 0, "bin load failed"

def detect(frame):
    rgb = np.ascontiguousarray(cv2.cvtColor(cv2.resize(frame, (SIZE, SIZE)), cv2.COLOR_BGR2RGB))
    mat = ncnn.Mat.from_pixels(rgb, ncnn.Mat.PixelType.PIXEL_RGB, SIZE, SIZE)
    mat.substract_mean_normalize([], [1/255.0, 1/255.0, 1/255.0])
    ex = net.create_extractor()
    ex.input("in0", mat)
    ret, out = ex.extract("out0")
    if ret != 0:
        return [], []
    arr = np.array(out)                      # shape (5, 8400)
    scores = arr[4]
    keep = np.isfinite(scores) & (scores >= a.conf)
    if not keep.any():
        return [], []
    cx, cy, bw, bh = arr[0][keep], arr[1][keep], arr[2][keep], arr[3][keep]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], axis=1).astype(float).tolist()
    confs = scores[keep].astype(float).tolist()
    idx = cv2.dnn.NMSBoxes(boxes, confs, a.conf, a.nms)
    idx = [] if idx is None or len(idx) == 0 else np.array(idx).reshape(-1).tolist()
    return [boxes[i] for i in idx], [confs[i] for i in idx]

cap = cv2.VideoCapture(a.video)
src_fps = cap.get(cv2.CAP_PROP_FPS) or 10
stride = max(1, int(round(src_fps * a.every)))
name = os.path.basename(a.video)

counts_path = LOGS / "counts.csv"
new = not counts_path.exists()
f = open(counts_path, "a", newline="")
w = csv.writer(f)
if new:
    w.writerow(["video", "video_time_s", "count", "mean_conf", "infer_ms", "cpu_temp_c", "wall_time"])

i, n, total_ms = 0, 0, 0.0
t_run = time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    vt = i / src_fps
    if a.max_sec and vt > a.max_sec:
        break
    if i % stride == 0:
        t0 = time.perf_counter()
        boxes, confs = detect(frame)
        ms = (time.perf_counter() - t0) * 1000
        w.writerow([name, round(vt, 1), len(boxes),
                    round(float(np.mean(confs)), 3) if confs else 0,
                    round(ms, 1), cpu_temp(), time.strftime("%Y-%m-%d %H:%M:%S")])
        f.flush()
        # annotated frame for the dashboard (visual only, stretched-image mapping)
        sx, sy = frame.shape[1] / SIZE, frame.shape[0] / SIZE
        for (x, y, bw, bh), c in zip(boxes, confs):
            cv2.rectangle(frame, (int(x * sx), int(y * sy)), (int((x + bw) * sx), int((y + bh) * sy)), (0, 255, 0), 2)
        cv2.putText(frame, f"count: {len(boxes)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imwrite(str(LOGS / "latest.jpg"), frame)
        n += 1
        total_ms += ms
        if n % 20 == 0:
            print(f"t={vt:.0f}s  count={len(boxes)}  {ms:.0f} ms  {cpu_temp():.1f} C", flush=True)
    i += 1
f.close()

if n:
    avg = total_ms / n
    print(f"\n{name}: {n} frames | avg {avg:.1f} ms | {1000/avg:.2f} FPS | end temp {cpu_temp():.1f} C | wall {time.time()-t_run:.0f}s")
    bp = LOGS / "benchmark.csv"
    newb = not bp.exists()
    with open(bp, "a", newline="") as bf:
        bw = csv.writer(bf)
        if newb:
            bw.writerow(["video", "imgsz", "threads", "frames", "avg_ms", "fps", "end_temp_c"])
        bw.writerow([name, SIZE, 4, n, round(avg, 1), round(1000 / avg, 2), cpu_temp()])
