"""
Tilapia Fish Detector — Streamlit App
======================================
Tests the fine-tuned YOLO checkpoint on uploaded images.
Optionally accepts ground-truth YOLO annotation files
and computes TP, FP, FN, Precision, Recall, and mAP@50.

Usage:
    streamlit run app.py
"""

import io
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# ─────────────────────────── page config ────────────────────────────
st.set_page_config(
    page_title="Tilapia Fish Detector",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── custom CSS ─────────────────────────────
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { background: #0f1b2d; }
    [data-testid="stMetric"] {
        background: #1a2d45;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #00b4d8;
    }
    [data-testid="stMetricLabel"] { color: #90e0ef !important; font-size: 0.78rem; }
    [data-testid="stMetricValue"] { color: #caf0f8 !important; font-size: 1.5rem; }
    h2 { color: #00b4d8; }
    .eval-warning {
        background: #2d1b00;
        border-left: 4px solid #ff9800;
        border-radius: 6px;
        padding: 10px 14px;
        color: #ffd180;
        font-size: 0.87rem;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────── helpers ────────────────────────────────

@st.cache_resource(show_spinner="Loading YOLO checkpoint …")
def load_model(weights_path: str) -> YOLO:
    return YOLO(weights_path)


def run_inference(model: YOLO, img_bgr: np.ndarray, conf: float):
    """Return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    results = model.predict(img_bgr, conf=conf, imgsz=640, verbose=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return results[0], elapsed_ms


def annotate_image(result) -> np.ndarray:
    """Draw bounding boxes + confidence labels on the image."""
    img = result.orig_img.copy()
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_val = float(box.conf[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 210, 255), 2)
        label = f"fish {conf_val:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 210, 255), -1)
        cv2.putText(img, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def parse_yolo_label(txt: str, img_w: int, img_h: int) -> np.ndarray:
    """
    Parse a YOLO-format annotation string.
    Returns (N, 4) float array of absolute [x1, y1, x2, y2] boxes.
    Line format: <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
    """
    boxes = []
    for line in txt.strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = map(float, parts[:5])
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h
        boxes.append([x1, y1, x2, y2])
    return np.array(boxes, dtype=float) if boxes else np.zeros((0, 4))


def iou_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between gt (M,4) and pred (N,4)."""
    if gt.shape[0] == 0 or pred.shape[0] == 0:
        return np.zeros((gt.shape[0], pred.shape[0]))
    iou = np.zeros((len(gt), len(pred)))
    for i, g in enumerate(gt):
        ix1 = np.maximum(g[0], pred[:, 0])
        iy1 = np.maximum(g[1], pred[:, 1])
        ix2 = np.minimum(g[2], pred[:, 2])
        iy2 = np.minimum(g[3], pred[:, 3])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        area_g = (g[2] - g[0]) * (g[3] - g[1])
        area_p = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        union = area_g + area_p - inter
        iou[i] = np.where(union > 0, inter / union, 0.0)
    return iou


def match_predictions(gt_boxes: np.ndarray, pred_boxes: np.ndarray,
                       iou_thresh: float = 0.5):
    """Greedy IoU matching at a single threshold. Returns (tp, fp, fn)."""
    if gt_boxes.shape[0] == 0 and pred_boxes.shape[0] == 0:
        return 0, 0, 0
    if gt_boxes.shape[0] == 0:
        return 0, len(pred_boxes), 0
    if pred_boxes.shape[0] == 0:
        return 0, 0, len(gt_boxes)
    iou = iou_matrix(gt_boxes, pred_boxes)
    matched_gt, matched_pred = set(), set()
    pairs = sorted(np.argwhere(iou >= iou_thresh),
                   key=lambda rc: -iou[rc[0], rc[1]])
    for r, c in pairs:
        if r not in matched_gt and c not in matched_pred:
            matched_gt.add(r)
            matched_pred.add(c)
    tp = len(matched_gt)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def compute_ap50(gt_boxes: np.ndarray, pred_boxes: np.ndarray,
                 pred_scores: np.ndarray) -> float:
    """11-point interpolated AP@IoU=0.50 for a single image."""
    if gt_boxes.shape[0] == 0 and pred_boxes.shape[0] == 0:
        return 1.0   # no GT and no predictions — vacuously perfect
    if gt_boxes.shape[0] == 0:
        return 0.0   # false positives with no GT
    if pred_boxes.shape[0] == 0:
        return 0.0   # missed all GT
    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    iou = iou_matrix(gt_boxes, pred_boxes)
    matched_gt = set()
    tps, fps = [], []
    for j in range(len(pred_boxes)):
        best_iou = iou[:, j].max() if len(gt_boxes) else 0.0
        best_gt  = int(iou[:, j].argmax()) if len(gt_boxes) else -1
        if best_iou >= 0.50 and best_gt not in matched_gt:
            matched_gt.add(best_gt)
            tps.append(1); fps.append(0)
        else:
            tps.append(0); fps.append(1)
    tp_cum  = np.cumsum(tps)
    fp_cum  = np.cumsum(fps)
    recalls    = tp_cum / max(len(gt_boxes), 1)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-9)
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        p = precisions[recalls >= thr].max() if any(recalls >= thr) else 0.0
        ap += p / 11
    return ap


# ─────────────────────────── sidebar ────────────────────────────────
with st.sidebar:
    st.markdown("## 🐟 Tilapia Fish Detector")
    st.markdown("*Fine-tuned YOLOv8 checkpoint — capstone project*")
    st.divider()

    st.markdown("### ⚙️ Model")
    weights_path = st.text_input(
        "Weights path",
        value="best (3).pt",
        help="Relative to the repo root, e.g. `best (3).pt` or `models/best.pt`.",
    )

    st.markdown("### 🎚️ Inference")
    conf_thresh = st.slider(
        "Confidence threshold",
        min_value=0.01, max_value=0.95, value=0.25, step=0.01,
        help="Detections below this score are suppressed.",
    )

    st.divider()
    st.markdown("### 📊 Evaluation Mode")
    eval_mode = st.toggle("Enable evaluation mode", value=False)
    gt_zip = None
    if eval_mode:
        st.markdown(
            '<div class="eval-warning">'
            "⚠️ <b>Reliable metrics require correct ground-truth annotations.</b> "
            "Upload YOLO <code>.txt</code> files whose filenames match your images "
            "(e.g. <code>fish01.txt</code> for <code>fish01.jpg</code>). "
            "Without matching annotations, evaluation is skipped for that image."
            "</div>",
            unsafe_allow_html=True,
        )
        gt_zip = st.file_uploader(
            "Ground-truth labels (.zip of YOLO .txt files)",
            type=["zip"],
            help="ZIP containing one YOLO-format .txt annotation per image.",
        )

    st.divider()
    st.caption(
        "Model weights are excluded from Git (`*.pt` in `.gitignore`). "
        "Do **not** commit `best (3).pt`."
    )

# ─────────────────────────── main area ──────────────────────────────
st.title("🐟 Tilapia Fish Detector")
st.markdown(
    "Upload one or more fish images. The app runs the fine-tuned YOLO checkpoint "
    "and displays annotated bounding boxes, confidence scores, fish count, "
    "and inference time."
)

# ── verify checkpoint ──
if not Path(weights_path).exists():
    st.error(
        f"❌  Checkpoint **`{weights_path}`** not found.\n\n"
        "**Where to put it:** Copy `best (3).pt` from Google Colab/Drive into "
        "the repository root — the same folder as `app.py`:\n\n"
        "```\n"
        "tilapia-edge-ai/\n"
        "├── app.py\n"
        "├── best (3).pt   ← place it here\n"
        "└── ...\n"
        "```\n\n"
        "Then refresh this page, or update the **Weights path** in the sidebar.\n\n"
        "> **Note:** `*.pt` files are in `.gitignore` — the weights will never be "
        "committed to GitHub."
    )
    st.stop()

model = load_model(weights_path)

# ── image upload ──
uploaded_files = st.file_uploader(
    "Upload fish images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

# ── parse ground-truth labels from ZIP (if supplied) ──
gt_labels: dict = {}
if eval_mode and gt_zip is not None:
    with zipfile.ZipFile(io.BytesIO(gt_zip.read())) as zf:
        for name in zf.namelist():
            p = Path(name)
            if p.suffix.lower() == ".txt":
                gt_labels[p.stem] = zf.read(name).decode("utf-8", errors="ignore")

if not uploaded_files:
    st.info("⬆️  Upload at least one image above to begin.")
    st.stop()

# ── dataset-level accumulators ──
all_tp = all_fp = all_fn = 0
all_ap50: list = []
image_rows: list = []

tabs = st.tabs([f.name for f in uploaded_files])

for tab, uf in zip(tabs, uploaded_files):
    with tab:
        file_bytes = np.frombuffer(uf.read(), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            st.error(f"Could not decode **{uf.name}**.")
            continue
        img_h, img_w = img_bgr.shape[:2]

        # ── run inference ──
        result, elapsed_ms = run_inference(model, img_bgr, conf_thresh)
        pred_boxes_xyxy = (
            result.boxes.xyxy.cpu().numpy()
            if len(result.boxes) else np.zeros((0, 4))
        )
        pred_scores = (
            result.boxes.conf.cpu().numpy()
            if len(result.boxes) else np.zeros(0)
        )
        fish_count = len(result.boxes)

        annotated_bgr = annotate_image(result)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        col_img, col_meta = st.columns([3, 2], gap="large")

        with col_img:
            st.image(annotated_rgb, use_container_width=True,
                     caption=f"Detections — {uf.name}")

        with col_meta:
            st.markdown("#### Detection Metrics")
            m1, m2, m3 = st.columns(3)
            m1.metric("🐟 Fish count", fish_count)
            m2.metric("⏱ Inference", f"{elapsed_ms:.1f} ms")
            m3.metric("🎚 Conf thr.", f"{conf_thresh:.2f}")

            if fish_count:
                rows_data = []
                for i, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    rows_data.append({
                        "#": i + 1,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "Conf": f"{float(box.conf[0]):.3f}",
                    })
                st.dataframe(
                    pd.DataFrame(rows_data).set_index("#"),
                    use_container_width=True,
                )
            else:
                st.info("No fish detected at this threshold. "
                        "Try lowering the confidence slider.")

            # ── per-image evaluation ──
            if eval_mode:
                st.markdown("#### Evaluation (vs. ground truth)")
                stem = Path(uf.name).stem
                if stem in gt_labels:
                    gt_boxes = parse_yolo_label(gt_labels[stem], img_w, img_h)
                    tp, fp, fn = match_predictions(gt_boxes, pred_boxes_xyxy, 0.50)
                    ap50 = compute_ap50(gt_boxes, pred_boxes_xyxy, pred_scores)
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0

                    all_tp += tp; all_fp += fp; all_fn += fn
                    all_ap50.append(ap50)

                    e1, e2, e3 = st.columns(3)
                    e1.metric("TP", tp)
                    e2.metric("FP", fp)
                    e3.metric("FN", fn)
                    e4, e5, e6 = st.columns(3)
                    e4.metric("Precision", f"{prec:.3f}")
                    e5.metric("Recall",    f"{rec:.3f}")
                    e6.metric("AP@50",     f"{ap50:.3f}")

                    image_rows.append({
                        "Image": uf.name,
                        "GT boxes": len(gt_boxes),
                        "Pred boxes": fish_count,
                        "TP": tp, "FP": fp, "FN": fn,
                        "Precision": round(prec, 3),
                        "Recall":    round(rec, 3),
                        "AP@50":     round(ap50, 3),
                    })
                else:
                    st.warning(
                        f"No ground-truth annotation found for **{uf.name}** "
                        f"(expected `{stem}.txt` inside the ZIP). "
                        "Evaluation is skipped for this image."
                    )

# ─────────────────────────── dataset summary ────────────────────────
if eval_mode and image_rows:
    st.divider()
    st.markdown("## 📊 Dataset-Level Evaluation Summary")
    st.markdown(
        '<div class="eval-warning">'
        "⚠️ <b>These metrics are only as reliable as your ground-truth annotations.</b> "
        "Incorrect, incomplete, or mismatched label files will produce misleading "
        "numbers. Always verify your annotations before drawing conclusions."
        "</div>",
        unsafe_allow_html=True,
    )

    dataset_prec  = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    dataset_rec   = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    dataset_map50 = float(np.mean(all_ap50)) if all_ap50 else 0.0

    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Total TP",          all_tp)
    d2.metric("Total FP",          all_fp)
    d3.metric("Total FN",          all_fn)
    d4.metric("Dataset Precision", f"{dataset_prec:.3f}")
    d5.metric("Dataset Recall",    f"{dataset_rec:.3f}")
    st.metric("mAP@50 (mean AP across all images)", f"{dataset_map50:.4f}")

    st.markdown("### Per-image breakdown")
    df_eval = pd.DataFrame(image_rows)
    st.dataframe(df_eval.set_index("Image"), use_container_width=True)

    csv_bytes = df_eval.to_csv(index=False).encode()
    st.download_button(
        "⬇️  Download evaluation CSV",
        data=csv_bytes,
        file_name="eval_results.csv",
        mime="text/csv",
    )

# ─────────────────────────── footer ─────────────────────────────────
st.divider()
st.caption(
    "Tilapia Edge-AI · Capstone project · "
    "Model weights excluded from Git (`*.pt` in `.gitignore`) · "
    "Do **not** commit `best (3).pt` to the repository."
)
