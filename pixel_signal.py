
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


COLUMNS = [
    "time_s",
    "flow_near",
    "flow_away",
    "diff_near",
    "diff_away",
    "glare_frac",
    "valid_frac",
    "sharpness",
    "global_shift",
]


def parse_box(value):
    """Parse X,Y,W,H and return x1,y1,x2,y2."""
    try:
        x, y, w, h = [int(v.strip()) for v in value.split(",")]
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            "Box must be X,Y,W,H, e.g. 180,260,300,390"
        ) from exc

    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(
            "X/Y must be non-negative and W/H must be positive"
        )
    return x, y, x + w, y + h


def crop_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Box {box} lies outside the video frame.")
    return x1, y1, x2, y2


def inside_box_mask(shape, box):
    """Boolean mask for a box within the ROI crop."""
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = True
    return mask


def mean_or_nan(values, mask, minimum_valid=0.20):
    selected = mask & np.isfinite(values)
    if selected.size == 0 or selected.mean() < minimum_valid:
        return float("nan")
    return float(np.mean(values[selected]))


def main():
    parser = argparse.ArgumentParser(
        description="Extract pixel motion and image-quality signals."
    )
    parser.add_argument("video", help="Input video path")
    parser.add_argument(
        "--roi", required=True, type=parse_box,
        help="Whole water ROI: X,Y,W,H"
    )
    parser.add_argument(
        "--near", required=True, type=parse_box,
        help="Near-outlet ROI: X,Y,W,H in full-frame coordinates"
    )
    parser.add_argument("--start", type=float, default=0,
                        help="Start time in video seconds")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="Maximum duration to process")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--sample-fps", type=float, default=2.0,
                        help="Sampling rate; default 2 frames/second")
    parser.add_argument("--calib-seconds", type=float, default=0,
                        help="Initial seconds used to calibrate glare threshold")
    args = parser.parse_args()

    if args.start < 0 or args.sample_fps <= 0 or args.calib_seconds < 0:
        parser.error("start/calib-seconds must be >= 0 and sample-fps > 0")
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error("max-seconds must be positive")

    video_path = Path(args.video)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"ERROR: Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise SystemExit("ERROR: Invalid video FPS or frame dimensions.")

    try:
        roi = crop_box(args.roi, width, height)
        near_full = crop_box(args.near, width, height)
    except ValueError as exc:
        cap.release()
        raise SystemExit(f"ERROR: {exc}") from exc

    rx1, ry1, rx2, ry2 = roi
    nx1, ny1, nx2, ny2 = near_full

    if not (rx1 <= nx1 < nx2 <= rx2 and ry1 <= ny1 < ny2 <= ry2):
        cap.release()
        raise SystemExit(
            "ERROR: --near must be fully inside --roi. "
            "Both boxes use full-frame X,Y,W,H coordinates."
        )

    roi_width, roi_height = rx2 - rx1, ry2 - ry1
    near_local = (nx1 - rx1, ny1 - ry1, nx2 - rx1, ny2 - ry1)
    near_mask = inside_box_mask((roi_height, roi_width), near_local)

    start_frame = min(frame_count, int(round(args.start * fps)))
    end_time = (
        args.start + args.max_seconds
        if args.max_seconds is not None
        else frame_count / fps
    )
    end_frame = min(frame_count, int(np.ceil(end_time * fps)))
    step = max(1, int(round(fps / args.sample_fps)))

    if start_frame >= end_frame:
        cap.release()
        raise SystemExit("ERROR: Start time is beyond the available video.")

    # Default bright-pixel glare threshold.
    # Calibration estimates a threshold from an initial video segment.
    glare_threshold = 245.0

    if args.calib_seconds > 0:
        calibration_values = []
        calibration_end = min(
            frame_count, int(round(args.calib_seconds * fps))
        )
        calibration_step = max(1, int(round(fps)))

        for idx in range(0, calibration_end, calibration_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(
                frame[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2GRAY
            )
            calibration_values.append(gray[::4, ::4].ravel())

        if calibration_values:
            pixels = np.concatenate(calibration_values)
            glare_threshold = float(np.clip(
                np.percentile(pixels, 99.5), 235, 250
            ))

    # Process sequentially and calculate optical flow at half resolution.
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    rows = []
    previous_gray = None
    previous_valid = None
    frame_index = start_frame

    scale = 0.5
    small_width = max(1, int(round(roi_width * scale)))
    small_height = max(1, int(round(roi_height * scale)))

    near_small = cv2.resize(
        near_mask.astype(np.uint8),
        (small_width, small_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    away_small = ~near_small

    while frame_index < end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        if (frame_index - start_frame) % step == 0:
            crop = frame[ry1:ry2, rx1:rx2]
            gray_full = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(
                gray_full,
                (small_width, small_height),
                interpolation=cv2.INTER_AREA,
            )

            bad = gray >= glare_threshold
            valid = gray < glare_threshold
            glare_frac = float(1.0 - valid.mean())
            valid_frac = float(valid.mean())
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            global_shift = float("nan")

            row = {
                "time_s": round(frame_index / fps, 3),
                "flow_near": float("nan"),
                "flow_away": float("nan"),
                "diff_near": float("nan"),
                "diff_away": float("nan"),
                "glare_frac": glare_frac,
                "valid_frac": valid_frac,
                "sharpness": sharpness,
                "global_shift": global_shift,
            }

            if previous_gray is not None:
                prev_gray = previous_gray
                valid = ~bad
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                # global (camera) motion = median flow over valid pixels
                gx = float(np.median(flow[..., 0][valid])) if valid.any() else 0.0
                gy = float(np.median(flow[..., 1][valid])) if valid.any() else 0.0
                global_shift = float(np.hypot(gx, gy))

                # flow with the camera motion removed
                mag = np.sqrt((flow[..., 0] - gx) ** 2 + (flow[..., 1] - gy) ** 2)

                # align previous frame to the current one before differencing
                M = np.float32([[1, 0, gx], [0, 1, gy]])
                prev_aligned = cv2.warpAffine(prev_gray, M, (gray.shape[1], gray.shape[0]),
                                              borderMode=cv2.BORDER_REPLICATE)
                diff = cv2.absdiff(gray, prev_aligned).astype(np.float32)
                diff_signal = (diff > 20).astype(np.float32)

                pair_valid = valid & previous_valid
                row["flow_near"] = mean_or_nan(
                    mag, pair_valid & near_small
                )
                row["flow_away"] = mean_or_nan(
                    mag, pair_valid & away_small
                )
                row["diff_near"] = mean_or_nan(
                    diff_signal, pair_valid & near_small
                )
                row["diff_away"] = mean_or_nan(
                    diff_signal, pair_valid & away_small
                )
                row["global_shift"] = global_shift

            rows.append(row)
            previous_gray = gray.copy()
            previous_valid = valid.copy()

        frame_index += 1

    cap.release()

    if not rows:
        raise SystemExit(
            "ERROR: No frames processed. Check start time and video."
        )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}; source FPS: {fps:.3f}")
    print(f"ROI: {args.roi}; near ROI: {args.near}")
    print(f"Glare threshold: {glare_threshold:.1f}")
    print(f"Rows written: {len(rows)}")
    print(
        f"Median glare_frac: "
        f"{np.nanmedian([r['glare_frac'] for r in rows]):.4f}"
    )
    print(
        f"Median valid_frac: "
        f"{np.nanmedian([r['valid_frac'] for r in rows]):.4f}"
    )

    for name in ("flow_near", "flow_away", "diff_near", "diff_away"):
        values = np.array([row[name] for row in rows], dtype=float)
        print(f"NaN fraction {name}: {np.mean(~np.isfinite(values)):.4f}")

    print(f"CSV: {output}")


if __name__ == "__main__":
    main()
