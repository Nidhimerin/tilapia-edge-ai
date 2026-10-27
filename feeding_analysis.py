"""
feeding_analysis.py
One script for the feeding video:
  1. samples frames at a fixed rate
  2. runs your fish detector on each sampled frame
  3. saves frames + YOLO-format pre-label .txt files (for annotation correction)
  4. computes behaviour features per frame
  5. builds the response curve and extracts onset latency, peak, duration, decline rate

Example:
  python feeding_analysis.py --video feeding.mp4 --weights best.pt --feed-time 20 \
      --roi 100,80,1180,700 --top-frac 0.3 --out out_feeding

pip install ultralytics opencv-python pandas numpy matplotlib
"""
import argparse
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO


def parse_box(s, default):
    if not s:
        return default
    v = [int(x) for x in s.split(",")]
    assert len(v) == 4, "box must be x1,y1,x2,y2"
    return tuple(v)


# ---------------------------------------------------------------- extraction + detection
def run_detection(args):
    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stem = os.path.splitext(os.path.basename(args.video))[0]

    rx1, ry1, rx2, ry2 = parse_box(args.roi, (0, 0, W, H))
    diag = float(np.hypot(rx2 - rx1, ry2 - ry1))
    band_y2 = ry1 + args.top_frac * (ry2 - ry1)  # surface band = top part of ROI

    frames_dir = os.path.join(args.out, "frames")
    labels_dir = os.path.join(args.out, "labels")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    step = max(1, round(fps / args.sample_fps))
    rows, prev_centers, sampled = [], None, 0
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % step:
            continue
        t = idx / fps

        res = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        xyxy = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
        centers = np.c_[(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2]
        inside = (centers[:, 0] >= rx1) & (centers[:, 0] <= rx2) & \
                 (centers[:, 1] >= ry1) & (centers[:, 1] <= ry2)
        xyxy, centers = xyxy[inside], centers[inside]
        n = len(centers)

        # features: dispersion and activity are NaN when n < 2
        if n >= 1:
            surf = float(np.mean(centers[:, 1] <= band_y2))
        else:
            surf = np.nan

        if n >= 2:
            disp = float(np.mean(np.linalg.norm(centers - centers.mean(0), axis=1)) / diag)
        else:
            disp = np.nan

        if n >= 2 and prev_centers is not None and len(prev_centers) >= 2:
            d = np.linalg.norm(centers[:, None, :] - prev_centers[None, :, :], axis=2)
            act = float(d.min(axis=1).mean() / diag)  # nearest-neighbour displacement
        else:
            act = np.nan
        prev_centers = centers if n >= 2 else prev_centers

        rows.append(dict(frame=idx, t=t, count=n, surfacing=surf, dispersion=disp, activity=act))

        # save frame + YOLO pre-label for annotation (single class 0)
        if sampled % args.save_every == 0:
            name = f"{stem}_{idx:05d}"
            cv2.imwrite(os.path.join(frames_dir, name + ".jpg"), frame)
            with open(os.path.join(labels_dir, name + ".txt"), "w") as f:
                for x1, y1, x2, y2 in xyxy:
                    f.write(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} "
                            f"{(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")
        sampled += 1
    cap.release()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "features.csv"), index=False)
    print(f"Sampled {len(df)} frames at ~{args.sample_fps} fps, video fps={fps:.1f}")
    return df, (rx1, ry1, rx2, ry2), band_y2


# ---------------------------------------------------------------- response curve
def zscore(series, base_mask):
    base_vals = series[base_mask].dropna()
    mu = base_vals.mean() if len(base_vals) else np.nan
    sd = base_vals.std() if len(base_vals) else np.nan
    return (series - mu) / (sd if sd and sd > 1e-9 else 1.0)


def analyse(df, args):
    df = df.copy()

    # Low-detection frames: derived features set to NaN before smoothing
    low_det = df["count"] < args.min_count
    df.loc[low_det, ["surfacing", "dispersion", "activity"]] = np.nan

    # Smoothing with limited gap interpolation (~1 s limit)
    win = max(1, int(args.smooth_s * args.sample_fps))
    interp_limit = max(1, int(1.0 * args.sample_fps))
    for c in ["surfacing", "dispersion", "activity"]:
        df[c] = df[c].interpolate(
            limit=interp_limit,
            limit_area="inside"
        )
        df[c + "_s"] = df[c].rolling(
            win,
            center=True,
            min_periods=1
        ).mean()

    # Baseline evaluation
    pre_feed_duration = max(0.0, float(args.feed_time - df["t"].min()))
    base = df["t"] < args.feed_time

    out = dict(feed_time=args.feed_time)

    if pre_feed_duration < args.min_baseline_s:
        print(f"WARNING: Pre-feed duration ({pre_feed_duration:.1f} s) is shorter than --min-baseline-s ({args.min_baseline_s:.1f} s). "
              "Baseline is unreliable; onset and thresholds should not be trusted.")
        out["baseline_too_short"] = True
    else:
        out["baseline_too_short"] = False

    if base.sum() < 5:
        print("WARNING: fewer than 5 pre-feed samples; baseline is unreliable. "
              "Onset and thresholds should not be trusted.")
        base = df["t"] < df["t"].min() + 2

    # combined response signal: more surfacing, more activity, tighter school = stronger response
    sig = (zscore(df["surfacing_s"], base)
           + zscore(df["activity_s"], base)
           - zscore(df["dispersion_s"], base)) / 3.0

    # Signal is NaN wherever any of the three smoothed features is NaN
    feat_nan = df["surfacing_s"].isna() | df["activity_s"].isna() | df["dispersion_s"].isna()
    sig[feat_nan] = np.nan
    df["signal"] = sig

    base_sig = sig[base].dropna()
    thr = max(args.k * base_sig.std(), 0.5) if len(base_sig) else 0.5
    if np.isnan(thr):
        thr = 0.5

    out["threshold"] = float(thr)
    out["n_baseline_samples"] = int(base.sum())

    post = df["t"] >= args.feed_time
    t = df["t"].to_numpy()
    s = sig.to_numpy()
    post_idx = np.where(post.to_numpy())[0]

    # Calculate post-feed valid signal fraction and reliability
    n_post = len(post_idx)
    if n_post > 0:
        n_valid_post = int((~np.isnan(s[post_idx])).sum())
        post_feed_valid_fraction = float(n_valid_post / n_post)
    else:
        post_feed_valid_fraction = 0.0

    out["post_feed_valid_fraction"] = round(post_feed_valid_fraction, 4)
    reliable = bool(post_feed_valid_fraction >= 0.7 and not out["baseline_too_short"])
    out["reliable"] = reliable

    if not reliable:
        print("\n" + "!" * 72)
        print("WARNING: Feeding response output is UNRELIABLE!")
        print("Onset, peak, duration, and decline rate should NOT be trusted.")
        if out["baseline_too_short"]:
            print(f" - Baseline too short ({pre_feed_duration:.1f} s < {args.min_baseline_s:.1f} s).")
        if post_feed_valid_fraction < 0.7:
            print(f" - Severe detector dropout: post-feed valid signal fraction is {post_feed_valid_fraction:.1%} (< 70%).")
        print("!" * 72 + "\n")

    # onset: first sustained crossing of threshold after the feeder trigger (NaN-safe)
    m = max(1, int(args.sustain_s * args.sample_fps))
    onset_i = None
    for i in post_idx:
        sub = s[i:i + m]
        if len(sub) == m and not np.isnan(sub).any() and (sub > thr).all():
            onset_i = i
            break
    out["onset_latency_s"] = float(t[onset_i] - args.feed_time) if onset_i is not None else None

    # peak: using nanargmax
    valid_post_s = s[post_idx]
    if len(post_idx) > 0 and not np.isnan(valid_post_s).all():
        pk = post_idx[np.nanargmax(valid_post_s)]
        out["peak_value"] = float(s[pk])
        out["time_to_peak_s"] = float(t[pk] - args.feed_time)

        # duration above frac * peak (contiguous around the peak)
        level = args.frac * s[pk]
        lo = pk
        while lo > 0 and not np.isnan(s[lo - 1]) and s[lo - 1] >= level:
            lo -= 1
        hi = pk
        while hi < len(s) - 1 and not np.isnan(s[hi + 1]) and s[hi + 1] >= level:
            hi += 1
        out["duration_s"] = float(t[hi] - t[lo])

        # decline rate: linear slope from peak to the end of the duration window
        # (falls back to 30 s after the peak if that window is too short)
        end = hi if hi - pk >= 3 else min(len(s) - 1, pk + int(30 * args.sample_fps))
        t_sub = t[pk:end + 1]
        s_sub = s[pk:end + 1]
        valid_fit = ~np.isnan(t_sub) & ~np.isnan(s_sub)
        if valid_fit.sum() >= 3:
            out["decline_rate_per_s"] = float(np.polyfit(t_sub[valid_fit], s_sub[valid_fit], 1)[0])
        else:
            out["decline_rate_per_s"] = None
    else:
        pk = None
        lo = None
        hi = None
        out["peak_value"] = None
        out["time_to_peak_s"] = None
        out["duration_s"] = None
        out["decline_rate_per_s"] = None

    if not reliable:
        out["unvalidated_values"] = {
            "onset_latency_s": out["onset_latency_s"],
            "peak_value": out["peak_value"],
            "time_to_peak_s": out["time_to_peak_s"],
            "duration_s": out["duration_s"],
            "decline_rate_per_s": out["decline_rate_per_s"],
        }

        out["onset_latency_s"] = None
        out["peak_value"] = None
        out["time_to_peak_s"] = None
        out["duration_s"] = None
        out["decline_rate_per_s"] = None

    out["note"] = (
        "Single feeding video: proof of concept only. "
        "The response index is heuristic, not a validated appetite "
        "or satiation measure."
    )

    return df, out, (onset_i, pk, lo, hi)


def plot(df, out, marks, args, band_y2):
    onset_i, pk, lo, hi = marks
    fig, ax = plt.subplots(5, 1, figsize=(9, 11), sharex=True)

    # Subplot 0: raw detection count with low-detection spans shaded
    ax[0].plot(df["t"], df["count"], label="raw count", color="tab:blue")
    ax[0].axhline(args.min_count, ls=":", c="gray", label=f"min-count ({args.min_count})")
    low_mask = df["count"] < args.min_count
    max_c = max(1, int(df["count"].max())) if not df["count"].isna().all() else 10
    ax[0].fill_between(df["t"], 0, max_c + 1, where=low_mask,
                       color="gray", alpha=0.25, step="mid", label=f"count < {args.min_count}")
    ax[0].set_ylabel("fish count")
    ax[0].legend(fontsize=8, loc="upper right")

    # Subplot 1: combined response signal
    ax[1].plot(df["t"], df["signal"], label="response signal (z)", color="tab:blue")
    ax[1].axhline(out["threshold"], ls=":", c="gray", label="onset threshold")
    if onset_i is not None:
        ax[1].axvline(df["t"].iloc[onset_i], c="green", ls="--", lw=1, label="onset")
    if pk is not None:
        ax[1].plot(df["t"].iloc[pk], df["signal"].iloc[pk], "k^", label="peak")
        if lo is not None and hi is not None:
            ax[1].axvspan(df["t"].iloc[lo], df["t"].iloc[hi], alpha=0.15, label="duration")
    ax[1].legend(fontsize=8, loc="upper right")
    ax[1].set_ylabel("signal")

    # Feed-time marker on every panel
    for a in ax:
        a.axvline(args.feed_time, c="red", ls="--", lw=1)

    # Subplots 2-4: behavioral features
    for a, (c, lab) in zip(ax[2:], [("surfacing", "surfacing fraction"),
                                   ("activity", "activity (norm.)"),
                                   ("dispersion", "dispersion (norm.)")]):
        a.plot(df["t"], df[c], alpha=0.35, label="raw")
        a.plot(df["t"], df[c + "_s"], label="smoothed")
        a.set_ylabel(lab)
        a.legend(fontsize=8, loc="upper right")

    ax[-1].set_xlabel("time (s)")
    fig.suptitle("Feeding response (red dashed = feeder trigger)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "response_curve.png"), dpi=150)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--weights", required=True, help="your trained fish detector (.pt)")
    p.add_argument("--feed-time", type=float, required=True,
                   help="seconds into the video when the feed hits the water")
    p.add_argument("--end-time", type=float, default=None,
                   help="optional end time in seconds; frames after this are excluded")
    p.add_argument("--out", default="out_feeding")
    p.add_argument("--roi", default="", help="water region x1,y1,x2,y2 in pixels (default: full frame)")
    p.add_argument("--top-frac", type=float, default=0.3,
                   help="top fraction of ROI counted as surface band; check this on a frame")
    p.add_argument("--sample-fps", type=float, default=2.0)
    p.add_argument("--save-every", type=int, default=4,
                   help="save every Nth sampled frame (+ pre-label) for annotation")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--smooth-s", type=float, default=4.0, help="smoothing window, seconds")
    p.add_argument("--k", type=float, default=2.0, help="onset threshold = k * baseline std")
    p.add_argument("--sustain-s", type=float, default=2.0, help="signal must stay above threshold this long")
    p.add_argument("--frac", type=float, default=0.5, help="duration = time above frac * peak")
    p.add_argument("--min-count", type=int, default=3,
                   help="frames with count < min-count have derived features set to NaN before smoothing")
    p.add_argument("--min-baseline-s", type=float, default=10.0,
                   help="minimum baseline duration in seconds (default: 10.0)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df, roi, band_y2 = run_detection(args)

    if args.end_time is not None:
        df = df[df["t"] <= args.end_time].copy()
        if df.empty:
            raise SystemExit("No frames remain after applying --end-time.")

    df, out, marks = analyse(df, args)
    df.to_csv(os.path.join(args.out, "features_smoothed.csv"), index=False)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    plot(df, out, marks, args, band_y2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
