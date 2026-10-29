
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def analyse(csv_path, event_time, baseline_end, response_end, out_dir):
    df = pd.read_csv(csv_path)
    required = {"time_s", "flow_near", "flow_away", "diff_near", "diff_away"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")

    time = df["time_s"].to_numpy(dtype=float)
    if len(time) < 3:
        raise ValueError(f"{csv_path}: not enough rows")

    dt = float(np.median(np.diff(time)))
    if dt <= 0:
        raise ValueError(f"{csv_path}: invalid time intervals")

    baseline_mask = (time >= 0) & (time < baseline_end)
    response_mask = (time >= event_time) & (time <= response_end)
    if baseline_mask.sum() < 10:
        raise ValueError("Too few baseline samples.")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    # Analyse the primary near-outlet difference signal and optical flow.
    for signal in ("diff_near", "flow_near"):
        values = df[signal].to_numpy(dtype=float)
        valid_baseline = baseline_mask & np.isfinite(values)
        valid_response = response_mask & np.isfinite(values)

        baseline = values[valid_baseline]
        response = values[valid_response]
        if len(baseline) < 10 or len(response) < 2:
            print(f"Skipping {signal}: insufficient valid data.")
            continue

        mean = float(np.mean(baseline))
        sd = float(np.std(baseline, ddof=1))
        median = float(np.median(baseline))
        if not np.isfinite(sd) or sd <= 1e-12:
            print(f"Skipping {signal}: baseline SD is zero or invalid.")
            continue

        z = (values - mean) / sd
        threshold = mean + 2 * sd

        # Count independent, non-overlapping 2-second baseline windows
        # in which every available sample exceeds mean + 2 SD.
        window_samples = max(1, int(round(2.0 / dt)))
        baseline_indices = np.flatnonzero(baseline_mask)
        chunks = [
            baseline_indices[i:i + window_samples]
            for i in range(0, len(baseline_indices), window_samples)
            if len(baseline_indices[i:i + window_samples]) == window_samples
        ]
        tested = 0
        fired = 0
        for chunk in chunks:
            segment = values[chunk]
            if not np.all(np.isfinite(segment)):
                continue
            tested += 1
            if np.all(segment > threshold):
                fired += 1

        peak_index = np.flatnonzero(valid_response)[
            np.argmax(z[valid_response])
        ]
        peak_z = float(z[peak_index])
        peak_time = float(time[peak_index])

        # Save the z-score for later kinetics analysis.
        df[f"{signal}_z"] = z

        summary.append({
            "file": Path(csv_path).name,
            "signal": signal,
            "baseline_start_s": 0,
            "baseline_end_s": baseline_end,
            "baseline_mean": mean,
            "baseline_sd": sd,
            "baseline_median": median,
            "event_time_s": event_time,
            "peak_time_s": peak_time,
            "peak_z": peak_z,
            "peak_signal": float(values[peak_index]),
            "baseline_windows_tested": tested,
            "baseline_windows_fired": fired,
            "false_alarm_window_fraction": fired / tested if tested else np.nan,
            "window_duration_s": window_samples * dt,
        })

        plt.figure(figsize=(12, 5))
        plt.plot(time, z, linewidth=0.8, label=f"{signal} z-score")
        plt.axhline(2, linestyle="--", label="z = 2")
        plt.axvline(event_time, linestyle=":", label="Feeding event")
        plt.axvline(baseline_end, linestyle="--", label="Baseline end")
        plt.xlabel("Video time (s)")
        plt.ylabel("Baseline z-score")
        plt.title(f"{Path(csv_path).name}: {signal}")
        plt.legend()
        plt.tight_layout()
        safe_name = Path(csv_path).stem
        plt.savefig(out_dir / f"{safe_name}_{signal}_zscore.png", dpi=150)
        plt.close()

    if not summary:
        raise ValueError(f"No valid signals found in {csv_path}")

    pd.DataFrame(summary).to_csv(
        out_dir / f"{Path(csv_path).stem}_baseline_summary.csv",
        index=False
    )
    df.to_csv(out_dir / f"{Path(csv_path).stem}_with_zscores.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Baseline z-scores and sustained-threshold false alarms."
    )
    parser.add_argument("csv", nargs="+", help="Input pixel-signal CSV files")
    parser.add_argument("--event-times", nargs="+", type=float, required=True,
                        help="One feeding-event time per input CSV")
    parser.add_argument("--baseline-end", type=float, default=600.0)
    parser.add_argument("--response-seconds", type=float, default=180.0)
    parser.add_argument("--out-dir", default="results/baseline")
    args = parser.parse_args()

    if len(args.csv) != len(args.event_times):
        parser.error("Provide exactly one --event-times value per CSV.")
    if args.baseline_end <= 0 or args.response_seconds <= 0:
        parser.error("Baseline and response durations must be positive.")

    out_dir = Path(args.out_dir)
    all_summaries = []

    for csv_path, event_time in zip(args.csv, args.event_times):
        print(f"\nAnalysing {csv_path} (event at {event_time:g}s)")
        if event_time < args.baseline_end:
            print("WARNING: event time is before the configured baseline end.")
        result = analyse(
            csv_path=csv_path,
            event_time=event_time,
            baseline_end=args.baseline_end,
            response_end=event_time + args.response_seconds,
            out_dir=out_dir,
        )
        all_summaries.extend(result)

    pd.DataFrame(all_summaries).to_csv(
        out_dir / "all_baseline_summaries.csv", index=False
    )
    print(f"\nSaved baseline outputs to: {out_dir.resolve()}")
    print(pd.DataFrame(all_summaries).to_string(index=False))


if __name__ == "__main__":
    main()
