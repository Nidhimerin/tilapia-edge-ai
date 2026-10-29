
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT = Path("out_splash_analysis/splash_scores.csv")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

FEED_TIME = 670.0
BASELINE_END = 600.0
RESPONSE_END = 790.0
SAMPLE_INTERVAL = 0.5

# Prototype thresholds. These must be validated on more feeding events.
ONSET_HOLD_SECONDS = 2.0
RECOVERY_HOLD_SECONDS = 5.0

if not INPUT.exists():
    raise FileNotFoundError(f"Missing input CSV: {INPUT}")

df = pd.read_csv(INPUT)
required = {"time_s", "near_outlet", "away_from_outlet"}
if not required.issubset(df.columns):
    raise ValueError(f"CSV must contain: {sorted(required)}")

df = df.sort_values("time_s").reset_index(drop=True)

if df[list(required)].isna().any().any():
    raise ValueError("Input contains missing values.")

times = df["time_s"].to_numpy(dtype=float)
if len(times) < 2 or not np.allclose(np.diff(times), SAMPLE_INTERVAL, atol=0.02):
    raise ValueError("Expected measurements sampled approximately every 0.5 seconds.")

# Smooth over approximately 3 seconds.
window = max(1, round(3 / SAMPLE_INTERVAL))

for col in ["near_outlet", "away_from_outlet"]:
    df[f"{col}_smooth"] = (
        df[col].rolling(window=window, center=True, min_periods=1).mean()
    )

baseline_mask = (times >= 0) & (times < BASELINE_END)
response_mask = (times >= FEED_TIME) & (times <= RESPONSE_END)

if baseline_mask.sum() < 20 or response_mask.sum() < 10:
    raise ValueError("Not enough baseline or response measurements.")

def analyse_signal(column):
    signal = df[f"{column}_smooth"].to_numpy()
    baseline = signal[baseline_mask]
    response_indices = np.flatnonzero(response_mask)

    mean = float(np.mean(baseline))
    sd = float(np.std(baseline, ddof=1))
    median = float(np.median(baseline))

    # Avoid division by zero for a nearly constant baseline.
    threshold = mean + 2 * sd
    recovery_threshold = mean + sd

    post = signal[response_indices]
    peak_local = int(np.argmax(post))
    peak_idx = int(response_indices[peak_local])
    peak_value = float(signal[peak_idx])

    # Sustained rise above baseline mean + 2 SD.
    hold_n = max(1, round(ONSET_HOLD_SECONDS / SAMPLE_INTERVAL))
    onset_idx = None

    for i in response_indices:
        end = i + hold_n
        if end <= len(signal) and np.all(signal[i:end] > threshold):
            if times[i] >= FEED_TIME:
                onset_idx = i
                break

    # Sustained return to baseline mean + 1 SD after the peak.
    recovery_n = max(1, round(RECOVERY_HOLD_SECONDS / SAMPLE_INTERVAL))
    recovery_idx = None

    for i in range(peak_idx + 1, len(signal) - recovery_n + 1):
        if times[i] > RESPONSE_END:
            break
        if np.all(signal[i:i + recovery_n] <= recovery_threshold):
            recovery_idx = i
            break

    result = {
        "signal": column,
        "baseline_mean": mean,
        "baseline_sd": sd,
        "baseline_median": median,
        "onset_threshold_mean_plus_2sd": threshold,
        "peak_time_s": float(times[peak_idx]),
        "peak_score": peak_value,
        "response_strength_peak_minus_baseline_mean": peak_value - mean,
        "peak_z_score": (
            (peak_value - mean) / sd if sd > 0 else np.nan
        ),
        "onset_time_s": (
            float(times[onset_idx]) if onset_idx is not None else np.nan
        ),
        "onset_latency_s": (
            float(times[onset_idx] - FEED_TIME)
            if onset_idx is not None else np.nan
        ),
        "recovery_time_s": (
            float(times[recovery_idx] - FEED_TIME)
            if recovery_idx is not None else np.nan
        ),
        "recovery_detected": recovery_idx is not None,
    }

    # Count false alarms in the pre-feed baseline using the same
    # 2-SD threshold and sustained 2-second rule.
    baseline_indices = np.flatnonzero(baseline_mask)
    false_windows = 0
    tested_windows = 0

    for i in baseline_indices:
        end = i + hold_n
        if end <= len(signal) and times[end - 1] < BASELINE_END:
            tested_windows += 1
            if np.all(signal[i:end] > threshold):
                false_windows += 1

    result["baseline_false_alarm_windows"] = false_windows
    result["baseline_tested_windows"] = tested_windows
    result["baseline_false_alarm_window_fraction"] = (
        false_windows / tested_windows if tested_windows else np.nan
    )

    return result

results = [
    analyse_signal("near_outlet"),
    analyse_signal("away_from_outlet"),
]

result_df = pd.DataFrame(results)
csv_out = OUT / "pi_feeding_test_prototype.csv"
result_df.to_csv(csv_out, index=False)

# Plot both smoothed signals and their baseline thresholds.
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

for ax, result in zip(axes, results):
    col = result["signal"]
    ax.plot(times, df[f"{col}_smooth"], label=f"{col} (smoothed)")
    ax.axvline(FEED_TIME, linestyle="--", label="Feeding episode")
    ax.axvspan(0, BASELINE_END, alpha=0.12, label="Baseline window")
    ax.axhline(
        result["onset_threshold_mean_plus_2sd"],
        linestyle=":",
        label="Baseline mean + 2 SD",
    )

    if np.isfinite(result["onset_time_s"]):
        ax.axvline(
            result["onset_time_s"], linestyle="-.",
            label="Estimated onset"
        )

    ax.set_ylabel("Disturbance score")
    ax.set_title(col)
    ax.legend()

axes[-1].set_xlabel("Video time (seconds)")
fig.suptitle("Preliminary Feeding-Response Analysis")
fig.tight_layout()

plot_out = OUT / "pi_feeding_test_prototype.png"
fig.savefig(plot_out, dpi=160)
plt.close(fig)

print("\nPRELIMINARY PROTOTYPE RESULTS")
print(result_df.to_string(index=False))
print(f"\nResults CSV: {csv_out}")
print(f"Plot: {plot_out}")
print(
    "\nCAUTION: One compound feeding episode only. "
    "These are exploratory features, not a validated detector."
)
