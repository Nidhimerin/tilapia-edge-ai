
import csv
import statistics
from pathlib import Path
from datetime import datetime

CSV_PATH = Path("out_splash_analysis/splash_scores.csv")
OUTPUT_PATH = Path("out_splash_analysis/pi_feeding_disturbance_report.txt")

EVENTS = [670, 680]  # 11:10 and 11:20 in the video
WINDOW_SECONDS = 5


def mean_score(rows, column, start, end):
    values = [
        float(row[column])
        for row in rows
        if start <= float(row["time_s"]) < end
    ]
    return statistics.mean(values) if values else None


def fmt(value):
    return f"{value:.4f}" if value is not None else "N/A"


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"{CSV_PATH} not found. Run splash_analysis.py first."
        )

    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    required = {"time_s", "near_outlet", "away_from_outlet"}
    if not rows or not required.issubset(rows[0].keys()):
        raise ValueError("CSV is empty or missing required columns.")

    lines = [
        "PI CAMERA FEEDING-EVENT DISTURBANCE REPORT",
        "=" * 44,
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        "Video: pi_feeding_test.mp4",
        "Method: frame-to-frame grayscale change score",
        "",
        "Purpose:",
        "Compare visible image disturbance near the feeder outlet",
        "with disturbance in a separate region away from the outlet.",
        "",
        "Results (mean score in 5-second windows):",
    ]

    for i, event in enumerate(EVENTS, 1):
        # Use non-overlapping windows around the two events.
        before = mean_score(
            rows, "near_outlet", event - WINDOW_SECONDS, event
        )
        after = mean_score(
            rows, "near_outlet", event, event + WINDOW_SECONDS
        )
        away_before = mean_score(
            rows, "away_from_outlet", event - WINDOW_SECONDS, event
        )
        away_after = mean_score(
            rows, "away_from_outlet", event, event + WINDOW_SECONDS
        )

        near_change = after - before if after is not None and before is not None else None
        away_change = (
            away_after - away_before
            if away_after is not None and away_before is not None
            else None
        )

        lines.extend([
            "",
            f"Event {i}: video time {event}s "
            f"({event // 60}:{event % 60:02d})",
            f"Near outlet: before={fmt(before)}, after={fmt(after)}, "
            f"change={fmt(near_change)}",
            f"Away from outlet: before={fmt(away_before)}, "
            f"after={fmt(away_after)}, change={fmt(away_change)}",
        ])

    lines.extend([
        "",
        "Interpretation and limitations:",
        "- A higher score indicates more frame-to-frame visual change.",
        "- Water flow, splashing, lighting, and fish movement can all affect it.",
        "- This is a visual disturbance proxy, not a validated feeding-response",
        "  or appetite measurement, and it does not prove fish consumed the feed.",
        "- The two events are only 10 seconds apart, so interpret their results",
        "  cautiously; nearby activity from one event may affect the next.",
        "- Validate against video observations and manual annotations before",
        "  making biological claims.",
        "",
    ])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved to: {OUTPUT_PATH}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
