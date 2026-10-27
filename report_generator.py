
"""Generate a cautious daily report from feeding-analysis results.

The numerical analysis is performed by Python. The optional LLM only
summarizes supplied results and must not invent measurements or diagnoses.
"""

import argparse
import json
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen


def rule_based_report(data):
    """Generate a report without an LLM."""
    lines = ["Tilapia Feeding-Response Daily Summary", ""]

    reliable = data.get("reliable", False)
    valid_fraction = data.get("post_feed_valid_fraction")
    baseline_short = data.get("baseline_too_short", True)

    lines.append(
        "Analysis reliability: "
        + ("SUITABLE FOR PRELIMINARY INTERPRETATION"
           if reliable else "UNRELIABLE — interpret with caution")
    )

    if valid_fraction is not None:
        lines.append(
            f"Valid post-feeding signal: {valid_fraction:.1%}"
        )

    if baseline_short:
        lines.append(
            "The pre-feeding baseline was too short for reliable "
            "threshold estimation."
        )

    if not reliable:
        lines.append(
            "Onset, peak, duration and decline measurements are not "
            "reported as validated results. Check detector performance, "
            "video quality and baseline duration before interpretation."
        )
    else:
        onset = data.get("onset_latency_s")
        peak = data.get("peak_value")
        time_peak = data.get("time_to_peak_s")
        duration = data.get("duration_s")
        decline = data.get("decline_rate_per_s")

        lines.append(
            "The analysis passed its configured reliability checks. "
            "These checks do not by themselves establish biological validity."
        )
        if onset is not None:
            lines.append(f"Estimated onset latency: {onset:.2f} s.")
        if peak is not None:
            lines.append(f"Peak behavioural response index: {peak:.2f}.")
        if time_peak is not None:
            lines.append(f"Time to peak after feeding: {time_peak:.2f} s.")
        if duration is not None:
            lines.append(f"Estimated response duration: {duration:.2f} s.")
        if decline is not None:
            lines.append(f"Estimated decline slope: {decline:.3f} index units/s.")

    lines.extend([
        "",
        "Limitations:",
        "The response index is a heuristic derived from visible behaviour. "
        "It is not a validated appetite, stress, disease or satiation measure.",
        "Interpret trends only after checking detections and comparing "
        "multiple feeding events.",
    ])
    return "\n".join(lines)


def llm_report(data, model, url):
    """Ask a locally running Ollama model to summarize verified results."""
    prompt = f"""
Write a concise daily report for a tilapia tank monitoring prototype.

Use only the supplied JSON. Do not invent values, diagnose disease or
stress, infer appetite or satiation, or claim causation.
If reliable is false, explicitly state that the feeding measurements
are unreliable and do not interpret their numerical values.
Distinguish observed measurements from possible explanations.
End with one practical validation step.

JSON results:
{json.dumps(data, indent=2)}
"""

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1}
    }).encode("utf-8")

    request = Request(
        url.rstrip("/") + "/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=90) as response:
        result = json.loads(response.read().decode("utf-8"))

    text = result.get("response", "").strip()
    if not text:
        raise ValueError("The LLM returned an empty response.")
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Generate a cautious report from summary.json"
    )
    parser.add_argument("--summary", default="out_feeding_v2/summary.json")
    parser.add_argument("--output", default="daily_report.txt")
    parser.add_argument(
        "--llm", action="store_true",
        help="Use a locally running Ollama model if available"
    )
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434"
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.is_file():
        raise SystemExit(
            f"Summary file not found: {summary_path}\n"
            "Run feeding_analysis.py first or pass --summary with its path."
        )

    with summary_path.open(encoding="utf-8") as file:
        data = json.load(file)

    if args.llm:
        try:
            report = llm_report(data, args.model, args.ollama_url)
            mode = f"Ollama LLM ({args.model})"
        except (URLError, HTTPError, TimeoutError, ValueError,
                json.JSONDecodeError, OSError) as error:
            print(f"LLM unavailable ({error}); using rule-based report.")
            report = rule_based_report(data)
            mode = "rule-based fallback"
    else:
        report = rule_based_report(data)
        mode = "rule-based"

    output_path = Path(args.output)
    output_path.write_text(report + "\n", encoding="utf-8")

    print(f"Report mode: {mode}")
    print(f"Saved report: {output_path}")
    print("-" * 55)
    print(report)


if __name__ == "__main__":
    main()