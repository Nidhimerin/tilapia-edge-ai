# Tilapia Feeding-Response Analysis Module

Part of the B.Tech capstone project: **"Low-Cost Edge AI Monitoring of Tilapia Count and Feeding-Response Dynamics in a Recirculating Aquaponics Tank Using Raspberry Pi 4"**. This module analyzes overhead tank video recorded during feeding events. It samples video frames, runs a fine-tuned YOLO fish detector, computes behavioral dynamics per frame (fish count, surface band occupation, school dispersion, and movement activity), and generates a multi-feature response curve. From this response curve, it extracts key feeding kinetics: onset latency, peak reaction amplitude, response duration, and decline rate. To support human-in-the-loop active learning, it also exports sampled frames alongside draft YOLO-format pre-annotation label files (`.txt`) for active verification and retraining.

---

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

> **Note**: Model weights and video footage are excluded from version control to keep the repository lightweight. Place your trained fish detector weights at `models/best.pt` (or specify via `--weights`) and store input videos under `footage/` (e.g. `footage/feeding/feeding1.mp4`).

---

## Defining the Region of Interest (ROI)

To ensure fish outside the water area or near reflections/feeders are excluded, define an ROI before analysis:

```bash
python pick_roi.py footage/feeding/feeding1.mp4
```

This interactive tool displays frame 30 of your video. Click and drag a box covering the water surface, then press **ENTER**. It will output a ready-to-use flag, for example:
```text
--roi 45,720,1040,1900
```
Copy and paste this flag directly into your `feeding_analysis.py` execution command.

---

## Example Usage

Run feeding-response analysis using the calibrated ROI:

```bash
python feeding_analysis.py --video footage/feeding/feeding1.mp4 --weights models/best.pt --feed-time 3 --sample-fps 5 --smooth-s 1.5 --sustain-s 1 --roi 45,720,1040,1900 --out out_feeding_v2
```

---

## CLI Flag Reference

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--video` | string | *required* | Path to the feeding video clip (e.g., `footage/feeding/feeding1.mp4`). |
| `--weights` | string | *required* | Path to the trained YOLO fish detector checkpoint (`.pt`). |
| `--feed-time` | float | *required* | **Timestamp (seconds)** into the video when feed pellets hit the water surface. Critical for baseline vs. post-feed segmentation. |
| `--roi` | string | `""` (full frame) | **Water region only (`x1,y1,x2,y2` in pixels)**. Must include the waterline, exclude the mechanical feeder and background tank walls, and remain identical across all compared trials. |
| `--top-frac` | float | `0.3` | **Top fraction of the ROI** designated as the water surface band (e.g. `0.3` = top 30% of ROI height). Verify on a sample frame to align with surface swimming. |
| `--min-count` | int | `3` | **Minimum fish detection threshold**. Frames with `count < min-count` are flagged as low-detection; their derived behavioral features are set to `NaN` prior to smoothing to avoid artifactual zero-dispersion spikes. |
| `--min-baseline-s` | float | `10.0` | **Minimum required baseline duration (s)** prior to `--feed-time`. If pre-feed duration is less than this, a warning is raised and `baseline_too_short: true` is recorded in `summary.json`. |
| `--sample-fps` | float | `2.0` | Frame sampling rate for inference (frames per second). |
| `--save-every` | int | `4` | Saves every *N*-th sampled frame and its YOLO pre-annotation `.txt` file for active learning. |
| `--conf` | float | `0.25` | Detector confidence threshold. |
| `--imgsz` | int | `640` | Input image resolution for YOLO inference. |
| `--smooth-s` | float | `4.0` | Temporal rolling window size (in seconds) used for feature smoothing. |
| `--k` | float | `2.0` | Multiplier for onset threshold calculation: `threshold = max(k * baseline_std, 0.5)`. |
| `--sustain-s` | float | `2.0` | Duration (seconds) the response signal must continuously exceed threshold to confirm feeding onset. |
| `--frac` | float | `0.5` | Peak fraction threshold used to compute duration around the peak (`duration = time where signal >= frac * peak`). |
| `--out` | string | `out_feeding` | Output directory path for summary, plots, CSVs, and pre-labels. |

---

## Output Artifacts

All outputs are saved to the directory specified by `--out`:

1. **`summary.json`**: Key feeding performance metrics, including onset latency, time to peak, peak response value, response duration, decline rate, `post_feed_valid_fraction`, `baseline_too_short`, and the overall `reliable` boolean flag.
2. **`features.csv`**: Raw frame-by-frame data (`frame`, timestamp `t`, `count`, `surfacing`, `dispersion`, `activity`).
3. **`features_smoothed.csv`**: Temporal features after limited gap interpolation (~1 s window) and centered rolling smoothing, plus the composite `signal`.
4. **`response_curve.png`**: High-resolution 5-panel visualization:
   - Panel 1: Raw detection count over time with low-detection spans (`count < min-count`) shaded.
   - Panel 2: Composite response signal ($z$-score), baseline threshold, onset marker, peak marker, and duration window.
   - Panel 3: Surfacing fraction (raw vs. smoothed).
   - Panel 4: Normalized swimming activity (raw vs. smoothed).
   - Panel 5: Normalized school dispersion (raw vs. smoothed).
   - Red dashed vertical line: Feeder trigger timestamp across all panels.
5. **`frames/`**: Sampled `.jpg` video frames saved at the `--save-every` interval.
6. **`labels/`**: Corresponding YOLO-format `.txt` bounding box coordinates (`0 x_center y_center width height`) for each saved frame.

---

## Known Issues

1. **Detector Dropout on `feeding1.mp4`**:
   During the feeding event in `feeding1.mp4` (shortly after the 4.4 s mark), vigorous surface splashing, pellet occlusion, and fish clustering cause the detector count to abruptly plummet from ~6 fish down to 0–2 fish. This dropout causes wide stretches of `NaN` features. The script flags this condition (`post_feed_valid_fraction < 0.70`), sets `"reliable": false` in `summary.json`, and outputs a prominent console warning. See [`examples/`](examples/) for an illustration of an unreliable run caused by this issue.
2. **Short Pre-Feed Baseline**:
   If the pre-feed baseline is under 10 seconds (for example, `--feed-time 3`), the baseline variance is computed over very few samples, making the onset detection threshold unstable. The script now warns and sets `"baseline_too_short": true`.
3. **Domain Shift in Training Data**:
   The current detector weights were fine-tuned primarily on calm, non-feeding pond/tank footage. Rapid agitation, water splashing, and partially overlapping bodies during feeding represent an out-of-distribution domain shift for the model.

---

## Next Steps for Teammate

1. **Inspect Failed Frames**:
   Open `out_feeding_v2/frames/` alongside `out_feeding_v2/labels/` to visually inspect where the detector loses fish during feeding.
2. **Evaluate Lower Confidence Threshold**:
   Re-run with a lower confidence threshold (e.g. `--conf 0.1` or `--conf 0.15`) to see if dropped fish boxes are recovered during water splashing.
3. **Active Learning & Annotation Correction**:
   Import `out_feeding_v2/frames/` and `out_feeding_v2/labels/` into Roboflow or CVAT. Correct missed fish and false boxes on the agitated feeding frames, and fine-tune the detector weights (`models/best.pt`) on these newly labeled feeding scenes.
4. **Re-run & Confirm Reliability**:
   Re-run `feeding_analysis.py` with the retrained detector weights and verify that `summary.json` reports:
   ```json
   "reliable": true
   ```
5. **Record Longer Baselines**:
   For subsequent feeding trials, ensure the camera records at least **10 to 15 seconds** of calm swimming before activating the feeder to establish a statistically robust baseline.
