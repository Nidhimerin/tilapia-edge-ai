#!/usr/bin/env python3
"""
kinetics.py - feeding-response descriptors from a pixel-signal CSV.

Usage (PowerShell, one line):
  python kinetics.py results/fixed/pi_feeding_test_signal_with_zscores.csv --signal diff_near --feed 670 --window 70 --out results/fixed/kin_pi_feeding_test

Reads the RAW column (e.g. diff_near), z-scores it against rows with
time_s < --baseline-end, and reports for the window [feed, feed+window]:

  Model-free : onset (first >=2 SD held >=2 s), peak z, time to peak,
               duration (full width at half maximum), decline time
               (peak -> half max), area under the z curve
  Fitted     : z(t) = A (1 - exp(-(t-t0)/tr)) exp(-(t-t0)/td) for t > t0
               -> onset t0, rise tr, decay td, peak, time to peak, area
               with 95% CIs from a block bootstrap of the residuals.

Set --window so the window ends BEFORE any known artifact (e.g. the
744-757.5 s shake span in pi_feeding_test: feed 670 + 70 = 740).
--mask drops rows where a boolean 'shake' column is True.
"""
import argparse
import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def model(t, A, t0, tr, td):
    s = np.clip(t - t0, 0, None)
    return A * (1 - np.exp(-s / tr)) * np.exp(-s / td)


def derived(p, grid):
    A, t0, tr, td = p
    y = model(grid, *p)
    i = int(np.argmax(y))
    return {"onset_s": float(t0), "rise_tau_s": float(tr), "decay_tau_s": float(td),
            "peak_z": float(y[i]), "time_to_peak_s": float(grid[i]),
            "area_zs": float(np.trapezoid(y, grid))}


def fit(t, z, p0, bounds):
    p, _ = curve_fit(model, t, z, p0=p0, bounds=bounds, maxfev=20000)
    return p


def runs_above(mask, min_len):
    """Index of first sample that starts a run of >= min_len True values."""
    n = 0
    for i, m in enumerate(mask):
        n = n + 1 if m else 0
        if n >= min_len:
            return i - min_len + 1
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--signal", default="diff_near")
    ap.add_argument("--feed", type=float, required=True)
    ap.add_argument("--window", type=float, default=70.0)
    ap.add_argument("--pre", type=float, default=20.0, help="seconds before feed kept for the fit")
    ap.add_argument("--baseline-end", type=float, default=600.0)
    ap.add_argument("--smooth", type=int, default=5, help="rolling-mean samples for model-free metrics")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--block-s", type=float, default=10.0)
    ap.add_argument("--mask", action="store_true")
    ap.add_argument("--out", required=True, help="output prefix")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    if a.mask and "shake" in df.columns:
        df.loc[df["shake"].astype(bool), a.signal] = np.nan
    base = df[df.time_s < a.baseline_end][a.signal].dropna()
    mu, sd = base.mean(), base.std()
    df["z"] = (df[a.signal] - mu) / sd
    step = float(df.time_s.diff().median())

    w = df[(df.time_s >= a.feed - a.pre) & (df.time_s <= a.feed + a.window)].dropna(subset=["z"])
    t = w.time_s.values - a.feed          # time relative to feed
    z = w.z.values
    nan_frac = float(1 - len(w) / max(1, int(round((a.window + a.pre) / step)) + 1))
    print(f"baseline mean={mu:.4f} sd={sd:.4f} (n={len(base)}); window rows={len(w)}; "
          f"missing fraction={nan_frac:.2f}")

    # ---- model-free ----
    zs = pd.Series(z).rolling(a.smooth, center=True, min_periods=1).mean().values
    post = t >= 0
    tp, zp = t[post], zs[post]
    res = {"signal": a.signal, "feed_s": a.feed, "window_s": a.window,
           "baseline_mean": float(mu), "baseline_sd": float(sd),
           "missing_fraction": nan_frac}
    ip = int(np.argmax(zp))
    res["mf_peak_z"] = float(zp[ip])
    res["mf_time_to_peak_s"] = float(tp[ip])
    k = max(1, int(round(2.0 / step)))
    j = runs_above(zp > 2.0, k)
    res["mf_onset_s"] = float(tp[j]) if j is not None else None
    half = zp[ip] / 2
    lo = ip
    while lo > 0 and zp[lo] > half:
        lo -= 1
    hi = ip
    while hi < len(zp) - 1 and zp[hi] > half:
        hi += 1
    res["mf_half_max_rise_s"] = float(tp[lo])
    res["mf_duration_fwhm_s"] = float(tp[hi] - tp[lo]) if zp[-1] <= half else None  # None = still above half at window end
    res["mf_decline_time_s"] = float(tp[hi] - tp[ip]) if zp[-1] <= half else None
    res["mf_area_zs"] = float(np.trapezoid(np.clip(zp, 0, None), tp))

    # ---- fitted ----
    p0 = [max(zp[ip], 1.0), 2.0, 3.0, 15.0]
    bounds = ([0, -10, 0.5, 1.0], [200, a.window, 60, 300])
    boots = []
    try:
        p = fit(t, z, p0, bounds)
        grid = np.linspace(0, a.window, 2000)
        res["fit"] = derived(p, grid)
        pred = model(t, *p)
        resid = z - pred
        res["fit"]["rmse"] = float(np.sqrt(np.mean(resid ** 2)))
        rng = np.random.default_rng(0)
        bl = max(1, int(round(a.block_s / step)))
        nb = int(np.ceil(len(resid) / bl))
        for _ in range(a.boot):
            starts = rng.integers(0, max(1, len(resid) - bl + 1), size=nb)
            r = np.concatenate([resid[s:s + bl] for s in starts])[:len(resid)]
            try:
                pb = fit(t, pred + r, p, bounds)
                boots.append(derived(pb, grid))
            except Exception:
                pass
        res["fit"]["bootstrap_ok"] = len(boots)
        if len(boots) >= 20:
            for key in ["onset_s", "rise_tau_s", "decay_tau_s", "peak_z", "time_to_peak_s", "area_zs"]:
                v = [b[key] for b in boots]
                res["fit"][key + "_ci95"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        if res["fit"]["decay_tau_s"] > 0.9 * bounds[1][3] or res["fit"]["rise_tau_s"] > 0.9 * bounds[1][2]:
            res["fit"]["warning"] = "a time constant is at its bound: the window may be too short to identify it"
    except Exception as e:
        res["fit"] = {"error": f"fit did not converge: {e}"}
        p = None

    with open(a.out + ".json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

    plt.figure(figsize=(9, 4))
    plt.plot(t, z, ".", ms=3, label="z (raw)")
    plt.plot(tp, zp, label="z smoothed")
    if p is not None:
        plt.plot(grid, model(grid, *p), "r", label="fit")
    plt.axhline(2, ls=":", c="gray"); plt.axvline(0, ls="--", c="gray")
    plt.xlabel("seconds since feed"); plt.ylabel("z"); plt.title(f"{a.signal}")
    plt.legend(); plt.tight_layout(); plt.savefig(a.out + ".png", dpi=120)
    print("wrote", a.out + ".json and .png")


if __name__ == "__main__":
    main()
