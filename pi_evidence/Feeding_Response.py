import streamlit as st, pandas as pd
from pathlib import Path

BASE = Path.home() / "splash_test"
CLIPS = {
    "Clip 2: feeds at 12:01 and 12:11 (721 s, 731 s)": {
        "folder": "out_clip2",
        "note": "Two feeds 10 s apart, treated as one episode.",
    },
    "Clip 1: feeds at 11:10 and 11:20 (670 s, 680 s)": {
        "folder": "out_clip1",
        "note": "Two feeds 10 s apart. The spike near 12 min is a camera shake, not fish activity.",
    },
}

st.title("Feeding response (offline analysis)")
choice = st.selectbox("Recording", list(CLIPS.keys()))
OUT = BASE / CLIPS[choice]["folder"]

csv = OUT / "splash_scores.csv"
if csv.exists():
    df = pd.read_csv(csv).set_index("time_s")
    st.line_chart(df[["near_outlet", "away_from_outlet"]])
else:
    st.warning(f"{csv} not found")

png = OUT / "splash_curve_regions.png"
if png.exists():
    st.image(str(png))

rep = OUT / "pi_feeding_disturbance_report.txt"
if rep.exists():
    st.code(rep.read_text())   # delete these 2 lines to hide the report until the mismatch is fixed

st.caption(CLIPS[choice]["note"] + " Disturbance score is a proxy, not a validated feeding-response measure.")
