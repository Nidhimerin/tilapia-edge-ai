import os
from datetime import datetime

import pandas as pd
import streamlit as st

LOGS = "/home/piadmin/tank/logs"

st.set_page_config(page_title="Tank Monitor", layout="wide")
st.title("Tilapia Tank Monitor (Raspberry Pi 4)")
st.caption(
    "Processed recorded footage, not a live stream. "
    "Counts are raw detector output and have not yet been validated "
    "against manual counts."
)

PI_FLAGS = {
    0: "under-voltage NOW", 1: "freq capped NOW",
    2: "throttled NOW", 3: "soft temp limit NOW",
    16: "under-voltage occurred", 17: "freq capping occurred",
    18: "throttling occurred", 19: "soft temp limit occurred",
}


def decode_throttle(hexstr):
    try:
        v = int(str(hexstr), 16)
    except ValueError:
        return "unreadable"
    names = [n for bit, n in PI_FLAGS.items() if v & (1 << bit)]
    return ", ".join(names) if names else "none"


@st.cache_data(ttl=5)
def load(name, required=()):
    """Return a DataFrame, or None if missing/empty/partial/wrong columns."""
    p = f"{LOGS}/{name}"
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_csv(p, on_bad_lines="skip")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return None
    if df.empty or any(c not in df.columns for c in required):
        return None
    return df


counts = load("counts.csv", ("video", "count", "video_time_s", "infer_ms"))
health = load("health.csv", ("timestamp", "cpu_temp_c", "throttled_hex", "uptime_s"))
bench = load("benchmark.csv")

# ---------------- Detection ----------------
if counts is not None:
    video = st.selectbox("Video", counts["video"].unique()[::-1])
    c = counts[counts["video"] == video].sort_values("video_time_s").copy()
    c["smoothed"] = c["count"].rolling(10, min_periods=1, center=True).median()

    left, right = st.columns(2)
    with left:
        st.subheader("Latest processed frame")
        img = f"{LOGS}/latest.jpg"
        if os.path.exists(img):
            try:
                with open(img, "rb") as f:
                    st.image(f.read())
                ts = datetime.fromtimestamp(os.path.getmtime(img))
                st.caption(
                    f"Written {ts:%Y-%m-%d %H:%M:%S}. This is the most recent "
                    "frame from the last detector run, not necessarily the "
                    "selected video."
                )
            except Exception:
                st.info("Frame is being updated, press Refresh.")
        else:
            st.info("No latest.jpg yet.")
    with right:
        st.subheader("Fish count over video time")
        st.line_chart(c.set_index("video_time_s")[["count", "smoothed"]])
        m1, m2, m3 = st.columns(3)
        m1.metric("Mean count", f"{c['count'].mean():.1f}")
        m2.metric("Peak count", int(c["count"].max()))
        m3.metric("Avg inference (ms)", f"{c['infer_ms'].mean():.0f}")
else:
    st.warning("No usable counts.csv yet. Run run_detect_video.py first.")

# ---------------- Health ----------------
st.subheader("Pi health")
if health is not None:
    health = health.copy()
    health["timestamp"] = pd.to_datetime(health["timestamp"], errors="coerce")
    health["uptime_s"] = pd.to_numeric(health["uptime_s"], errors="coerce")
    health = health.dropna(subset=["timestamp"]).tail(2000)
if health is not None and len(health):
    h1, h2 = st.columns(2)
    with h1:
        st.line_chart(health.set_index("timestamp")["cpu_temp_c"])
    with h2:
        last = health.iloc[-1]
        st.metric("Max CPU temp in shown log (°C)", f"{health['cpu_temp_c'].max():.1f}")
        st.metric("Throttle status (latest)", decode_throttle(last["throttled_hex"]))
        up = last["uptime_s"]
        st.metric("Uptime (min)", int(up // 60) if pd.notna(up) else "n/a")
else:
    st.info("No usable health.csv yet. Start health_logger.py.")

# ---------------- Benchmark ----------------
st.subheader("Benchmark")
if bench is not None:
    st.dataframe(bench)
else:
    st.info("No benchmark.csv yet.")

st.button("Refresh")
