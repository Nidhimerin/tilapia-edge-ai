import csv, os, subprocess, time, psutil

OUT = "/home/piadmin/tank/logs/health.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

def cpu_temp():
    s = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
    return float(s.split("=")[1].split("'")[0])

def throttled():
    return subprocess.check_output(["vcgencmd", "get_throttled"]).decode().strip().split("=")[1]

new = (not os.path.exists(OUT)) or os.path.getsize(OUT) == 0
with open(OUT, "a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["timestamp", "cpu_temp_c", "throttled_hex", "cpu_pct", "cpu_mhz", "ram_pct", "uptime_s"])
    while True:
        w.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            cpu_temp(),
            throttled(),
            psutil.cpu_percent(interval=1),
            round(psutil.cpu_freq().current),
            psutil.virtual_memory().percent,
            int(time.time() - psutil.boot_time()),
        ])
        f.flush()
        time.sleep(9)
