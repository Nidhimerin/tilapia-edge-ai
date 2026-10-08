import argparse
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="models/best_finetuned.pt")
ap.add_argument("--formats", nargs="+", default=["onnx", "ncnn"])
a = ap.parse_args()

for f in a.formats:
    print("exported:", YOLO(a.weights).export(format=f, imgsz=640))