from pathlib import Path
import ncnn

model_dir = Path("models/best (2)_ncnn_model")
param_file = model_dir / "model.ncnn.param"
bin_file = model_dir / "model.ncnn.bin"

print("Parameter file exists:", param_file.exists())
print("Model binary exists:", bin_file.exists())

net = ncnn.Net()
ret = net.load_param(str(param_file))
print("load_param return code:", ret)

if ret != 0:
    raise SystemExit("Failed to load model parameters.")

ret = net.load_model(str(bin_file))
print("load_model return code:", ret)

if ret != 0:
    raise SystemExit("Failed to load model weights.")

print("SUCCESS: NCNN loaded the fish model.")
print("Model input/output names from parameter file:")
with open(param_file) as f:
    for line in f:
        if line.startswith("Input") or line.startswith("Output"):
            print(line.strip())

net = None
