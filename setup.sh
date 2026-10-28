#!/usr/bin/env bash
# Training stack for the CV lab: PyTorch (CUDA) + Ultralytics YOLO + OpenCV.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo ">> creating venv"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo ">> upgrading pip"
python -m pip install -q --upgrade pip wheel setuptools

echo ">> installing torch + ultralytics (large download, be patient)"
# default PyPI index: resolves CUDA 13 wheels for python 3.14 (verified with pip --dry-run)
python -m pip install --upgrade torch torchvision
python -m pip install --upgrade ultralytics opencv-python

echo ">> verifying"
python - <<'EOF'
import torch, platform
print("python   :", platform.python_version())
print("torch    :", torch.__version__)
print("cuda     :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu      :", torch.cuda.get_device_name(0))
    print("vram GiB :", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
try:
    import ultralytics
    print("ultralytics:", ultralytics.__version__)
except Exception as e:
    print("ultralytics FAILED:", e)
EOF
echo ">> done. activate with: source $(pwd)/.venv/bin/activate"
