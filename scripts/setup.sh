#!/usr/bin/env bash
# One-time environment setup. Safe to re-run.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"

echo "==> creating virtualenv in .venv"
"$PYTHON" -m venv .venv

echo "==> installing requirements"
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

mkdir -p models results analysis

cat <<'EOF'

Setup complete.

Next:
  source .venv/bin/activate
  ./scripts/download_models.sh
  ./scripts/run_bench.sh

On a Jetson or a CUDA laptop, rebuild llama-cpp-python with GPU support:
  CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
and set runtime.n_gpu_layers in configs/default.yaml.
EOF
