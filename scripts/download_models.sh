#!/usr/bin/env bash
# Download the three GGUF models named in configs/default.yaml into models/.
# Roughly 4.5 GB total. Nothing large is ever committed to the repo.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p models

fetch () {
  local repo="$1" file="$2"
  if [ -f "models/$(basename "$file")" ]; then
    echo "==> already have $(basename "$file"), skipping"
    return
  fi
  echo "==> downloading $file from $repo"
  python -c "
from huggingface_hub import hf_hub_download
import shutil, sys
path = hf_hub_download(repo_id='$repo', filename='$file')
shutil.copy(path, 'models/' + '$file'.split('/')[-1])
print('saved to models/' + '$file'.split('/')[-1])
"
}

# Arm A: the resident generalist
fetch "microsoft/Phi-3-mini-4k-instruct-gguf" "Phi-3-mini-4k-instruct-q4.gguf"

# Arm B: the two specialists
fetch "bartowski/Llama-3.2-1B-Instruct-GGUF" "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
fetch "Qwen/Qwen2.5-1.5B-Instruct-GGUF" "qwen2.5-1.5b-instruct-q4_k_m.gguf"

echo
echo "models/ now contains:"
ls -lh models/
echo
echo "If a filename 404s, the repo has renamed its quant files."
echo "Browse the repo on huggingface.co, pick the Q4_K_M file, and update"
echo "both configs/default.yaml and this script."
