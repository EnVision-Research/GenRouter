#!/usr/bin/env bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Change GPU numbers directly. Comment out services you do not need.
CUDA_VISIBLE_DEVICES=0 python scripts/services/z_image.py &
CUDA_VISIBLE_DEVICES=1 python scripts/services/qwen_image.py --persistent-pipeline &
CUDA_VISIBLE_DEVICES=2 python scripts/services/qwen_image_edit.py --persistent-pipeline &
CUDA_VISIBLE_DEVICES=3 python scripts/services/chat_qwen35.py &

# WISE judge. Create this environment from eval/wise/environment.yml first.
CUDA_VISIBLE_DEVICES=4,5 conda run -n genrouter-wise --no-capture-output \
  vllm serve models/Qwen3.5-35B-A3B \
  --served-model-name Qwen3.5-35B-A3B \
  --tensor-parallel-size 2 \
  --port 8000 &

wait
