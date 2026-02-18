#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Allow overriding Hugging Face cache directories to point to a larger disk.
: "${HF_CACHE_DIR:=/media/mahwiz/mahwiz/huggingface_cache_backup}"
mkdir -p "${HF_CACHE_DIR}"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}/datasets"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}/transformers"

./.venv/bin/python -m src.tokenizer_utils
./.venv/bin/python -m src.pretrain
