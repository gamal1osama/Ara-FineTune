#!/bin/bash
# Reusable setup script to install LLaMA-Factory from source.
# Extracted from notebook: Setup section.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    . "$REPO_ROOT/.env"
    set +a
fi

LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:-$REPO_ROOT/LLaMA-Factory}"
LLAMAFACTORY_TRAIN_NAME="${LLAMAFACTORY_TRAIN_NAME:-news_finetune_train}"
LLAMAFACTORY_VAL_NAME="${LLAMAFACTORY_VAL_NAME:-news_finetune_val}"
LLAMAFACTORY_TRAIN_FILE="${LLAMAFACTORY_TRAIN_FILE:-${LLAMAFACTORY_DATASET_TRAIN_PATH:-/gdrive/MyDrive/ara-finetune/datasets/llamafactory-finetune-data/train.json}}"
LLAMAFACTORY_VAL_FILE="${LLAMAFACTORY_VAL_FILE:-${LLAMAFACTORY_DATASET_VAL_PATH:-/gdrive/MyDrive/ara-finetune/datasets/llamafactory-finetune-data/val.json}}"
LLAMAFACTORY_MODEL_NAME_OR_PATH="${LLAMAFACTORY_MODEL_NAME_OR_PATH:-${BASE_MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}}"
LLAMAFACTORY_TEMPLATE="${LLAMAFACTORY_TEMPLATE:-qwen}"
LLAMAFACTORY_CUTOFF_LEN="${LLAMAFACTORY_CUTOFF_LEN:-3500}"
LLAMAFACTORY_LORA_RANK="${LLAMAFACTORY_LORA_RANK:-64}"
LLAMAFACTORY_LORA_TARGET="${LLAMAFACTORY_LORA_TARGET:-all}"
LLAMAFACTORY_TRAIN_BATCH_SIZE="${LLAMAFACTORY_TRAIN_BATCH_SIZE:-1}"
LLAMAFACTORY_GRAD_ACCUMULATION_STEPS="${LLAMAFACTORY_GRAD_ACCUMULATION_STEPS:-4}"
LLAMAFACTORY_LR="${LLAMAFACTORY_LR:-1.0e-4}"
LLAMAFACTORY_NUM_EPOCHS="${LLAMAFACTORY_NUM_EPOCHS:-3.0}"
LLAMAFACTORY_LR_SCHEDULER_TYPE="${LLAMAFACTORY_LR_SCHEDULER_TYPE:-cosine}"
LLAMAFACTORY_WARMUP_RATIO="${LLAMAFACTORY_WARMUP_RATIO:-0.1}"
LLAMAFACTORY_BF16="${LLAMAFACTORY_BF16:-true}"
LLAMAFACTORY_LOGGING_STEPS="${LLAMAFACTORY_LOGGING_STEPS:-10}"
LLAMAFACTORY_SAVE_STEPS="${LLAMAFACTORY_SAVE_STEPS:-500}"
LLAMAFACTORY_EVAL_STEPS="${LLAMAFACTORY_EVAL_STEPS:-100}"
LLAMAFACTORY_PREPROCESSING_NUM_WORKERS="${LLAMAFACTORY_PREPROCESSING_NUM_WORKERS:-16}"
LLAMAFACTORY_REPORT_TO="${LLAMAFACTORY_REPORT_TO:-wandb}"
LLAMAFACTORY_RUN_NAME="${LLAMAFACTORY_RUN_NAME:-newsx-finetune-llamafactory}"
LLAMAFACTORY_HUB_MODEL_ID="${LLAMAFACTORY_HUB_MODEL_ID:-Gamal1/news-analyzer}"
LLAMAFACTORY_EXPORT_HUB_MODEL_ID="${LLAMAFACTORY_EXPORT_HUB_MODEL_ID:-$LLAMAFACTORY_HUB_MODEL_ID}"
LLAMAFACTORY_HUB_PRIVATE_REPO="${LLAMAFACTORY_HUB_PRIVATE_REPO:-true}"
LLAMAFACTORY_HUB_STRATEGY="${LLAMAFACTORY_HUB_STRATEGY:-checkpoint}"

echo "Cloning LLaMA-Factory repository..."
if [ ! -d "$LLAMAFACTORY_DIR/.git" ]; then
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git "$LLAMAFACTORY_DIR"
fi

echo "Installing LLaMA-Factory in editable mode..."
cd "$LLAMAFACTORY_DIR" && pip install -e .

echo "Injecting custom dataset definitions into LLaMA-Factory/data/dataset_info.json..."
python3 - <<PY
import json
from pathlib import Path

dataset_info_path = Path("data/dataset_info.json")
with dataset_info_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

data["${LLAMAFACTORY_TRAIN_NAME}"] = {
        "file_name": "${LLAMAFACTORY_TRAIN_FILE}",
    "columns": {
        "prompt": "instruction",
        "query": "input",
        "response": "output",
        "system": "system",
        "history": "history"
    }
}

data["${LLAMAFACTORY_VAL_NAME}"] = {
    "file_name": "${LLAMAFACTORY_VAL_FILE}",
    "columns": {
        "prompt": "instruction",
        "query": "input",
        "response": "output",
        "system": "system",
        "history": "history"
    }
}

with dataset_info_path.open("w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
PY

echo "Copying the project training config into LLaMA-Factory/examples/train_lora/..."
mkdir -p examples/train_lora
cat > examples/train_lora/news_finetune.yaml <<EOF
### model
model_name_or_path: ${LLAMAFACTORY_MODEL_NAME_OR_PATH}
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: ${LLAMAFACTORY_LORA_RANK}
lora_target: ${LLAMAFACTORY_LORA_TARGET}

### dataset
dataset: ${LLAMAFACTORY_TRAIN_NAME}
eval_dataset: ${LLAMAFACTORY_VAL_NAME}
template: ${LLAMAFACTORY_TEMPLATE}
cutoff_len: ${LLAMAFACTORY_CUTOFF_LEN}
overwrite_cache: true
preprocessing_num_workers: ${LLAMAFACTORY_PREPROCESSING_NUM_WORKERS}

### output
output_dir: ${LORA_PATH:-/gdrive/MyDrive/ara-finetune/models/}
logging_steps: ${LLAMAFACTORY_LOGGING_STEPS}
save_steps: ${LLAMAFACTORY_SAVE_STEPS}
plot_loss: true

### train
per_device_train_batch_size: ${LLAMAFACTORY_TRAIN_BATCH_SIZE}
gradient_accumulation_steps: ${LLAMAFACTORY_GRAD_ACCUMULATION_STEPS}
learning_rate: ${LLAMAFACTORY_LR}
num_train_epochs: ${LLAMAFACTORY_NUM_EPOCHS}
lr_scheduler_type: ${LLAMAFACTORY_LR_SCHEDULER_TYPE}
warmup_ratio: ${LLAMAFACTORY_WARMUP_RATIO}
bf16: ${LLAMAFACTORY_BF16}
ddp_timeout: 180000000

### eval
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: ${LLAMAFACTORY_EVAL_STEPS}

report_to: ${LLAMAFACTORY_REPORT_TO}
run_name: ${LLAMAFACTORY_RUN_NAME}

push_to_hub: true
hub_model_id: "${LLAMAFACTORY_HUB_MODEL_ID}"
export_hub_model_id: "${LLAMAFACTORY_EXPORT_HUB_MODEL_ID}"
hub_private_repo: ${LLAMAFACTORY_HUB_PRIVATE_REPO}
hub_strategy: ${LLAMAFACTORY_HUB_STRATEGY}
EOF

cd ..

echo "LLaMA-Factory setup complete."
