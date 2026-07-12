#!/bin/bash
# Script to launch the vLLM server with LoRA adapter support in the background.
# Extracted from notebook: VLLM > Run VLLM section.

# Load local environment variables if .env exists
if [ -f .env ]; then
  export $(echo $(cat .env | grep -v '^#') | xargs)
fi

# Fallback values matching notebook configuration
BASE_MODEL="${BASE_MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}"
LORA_PATH="${LORA_PATH:-/gdrive/MyDrive/ara-finetune/models}"
LORA_MODULE_NAME="${LORA_MODULE_NAME:-news-lora}"
PORT="${VLLM_PORT:-8000}"
GPU_UTIL="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_LEN="${MAX_MODEL_LEN:-3000}"

echo "Starting vLLM server in the background..."
echo "  Base Model:       $BASE_MODEL"
echo "  LoRA Module:      $LORA_MODULE_NAME=$LORA_PATH"
echo "  Port:             $PORT"
echo "  GPU Utilization:  $GPU_UTIL"
echo "  Max Model Len:    $MAX_LEN"

# Ensure log directory exists
mkdir -p logs

nohup python -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL" \
  --served-model-name qwen-base \
  --enable-lora \
  --lora-modules "$LORA_MODULE_NAME=$LORA_PATH" \
  --dtype=float16 \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max_lora_rank 64 \
  --max-model-len "$MAX_LEN" \
  --enforce-eager \
  --port "$PORT" > logs/vllm.log 2>&1 &
  

echo "vLLM server started background PID: $!"
echo "You can follow startup progress with: tail -f logs/vllm.log"
