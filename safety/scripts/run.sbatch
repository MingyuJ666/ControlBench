#!/usr/bin/env bash
#SBATCH --job-name=controlbench-safety
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:30:00
#SBATCH --array=0-2%2
#SBATCH --output=slurm-%A_%a.out
set -euo pipefail

# Supply absolute paths in the submission environment; use your site's partition.
: "${SAFETY_DIR:?Path to the safety section}"
: "${SAFETY_DATA:?Path to prepared data}"
: "${SAFETY_OUTPUT:?New output root}"
: "${QWEN_MODEL_PATH:?Local Qwen3-8B checkpoint}"
: "${GEMMA_MODEL_PATH:?Local gemma-2-9b-it checkpoint}"
: "${LLAMA_MODEL_PATH:?Local Llama-3.1-8B-Instruct checkpoint}"
models=(qwen3_8b gemma2_9b llama31_8b)
paths=("$QWEN_MODEL_PATH" "$GEMMA_MODEL_PATH" "$LLAMA_MODEL_PATH")
index=${SLURM_ARRAY_TASK_ID:?Submit as an array}
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
srun "${SAFETY_PYTHON:-python}" "$SAFETY_DIR/run.py" all \
  --config "${SAFETY_CONFIG:-$SAFETY_DIR/configs/steering.json}" \
  --model "${models[$index]}" --model-path "${paths[$index]}" \
  --data "$SAFETY_DATA" --out "$SAFETY_OUTPUT/${models[$index]}" \
  --sae-dir "${SAFETY_SAE_DIR:-$SAFETY_DIR/dictionaries}"
