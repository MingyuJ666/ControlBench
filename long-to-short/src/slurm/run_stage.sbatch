#!/usr/bin/env bash
# One model/stage or one grid shard. Values are supplied by submit_experiment.sh.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
# 32G is sufficient for one 8B/9B subject model plus tokenizer and capture buffers.
# Captures and generated rows are streamed to scratch rather than kept in RAM.
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=src/slurm/out/%x-%j.out
#SBATCH --error=src/slurm/out/%x-%j.err

#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
: "${L2S_STAGE:?L2S_STAGE is required}"
: "${L2S_MODEL:?L2S_MODEL is required}"
: "${L2S_TRACK:?L2S_TRACK is required (iti or non-iti)}"

project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
scratch=${L2S_SCRATCH_ROOT:-$project/.cache/l2s}
venv=${L2S_GPU_VENV:-$scratch/.venv-gpu}
if [[ ! -s "$venv/.READY" ]]; then
  echo "GPU environment is not ready: $venv/.READY" >&2
  exit 2
fi

export L2S_GPU_PYTHON="$venv/bin/python"
export L2S_RUN_ROOT="${L2S_RUN_ROOT:-$scratch/l2s-$L2S_TRACK}"
export L2S_TMPDIR="${L2S_TMPDIR:-$scratch/tmp}"
# Avoid allocator fragmentation during long teacher-forced captures. This is a
# memory-allocation setting only; it does not change model math or artifacts.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$project/src/slurm/out" "$L2S_RUN_ROOT" "$L2S_TMPDIR"
cd "$project"

args=("$L2S_STAGE" --model "$L2S_MODEL" --track "$L2S_TRACK")
read -r -a eval_sets <<< "${L2S_EVAL_SETS:-math500}"
args+=(--eval-sets "${eval_sets[@]}" --max-new-tokens "${L2S_MAX_NEW_TOKENS:-3072}")
case "${L2S_ZERO_CHECK:-false}" in
  true) args+=(--zero-check) ;;
  false) ;;
  *) echo "L2S_ZERO_CHECK must be true or false" >&2; exit 2 ;;
esac
if [[ -n "${L2S_METHODS:-}" ]]; then
  read -r -a methods <<< "$L2S_METHODS"
  args+=(--methods "${methods[@]}")
fi
if [[ -n "${L2S_LAYERS:-}" ]]; then
  read -r -a layers <<< "$L2S_LAYERS"
  args+=(--layers "${layers[@]}")
fi
if [[ -n "${L2S_STRENGTHS:-}" ]]; then
  read -r -a strengths <<< "$L2S_STRENGTHS"
  args+=(--strengths "${strengths[@]}")
fi
if [[ -n "${L2S_SHARD:-}" ]]; then
  args+=(--shard "$L2S_SHARD")
fi

echo "Starting: src/inference.sh ${args[*]}"
exec bash src/inference.sh "${args[@]}"
