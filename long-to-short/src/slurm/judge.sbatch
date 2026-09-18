#!/usr/bin/env bash
# Answer-equivalence judge after every subject-model grid shard completes.
#SBATCH --job-name=l2s-judge
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
# The 27B judge needs more host RAM during checkpoint loading than subject jobs.
# Select a GPU with sufficient VRAM for the judge on the target cluster.
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=src/slurm/out/%x-%j.out
#SBATCH --error=src/slurm/out/%x-%j.err

#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
: "${L2S_TRACK:?L2S_TRACK is required}"
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
mkdir -p "$project/src/slurm/out" "$L2S_RUN_ROOT" "$L2S_TMPDIR"
cd "$project"
exec bash src/inference.sh judge --device cuda:0 --track "$L2S_TRACK"
