#!/usr/bin/env bash
# CPU report/aggregation after the answer judge.
#SBATCH --job-name=l2s-report
#SBATCH --cpus-per-task=4
# Reports stream JSONL rows and do not load a subject model.
#SBATCH --mem=16G
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
cd "$project"
exec bash src/inference.sh report --track "$L2S_TRACK"
