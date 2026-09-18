#!/usr/bin/env bash
# One-time GPU environment setup on scratch.
#SBATCH --job-name=l2s-setup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=src/slurm/out/%x-%j.out
#SBATCH --error=src/slurm/out/%x-%j.err

#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail

project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
scratch=${L2S_SCRATCH_ROOT:-$project/.cache/l2s}
venv=${L2S_GPU_VENV:-$scratch/.venv-gpu}
mkdir -p "$project/src/slurm/out" "$scratch/tmp" "$scratch/pip-cache"
# Both tracks share the environment. Serialize its first install and reuse it
# once ready, so a second submission cannot upgrade a running job's packages.
exec 9>"$scratch/.venv-gpu.setup.lock"
flock 9
if [[ -s "$venv/.READY" ]]; then
  echo "Reusing GPU environment: $venv"
  exit 0
fi
export TMPDIR="$scratch/tmp"
export PIP_CACHE_DIR="$scratch/pip-cache"

# Load any site-specific Python module before this script if needed. Recreate a
# partial environment if an earlier attempt used an unsupported interpreter.
if [[ -x "$venv/bin/python" ]] && ! "$venv/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  rm -rf "$venv"
fi
python3 -m venv "$venv"
"$venv/bin/python" -m pip install --upgrade pip

# Install PyTorch before the remaining requirements. The GPU allocation below
# validates that the selected wheel can see CUDA and supports bfloat16.
"$venv/bin/python" -m pip install --upgrade torch
"$venv/bin/python" -m pip install -r "$project/src/requirements-inference.txt"
"$venv/bin/python" -m pip install "$project/repsteer"

"$venv/bin/python" - <<'PY'
import torch
import transformers
import repsteer

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the setup allocation")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("The allocated GPU does not support bfloat16")
print("GPU:", torch.cuda.get_device_name(0))
print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("repsteer:", repsteer.__version__)
PY

printf 'ready\n' > "$venv/.READY"
chmod 600 "$venv/.READY"
echo "GPU environment ready: $venv"
