#!/usr/bin/env bash
set -euo pipefail
set -a
source "$HOME/.env"
set +a
L2S_SRC="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$L2S_SRC${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${L2S_TMPDIR:-${TMPDIR:-/tmp}}"
if [[ "${1:-}" != "plan" ]]; then
  mkdir -p -- "$TMPDIR"
fi
exec "${L2S_GPU_PYTHON:-python3}" -m l2s_inference.cli "$@"
