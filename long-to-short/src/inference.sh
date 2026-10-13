#!/usr/bin/env bash
set -euo pipefail
set -a
source "$HOME/.env"
set +a
# The repository launcher can select an interpreter after loading local settings.
if [[ "${1:-}" == "--python" ]]; then
  [[ $# -ge 2 && -n "$2" ]] || { echo "--python requires an executable" >&2; exit 2; }
  export L2S_GPU_PYTHON="$2"
  shift 2
fi
L2S_SRC="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$L2S_SRC${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${L2S_TMPDIR:-${TMPDIR:-/tmp}}"
if [[ "${1:-}" != "plan" ]]; then
  mkdir -p -- "$TMPDIR"
fi
exec "${L2S_GPU_PYTHON:-python3}" -m l2s_inference.cli "$@"
