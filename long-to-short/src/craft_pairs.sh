#!/usr/bin/env bash
set -euo pipefail
set -a
source "$HOME/.env"
set +a
# The repository launcher can select an interpreter after loading local settings.
if [[ "${1:-}" == "--python" ]]; then
  [[ $# -ge 2 && -n "$2" ]] || { echo "--python requires an executable" >&2; exit 2; }
  export L2S_PYTHON="$2"
  shift 2
fi
L2S_SRC="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$L2S_SRC${PYTHONPATH:+:$PYTHONPATH}"
exec "${L2S_PYTHON:-$L2S_SRC/../.venv/bin/python}" -m l2s.craft_pairs "$@"
