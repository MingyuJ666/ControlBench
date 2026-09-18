#!/usr/bin/env bash
set -euo pipefail
set -a
source "$HOME/.env"
set +a
L2S_SRC="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$L2S_SRC${PYTHONPATH:+:$PYTHONPATH}"
exec "${L2S_PYTHON:-$L2S_SRC/../.venv/bin/python}" -m l2s.craft_pairs "$@"
