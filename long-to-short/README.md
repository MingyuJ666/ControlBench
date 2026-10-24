# Long-to-Short Reasoning Steering

This directory contains the implementation of the long-to-short (L2S) reasoning-steering experiment. It includes the frozen 64-pair extraction set in `data/l2s/extraction.jsonl`; generated answers, fitted artifacts, reports, caches, virtual environments, and Slurm logs are **not** included.

The experiment asks whether activations learned from matched long and short mathematical derivations can make a model produce shorter reasoning without losing answer accuracy. Pair construction and GPU inference are separate stages.

See the [repository guide](../README.md) for the other ControlBench domains.
Run the commands in this guide from **`long-to-short/`** (`cd long-to-short`
from the repository root).

## What is in this directory

```text
src/
  l2s/                 # Stage A: construct and validate 64 long/short pairs
  l2s_inference/       # Stage B: fit steering directions, generate, judge, report
  tests/               # Small offline unit tests; no model or network access
  craft_pairs.sh       # Stage-A launcher
  inference*.sh        # Stage-B launchers for the two execution tracks
  slurm/               # Portable Slurm templates; adapt resources to the cluster
  requirements-*.txt   # Separate CPU and GPU environments
```

`runs/` and `results/` are intentionally ignored by Git. They contain generated outputs, not the code handoff.

## Experiment flow

1. **Construct pairs (CPU / API).** `l2s.craft_pairs` samples eligible MATH training questions, creates a long solution, extracts its key-step skeleton, rewrites a short solution using the same method, and applies correctness, length, and coverage filters. It only publishes `extraction.jsonl` after exactly 64 distinct pairs pass.
2. **Fit steering artifacts (GPU).** `l2s_inference.fit` captures the reasoning spans of the frozen pairs and fits CAA, PCA, LAT, and Linear-Probe directions. Gemma additionally supports SAE; the separate ITI track fits a global all-layer Top-48 profile with a pair-grouped holdout.
3. **Evaluate and report (GPU/CPU).** Baselines and the positive-strength grid run on MATH-500 by default (AMC12 is optional). The answer judge resolves non-exact but non-empty answer mismatches; the report stage aggregates the JSONL output.

The non-ITI and ITI tracks use separate output roots so that they can be run and resumed independently.

## Data policy

The frozen pairs used by the experiment are in `data/l2s/extraction.jsonl` (exactly 64 distinct pairs). Stage B reads this directory by default, so Stage A is only needed to regenerate pairs; with this file present, `craft_pairs` reports the stage as complete without making teacher calls.

To use a different pair directory, pass it explicitly; it must contain `extraction.jsonl` with exactly 64 distinct pairs:

```bash
bash src/inference_non_iti.sh plan --pairs /absolute/path/to/l2s
```

## Setup

Pair construction and inference intentionally use separate environments because they require incompatible Transformers major versions.

For pair construction, create a Python environment and install the locked dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r src/requirements-pairs.lock
bash src/craft_pairs.sh --prepare-only
bash src/craft_pairs.sh
```

For GPU inference, use Python 3.10+ and install a CUDA-compatible PyTorch build before the inference requirements. The code also needs the unmodified `repsteer` dependency at commit `2977008e32cc283dc2f37550bdfe215de1a7c02a`:

```bash
python3 -m venv .venv-gpu
.venv-gpu/bin/pip install torch
.venv-gpu/bin/pip install -r src/requirements-inference.txt
git clone https://github.com/binhu02/repsteer /tmp/repsteer
git -C /tmp/repsteer checkout 2977008e32cc283dc2f37550bdfe215de1a7c02a
.venv-gpu/bin/pip install /tmp/repsteer
export L2S_GPU_PYTHON="$PWD/.venv-gpu/bin/python"
```

Both stages load `~/.env`. It must define absolute Hugging Face cache paths (`HF_HOME`, `HF_HUB_CACHE`, `HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE`). Stage A additionally requires `OPENAI_API_KEY` and `OPENAI_BASE_URL` for the teacher endpoint.

## Running and checking

Inspect a planned GPU workload without loading a model:

```bash
L2S_GPU_PYTHON=python3 bash src/inference_non_iti.sh plan
L2S_GPU_PYTHON=python3 bash src/inference_iti.sh plan
```

The ordinary order for one model is `fit`, `baselines`, `grid`, `judge`, then `report`.
The non-ITI and ITI wrappers default to `runs-non-iti/` and `runs-iti/`, respectively;
the combined `inference.sh` track defaults to `runs/`. Set `L2S_RUN_ROOT` or `--root`
to put artifacts elsewhere. An explicit root overrides the track-specific defaults,
so choose separate roots for the two tracks and keep each root consistent across
stages. Examples:

```bash
bash src/inference_non_iti.sh fit --model Llama-3.1-8B
bash src/inference_non_iti.sh baselines --model Llama-3.1-8B
bash src/inference_non_iti.sh grid --model Llama-3.1-8B
bash src/inference_non_iti.sh judge
bash src/inference_non_iti.sh report
```

Run the lightweight test suite from this directory:

```bash
PYTHONPATH=src python3 -m unittest discover -s src/tests -p 'test_*.py' -v
```

The Slurm helpers in `src/slurm/` are templates. Review GPU type, memory, time limit, module setup, and scratch location before submission. They use `L2S_SCRATCH_ROOT` (default: `.cache/l2s/`) and never encode a user-specific checkout path.
