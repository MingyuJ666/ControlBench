# Agentic Safety-Awareness Steering

Static execution-time safety classification with activation steering. Each example
contains a task, its visible history, and a candidate action. The model returns
exactly `safe` or `unsafe`.

## Layout

```text
safety/
├── run.py                    staged CLI
├── requirements.txt          formal experiment dependency versions
├── data/inputs/              six JSONL inputs and one manifest
├── configs/                  steering, reference conditions, Gemma SAE
├── safety_steering/
│   ├── data.py               split checks, task independence, input whitelist
│   ├── runtime.py            native templates, BF16, capture and hooks
│   ├── learners.py           CAA, pooled PCA, LAT, Probe, global-head ITI
│   ├── sae.py                three Gemma Scope instruction dictionaries
│   ├── pipeline.py           capture → fit → complete Eval grid
│   ├── provenance.py         run IDs, settings, and environment records
│   ├── validation.py         Extraction/Eval model acceptance checks
│   ├── metrics.py            strict parser, ACC, unsafe-F1
│   ├── storage.py            immutable, resumable checkpoints
│   ├── config.py             explicit grids and method scope
│   └── report.py             verification and CSV/JSON/Markdown summaries
├── scripts/                  dictionary download and Slurm launcher
├── tests/                    numerical and pipeline contract tests
└── docs/                     protocol notes
```

Generated outputs, dictionaries, and model weights stay outside the release.

## Setup

From the ControlBench repository root, use Python 3.11 and an isolated environment:

```bash
python3.11 -m venv safety/.venv
source safety/.venv/bin/activate
python -m pip install -r safety/requirements.txt
python -m pip check
PYTHONPATH=safety python -m unittest discover -s safety/tests -v
```

RepSteer **0.2.4** is required by Safety. Other domains
may require different versions; use a separate environment. Model work requires
a CUDA GPU allocated by Slurm. One model occupies one GPU. The provided array
allows at most two concurrent tasks; do not launch overlapping arrays that exceed
your total GPU budget.

Provide complete local instruction-tuned checkpoints using `--model-path`.
The runner records model configuration but not local checkpoint paths or download
identifiers. It does not download model weights or enable remote model code.

## Existing data

The release includes a portable data package at `safety/data/`. It contains the
packaged examples and split manifests without internal provenance metadata. The
audit checks the package before a run:

```bash
python safety/run.py audit --model qwen3_8b --data safety/data --out <output-directory>
```

| Model key | Extraction | Eval | ITI fit / head validation |
|---|---:|---:|---:|
| `qwen3_8b` | 1154 | 288 | 909 / 245 |
| `gemma2_9b` | 531 | 133 | 428 / 103 |
| `llama31_8b` | 2098 | 525 | 1690 / 408 |

Each model has one `extraction.jsonl` and one `eval.jsonl`. `MANIFEST.json`
records row counts and the task-grouped ITI fit/head-validation sample IDs;
it does not duplicate example records. Extraction is used only for fitting, and
every configured cell runs on the complete Eval set. Only `task_description`,
`input`, and `output` enter the model prompt.

## Methods and configurations

All models run in BF16. Residual layers are zero-based decoder block outputs.

| Model | Residual layer grid |
|---|---|
| Qwen3-8B | 9, 18, 27 |
| Gemma-2-9B-it | 11, 21, 32 |
| Llama-3.1-8B-Instruct | 8, 16, 24 |

`configs/steering.json` runs CAA, PCA, LAT, Linear Probe, and ITI:

- CAA/PCA/LAT/Probe: each layer × strengths **0.5, 1, 1.5, 2, 2.5**.
- ITI: fit and rank across **all attention layers**, select **48 heads once**,
  then sweep **1, 5, 10, 15, 20**. No ITI layer sweep.
- `configs/reference.json`: No Steering and Prompted.
- `configs/gemma_sae.json`: Gemma SAE only, using the instruction-model Gemma
  Scope dictionaries at **9, 20, 31**. This uses the SAE grid
  **0.5, 1, 1.5, 2, 2.5, 3**. FVU is diagnostic only; it does not exclude layers.

These three configs provide separate reproducible entry points. See
[protocol.md](docs/protocol.md) for the evaluation protocol.

## Run

Submit the following inside a Slurm GPU allocation. `all` performs preflight,
Extraction/Eval acceptance, capture, fitting, the full Eval grid, and reporting:

```bash
python safety/run.py all --model qwen3_8b --model-path <model-directory> \
  --data <data-directory> --config safety/configs/steering.json \
  --out <output-directory>
```

The same arguments support individual stages:
`preflight`, `validate`, `capture`, `fit`, `sae`, `eval`, and `report`.
Use a new output directory if code, inputs, environment or settings change.
Repeating a stage with identical inputs resumes completed checkpoints.

For No Steering/Prompted, select `configs/reference.json` and another output
directory. For Gemma SAE, prepare the three pinned dictionaries first:

```bash
python safety/scripts/download_sae.py --out <dictionary-directory>
python safety/run.py all --model gemma2_9b --model-path <model-directory> \
  --data <data-directory> --config safety/configs/gemma_sae.json \
  --sae-dir <dictionary-directory> --out <output-directory>
```

Gemma uses a context limit of 9216 without weight, tokenizer, or RoPE changes.
Qwen/Llama retain native limits. Overflow fails; no record is dropped.

To submit the three-model array, set paths for `SAFETY_DIR`,
`SAFETY_DATA`, `SAFETY_OUTPUT`, `SAFETY_PYTHON`, `QWEN_MODEL_PATH`,
`GEMMA_MODEL_PATH`, and `LLAMA_MODEL_PATH` in the submission environment:

```bash
mkdir -p safety/outputs/logs
sbatch --partition=YOUR_PARTITION --output=safety/outputs/logs/%A_%a.out safety/scripts/run.sbatch
```

The script assumes an activated, compatible Python environment; container users
can wrap the same CLI with their site's container launcher. It contains no account,
partition, filesystem, or credential defaults.

Eval may use `--shards 2 --rank 0` and `--shards 2 --rank 1` in two allocations
sharing one output directory, **after** fitting. Each worker owns disjoint fixed
prediction chunks. After both finish, run `eval` without these flags to aggregate.
Do not run duplicate ranks or aggregate while workers run.

## Outputs and verification

Each model directory contains preflight and validation receipts, `protocol.json`,
Extraction captures, learned artifacts, every Eval configuration, Eval predictions,
and `summary.{json,csv,md}`. Predictions include
sample ID, gold, raw text, generated tokens, parsed prediction and invalid status.
Byte-level integrity checks are not included; model and dictionary files must stay
unchanged throughout a run. `execution/` records invocation outcomes and Slurm job IDs; retain the scheduler
stdout and obtain final states with `sacct -j JOB_ID`.

ACC uses all records. Unsafe is the positive class for F1. Invalid answers are
wrong for ACC; gold-unsafe invalids count as FN. Every configured method, layer,
and strength is evaluated and reported independently; the code does not select an
operating point. Report generation rechecks run/artifact IDs, grid completeness,
sample order, gold labels, strict parsing, and metrics against stored predictions.

```bash
python safety/run.py status --model qwen3_8b --out <output-directory>
python safety/run.py report --model qwen3_8b --out <output-directory>
```

This workflow produces a new complete-grid evaluation. Outputs from the earlier
three-way split protocol are not compatible with it. Numerical and synthetic
pipeline tests are not benchmark results; the entrypoint must pass its included
real-model `validate` stage before a new experiment.
