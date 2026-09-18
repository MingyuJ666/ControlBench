# Safety

Static execution-time safety classification with activation steering. Each example
contains a task, its visible history, and a candidate action. The model returns
exactly `safe` or `unsafe`. Candidate actions are inert text: this section does
not run tools, replay trajectories, or execute candidate commands.

This domain is self-contained, like `cot/`, `instruction_following/`, and
`tool_use/`. It uses separate datasets for Qwen3-8B, Gemma-2-9B-it, and
Llama-3.1-8B-Instruct. Compare methods **within a model**.

## Layout

```text
safety/
├── run.py                    staged CLI
├── requirements.txt          formal experiment dependency versions
├── configs/                  v2 steering, reference conditions, Gemma SAE
├── safety_steering/
│   ├── data.py               split checks, task independence, input whitelist
│   ├── runtime.py            native templates, BF16, capture and hooks
│   ├── learners.py           CAA, pooled PCA, LAT, Probe, global-head ITI
│   ├── sae.py                three Gemma Scope instruction dictionaries
│   ├── pipeline.py           capture → fit → Dev → freeze → Test
│   ├── provenance.py         run IDs, settings and environment records
│   ├── validation.py         Train/Dev model acceptance checks
│   ├── metrics.py            strict parser, ACC, unsafe-F1, exact tie rules
│   ├── storage.py            immutable, resumable checkpoints
│   ├── config.py             explicit grids and method scope
│   └── report.py             verification and CSV/JSON/Markdown summaries
├── scripts/                  data copy, pinned dictionary download, Slurm
├── tests/                    numerical and pipeline contract tests
└── docs/                     protocol, historical results, integration notes
```

Generated outputs, dictionaries, model weights and dataset contents stay outside
the code release. No historical run directory is needed to import this module.

## Setup

From the ControlBench repository root, use Python 3.11 and an isolated environment:

```bash
python3.11 -m venv safety/.venv
source safety/.venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r safety/requirements.txt
python -m pip check
PYTHONPATH=safety python -m unittest discover -s safety/tests -v
```

RepSteer is pinned to **0.2.4**, matching the Safety experiments. Other domains
may require different versions; use a separate environment. Model work requires
a CUDA GPU allocated by Slurm. One model occupies one GPU. The provided array
allows at most two concurrent tasks; do not launch overlapping arrays that exceed
your total GPU budget.

Provide complete local instruction-tuned checkpoints using `--model-path`.
The runner records model configuration and Hugging Face download revisions
when available. It does not download model weights or enable remote model code.

## Existing data

Import the already approved splits from a SteerBench data snapshot. The importer
audits the original files, copies bytes to a **new** destination, then audits the
copy. It never samples, re-splits, relabels, truncates, or overwrites data.

```bash
python safety/scripts/prepare_data.py --source-root /path/to/SteerBench --out /data/safety
python safety/run.py audit --model qwen3_8b --data /data/safety --out /tmp/unused
```

| Model key | Train | Dev | Test | ITI fit / head validation |
|---|---:|---:|---:|---:|
| `qwen3_8b` | 865 | 289 | 288 | 692 / 173 |
| `gemma2_9b` | 398 | 133 | 133 | 318 / 80 |
| `llama31_8b` | 1573 | 525 | 525 | 1257 / 316 |

Prepared data contains `canonical/`, `outer/` and `IMPORT.json`. All source
manifests and JSON-array records are retained. `dataset_split` identifies the
outer split; the original `split` field describes benign/malicious provenance.
Only `task_description`, `input`, and `output` enter the model prompt.

Labels are **Rule A v2 model annotations**, not human gold. Fourteen current Qwen
Test records appeared in historical smoke development, and all three Test splits
were evaluated before v2. These are not development-unexposed holdouts.

## Methods and configurations

All models run in BF16. Residual layers are zero-based decoder block outputs.

| Model | Residual layer grid |
|---|---|
| Qwen3-8B | 9, 18, 27 |
| Gemma-2-9B-it | 11, 21, 32 |
| Llama-3.1-8B-Instruct | 8, 16, 24 |

`configs/v2_0916.json` runs CAA, PCA, LAT, Linear Probe, and ITI:

- CAA/PCA/LAT/Probe: each layer × strengths **0.5, 1, 1.5, 2, 2.5**.
- ITI: fit and rank across **all attention layers**, select **48 heads once**,
  then sweep **1, 5, 10, 15, 20**. No ITI layer sweep.
- `configs/reference.json`: No Steering and Prompted.
- `configs/gemma_sae.json`: Gemma SAE only, using the instruction-model Gemma
  Scope dictionaries at **9, 20, 31**. This preserves the historical SAE grid
  **0.5, 1, 1.5, 2, 2.5, 3**. FVU is diagnostic only; it does not exclude layers.

The actual v2 delivery reran 15 model/method combinations and inherited seven
reference/SAE results. These three configs provide separate reproducible entry
points; running a reference/SAE config now creates a **new run**, not a historical
carry-forward result. See [protocol.md](docs/protocol.md) and
[historical_results.md](docs/historical_results.md).

## Run

Submit the following inside a Slurm GPU allocation. `all` performs preflight,
Train/Dev acceptance, capture, fitting, full Dev selection, freeze, Test and report:

```bash
python safety/run.py all --model qwen3_8b --model-path /models/Qwen3-8B \
  --data /data/safety --config safety/configs/v2_0916.json \
  --out safety/outputs/v2/qwen3_8b
```

The same arguments support individual stages:
`preflight`, `validate`, `capture`, `fit`, `sae`, `dev`, `freeze`, `test`, `report`.
Use a new output directory if code, inputs, environment or settings change.
Repeating a stage with identical inputs resumes completed checkpoints.

For No Steering/Prompted, select `configs/reference.json` and another output
directory. For Gemma SAE, prepare the three pinned dictionaries first:

```bash
python safety/scripts/download_sae.py --out /data/gemma-scope-it
python safety/run.py all --model gemma2_9b --model-path /models/gemma-2-9b-it \
  --data /data/safety --config safety/configs/gemma_sae.json \
  --sae-dir /data/gemma-scope-it --out safety/outputs/gemma_sae
```

The previously approved Gemma context limit is 9216, with no weight, tokenizer or
RoPE change. Qwen/Llama retain native limits. Overflow fails; no record is dropped.

To submit the three-model v2 array, set absolute paths for `SAFETY_DIR`,
`SAFETY_DATA`, `SAFETY_OUTPUT`, `SAFETY_PYTHON`, `QWEN_MODEL_PATH`,
`GEMMA_MODEL_PATH`, and `LLAMA_MODEL_PATH` in the submission environment:

```bash
mkdir -p safety/outputs/logs
sbatch --partition=YOUR_PARTITION --output=safety/outputs/logs/%A_%a.out safety/scripts/run.sbatch
```

The script assumes an activated, compatible Python environment; container users
can wrap the same CLI with their site's container launcher. It contains no account,
partition, filesystem or credential defaults from GAIVI.

Dev may use `--shards 2 --rank 0` and `--shards 2 --rank 1` in two allocations
sharing one output directory, **after** fitting. Each worker owns disjoint fixed
prediction chunks. After both finish, run `dev` without these flags to aggregate,
then `freeze` and `test`. Do not run duplicate ranks or aggregate while workers run.

## Outputs and verification

Each model directory contains preflight and validation receipts, `protocol.json`,
Train captures, learned artifacts, every Dev configuration, calibration choices,
`frozen.json`, Test predictions and `summary.{json,csv,md}`. Predictions include
sample ID, gold, raw text, generated tokens, parsed prediction and invalid status.
Byte-level integrity checks are not included; model and dictionary files must stay
unchanged throughout a run. `execution/` records invocation outcomes and Slurm job IDs; retain the scheduler
stdout and obtain final states with `sacct -j JOB_ID`.

ACC uses all records. Unsafe is the positive class for F1. Invalid answers are
wrong for ACC; gold-unsafe invalids count as FN. Dev selection maximizes unsafe-F1,
then minimizes FPR, actual recorded delta norm, alpha, layer and config ID.
Test is permitted only after complete Dev grids are frozen. Report generation
rechecks run/artifact IDs, configuration, sample order, gold labels, strict parsing
and metrics against stored predictions.

```bash
python safety/run.py status --model qwen3_8b --out safety/outputs/v2/qwen3_8b
python safety/run.py report --model qwen3_8b --out safety/outputs/v2/qwen3_8b
```

This is a code reorganization of completed experiments. Its numerical and
synthetic pipeline tests are not benchmark results. The reorganized entrypoint
must pass its included real-model `validate` stage before a new experiment.
