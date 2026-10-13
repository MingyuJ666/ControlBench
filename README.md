# ControlBench

ControlBench evaluates activation steering across five domains: instruction
following, tool use, mathematical reasoning, safety awareness, and long-to-short
reasoning. It compares baseline generation with methods including CAA, PCA, LAT,
linear probing, ITI, and SAE.

[bench.py](bench.py) provides a common entry point for selecting a domain and its
Python environment.

## Domains

| Domain | CLI ID | Evaluation |
|---|---|---|
| [Instruction following](instruction-following/README.md) | `instruction_following` | Five response-format constraints, scored by format rules and a semantic judge |
| [Tool use](tool-use/README.md) | `tool_use`, `tool_use_multi_turn` | Call, clarify, or abstain decisions; scripted follow-ups and executable BFCL conversations |
| [Reasoning](reasoning/README.md) | `reasoning` | Mathematical answer accuracy on MATH500, AMC2023, and AIME2024 |
| [Safety](safety/README.md) | `safety` | Candidate-action classification as `safe` or `unsafe`; accuracy and unsafe-F1 |
| [Long-to-short](long-to-short/README.md) | `long_to_short` | Mathematical answer accuracy, generated length, and compression |

Tool use has two tracks. `tool_use` evaluates individual decisions and scripted
follow-ups. `tool_use_multi_turn` executes tool calls against stateful BFCL
backends and feeds the results into the conversation. Both tracks learn from
the same contrastive pairs.

## Setup

Install the environment for the domain you want to run. The domain guides list
the dependencies, model configuration, and data preparation steps.

| Environment | Installation guide | Run inputs |
|---|---|---|
| IF and tool use | [IF setup](instruction-following/README.md#setup) / [Tool-use setup](tool-use/README.md#setup) | Python 3.10+, CUDA PyTorch, model access; `bfcl-eval==2026.3.23` for executable tool use |
| IF/tool-use judge | Each domain's `requirements-judge.txt` in a separate environment | `JUDGE_PYTHON` selects the vLLM interpreter; pinned dependencies target A100 GPUs |
| Reasoning | [Setup and inputs](reasoning/README.md#fresh-checkout-prerequisites) | Local model configuration, fixed split and audit artifacts, model-specific negative solutions |
| Safety | [Safety setup](safety/README.md#setup) | Python 3.11, local model checkpoints, Slurm GPU allocation; Gemma dictionaries for SAE |
| Long-to-short | [Pair construction and inference setup](long-to-short/README.md#setup) | Separate CPU/API and GPU environments, pinned RepSteer, and 64 frozen training pairs |

Long-to-short's wrappers load cache settings from `~/.env`; pair construction
also uses the configured teacher API.

## Unified launcher

Run from the **ControlBench repository root**:

```bash
python bench.py --list
python bench.py --domain reasoning -- --help
```

The command format is:

```text
python bench.py --domain DOMAIN [--entry ENTRY] [--python EXECUTABLE] -- DOMAIN_ARGUMENTS
```

| Launcher option | Purpose |
|---|---|
| `--domain` | Select one of the CLI IDs above |
| `--entry` | Select the domain's workflow; defaults are listed below |
| `--python` | Select the interpreter; defaults to the Python running `bench.py` |
| `--preview` | Print the command, working directory, and environment overrides |
| `--` | Pass the following arguments to the domain's CLI |

| Domain | Entries | Default | Working directory |
|---|---|---|---|
| IF and both tool-use tracks | `campaign`, `run`, `selfcheck` | `campaign` | Repository root |
| Reasoning | `main` | `main` | Repository root |
| Safety | `main` | `main` | Repository root |
| Long-to-short | `non-iti`, `iti`, `inference`, `pairs` | `non-iti` | `long-to-short/` |

For example, select an installed environment and inspect a tool-use launch:

```bash
python bench.py --domain tool_use_multi_turn \
  --python .venv-if-tool-use/bin/python --preview -- \
  --out outputs/tool_use_multi_turn --methods unsteered --sample 8 --gpus 0
```

The launcher sets the working directory and import path. Relative domain
arguments use the working directory in the table; `--python` is resolved from
the calling directory. For long-to-short, `--python` takes precedence over the
interpreter setting in `~/.env`. The IF/tool-use judge uses `JUDGE_PYTHON`.

## Run a domain

### Instruction following

The IF campaign learns steering directions, generates responses, applies the
format rules, and obtains semantic judgments. In the IF environment, preview a
sampled campaign:

```bash
export JUDGE_PYTHON="$PWD/.venv-judge/bin/python"
python bench.py --domain instruction_following -- \
  --out outputs/instruction_following \
  --model Qwen/Qwen3-8B --methods unsteered,prompted,caa,lat \
  --layers 9 --strengths 0.5,1.0 \
  --train-sample 32 --train-seed 42 --sample 16 --seed 42 \
  --gpus 0 --batch 4 --max-new 512 --dry-run
```

Remove `--dry-run` to execute. `--eval-split full|dev|test` selects 1,014/405/609
prompts per target. The [IF guide](instruction-following/README.md#run-experiments)
contains full-data commands, method grids, and SAE settings.

### Tool use

The tool-use campaign learns from action-contrast pairs, evaluates tool decisions
or executable conversations, and scores the resulting responses and traces.
In the tool-use environment, preview the decision track:

```bash
export JUDGE_PYTHON="$PWD/.venv-judge/bin/python"
python bench.py --domain tool_use -- \
  --out outputs/tool_use \
  --model Qwen/Qwen3-8B --methods unsteered,prompted,caa,lat \
  --layers 9 --strengths 1.0 \
  --train-sample 32 --train-seed 42 --sample 8 --seed 42 \
  --gpus 0 --batch 2 --max-new 1024 --dry-run
```

For executable conversations, use `--domain tool_use_multi_turn` with
`--out outputs/tool_use_multi_turn`. Remove `--dry-run` to execute. The [tool-use guide](tool-use/README.md#run-experiments)
describes both tracks, their methods, and their metrics.

### Reasoning

Configure `reasoning/models.local.json`, the experiment's fixed split and audit
artifacts, and the negative solutions for the selected model using the
[reasoning setup](reasoning/README.md#fresh-checkout-prerequisites).

```bash
python bench.py --domain reasoning -- \
  suite dry-run --methods caa pca lat probe iti
```

The suite fits directions, selects configurations on dev, and evaluates them on
the test sets. Use `suite start` for a background run or `suite run` for a
foreground run, with explicit `--device` and `--output` settings.

### Safety

Audit the packaged extraction and evaluation data:

```bash
python bench.py --domain safety -- audit --model qwen3_8b \
  --data safety/data --out safety/outputs/qwen-steering
```

Inside a Slurm GPU allocation, run the configured stages with a local checkpoint:

```bash
python bench.py --domain safety -- all --model qwen3_8b \
  --model-path /absolute/path/to/model --data safety/data \
  --config safety/configs/steering.json --out safety/outputs/qwen-steering
```

`all` performs preflight, model acceptance checks, capture, fitting, evaluation,
and reporting. The [safety guide](safety/README.md#methods-and-configurations)
describes the steering, reference, and Gemma SAE configurations.

### Long-to-short

Prepare the inference environment and a directory containing `extraction.jsonl`
with 64 distinct pairs. The [long-to-short guide](long-to-short/README.md#setup)
covers pair construction and the pinned dependencies.

```bash
python bench.py --domain long_to_short --entry non-iti \
  --python long-to-short/.venv-gpu/bin/python -- \
  plan --pairs /absolute/path/to/l2s
```

Run the stages in order: `fit → baselines → grid → judge → report`. Use
`--entry iti` for the ITI track and `--entry pairs` for pair construction.

## Data

Paths are relative to the repository root. Each domain guide describes its data
construction and evaluation protocol.

| Domain | Input location | Data |
|---|---|---|
| Instruction following | `instruction-following/data/inputs/` | 1,014 extraction and 1,014 evaluation prompts, shared across five targets |
| Tool use | `tool-use/data/inputs/` | 8,100 training pairs, 3,109 decision items, and 600 BFCL conversations |
| Reasoning | `reasoning/data/` and the configured split/audit artifacts | MATH500, teacher and model solutions, AMC2023, and AIME2024 |
| Safety | `safety/data/inputs/` | Model-specific extraction and evaluation sets for Qwen, Gemma, and Llama |
| Long-to-short | Directory selected by `--pairs` | 64 frozen long/short pairs; evaluation datasets loaded separately |

## Outputs

`bench.py` launches every domain through the same interface. Each domain writes
to the output location defined by its workflow.

| Domain | Default output location | Configuration |
|---|---|---|
| Instruction following | `data/instruction_following/outputs/repsteer-native-v2/` | `--out DIR` |
| Tool use | `data/tool_use/outputs/repsteer-native-v2/` and `data/tool_use_multi_turn/outputs/repsteer-native-v2/` | `--out DIR` |
| Reasoning | `reasoning/outputs/v2/<workflow>/`; suite defaults to `steering_suite/` | `--output` |
| Safety | Directory supplied to `--out`, such as `safety/outputs/<run>/` | Required `--out` |
| Long-to-short | `long-to-short/runs-non-iti/` and `long-to-short/runs-iti/` | `--root` or `L2S_RUN_ROOT` |

IF and both tool-use tracks accept `--out DIR` after the `--` separator.
Results go directly under `DIR/artifacts/`, `DIR/runs/`, and `DIR/cache/`;
campaign logs go to `DIR/logs/<model>/<experiment>/`. `--logs` selects another
log directory. Use a separate output directory for each domain or track and
pass the same `--out` when resuming or running individual stages.

`STEERING_DATA` selects an external IF/tool-use data root: inputs come from
`$STEERING_DATA/<domain>/inputs/`, and default outputs go to
`$STEERING_DATA/<domain>/outputs/repsteer-native-v2/`. Both tool-use tracks read
`tool_use/inputs/`. `--out` overrides only the output location.

Long-to-short's combined `--entry inference` workflow defaults to
`long-to-short/runs/`. Its pair-construction workflow writes to
`long-to-short/data/l2s/` by default; `--output` selects another pair directory.

For reproduction, record the repository and model revisions, environment, input
manifests, split, sampling seeds, method grid, token budget, and full command.
Store per-example generations, judge outputs, and summaries with the run configuration.

## Repository layout

```text
ControlBench/
├── bench.py                 # Unified launcher
├── instruction-following/   # IF guide, core runtime, domain code, and inputs
├── tool-use/                # Tool-use guide, core runtime, both tracks, and inputs
├── reasoning/               # Mathematical reasoning pipeline
├── safety/                  # Safety classification pipeline
├── long-to-short/           # Pair construction and reasoning compression
├── tests/                   # Launcher and IF/tool-use regression tests
└── THIRD_PARTY_NOTICES.md    # Notices for imported code
```

## Checks and troubleshooting

Run each suite in its domain's environment. The IF/tool-use suite covers IF and both tool-use
tracks, using `bfcl-eval==2026.3.23`, shipped data, mocked judges, and small CPU models.

```bash
# Unified launcher.
python -m unittest discover -s tests -v

# Instruction following.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PWD/instruction-following" \
  python -m unittest discover -s tests/if_tool_use -v

# Tool use.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PWD/tool-use" \
  python -m unittest discover -s tests/if_tool_use -v

# Reasoning.
python -m unittest discover -s reasoning/tests -v

# Safety.
PYTHONPATH=safety python -m unittest discover -s safety/tests -v

# Long-to-short.
(cd long-to-short && PYTHONPATH=src python -m unittest discover -s src/tests -v)
```

| Domain | Input checks and run status |
|---|---|
| Instruction following | `--entry selfcheck`; campaign logs in `<out>/logs/`, or `logs/instruction_following/` by default |
| Tool use | `--entry selfcheck` for either track; campaign logs in `<out>/logs/`, or `logs/<domain>/` by default |
| Reasoning | `status`; per-run `status.json` and `run.log` |
| Safety | `audit`, `status`, and `report` with the run's `--model` and `--out` |
| Long-to-short | `plan`, `judge`, and `report` with the run's output root |

Imported-code notices are collected in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
