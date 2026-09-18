# Instruction Following

This domain evaluates whether activation steering makes a model follow a response
format while still answering the user's request. It uses the shared
[`control_bench` runtime](../if-tool-use-core/INSTALL.md), with CLI domain ID
`instruction_following`.

During direction learning, each carrier prompt forms a contrastive pair: the
prompt with the target instruction and the same prompt without it. During
evaluation, steering methods and `unsteered` receive the plain request;
the `prompted` baseline appends the target instruction.

## Targets and data

| Target ID | Requested response format |
|---|---|
| `lowercase` | No uppercase or titlecase characters |
| `fixed_ending` | End with the exact sentence `Hope this helps!` |
| `question_ending` | End with a question |
| `one_sentence` | Exactly one sentence |
| `no_comma` | No commas |

The exact instructions and rule checks live in
[`tasks.py`](src/instruction_following/tasks.py).

| Input under `data/inputs/` | Rows | Use |
|---|---:|---|
| `prompts/extraction.jsonl` | 1,014 | Construct contrastive training pairs for each target |
| `prompts/eval.jsonl` | 1,014 | Evaluate each target on the same carrier prompts |

The two prompt sets are disjoint and come from OpenAssistant OASST1/OASST2.
[`MANIFEST.json`](data/inputs/MANIFEST.json) records source revisions,
construction details, row counts, and SHA-256 checksums. The default evaluation
uses all 1,014 prompts per target; sampling is explicit.

## Setup

Keep this directory together with `tool-use/` and `if-tool-use-core/`.
Run all commands below from the **ControlBench repository root**,
using Python 3.10+:

```bash
python -m venv .venv-if-tool-use
source .venv-if-tool-use/bin/activate
python -m pip install -r if-tool-use-core/requirements.txt
export PYTHONPATH="$PWD/if-tool-use-core${PYTHONPATH:+:$PYTHONPATH}"
```

Commands run directly from the source checkout. Set `PYTHONPATH` again when
opening a new shell, from the repository root.

Generation needs a suitable GPU, a compatible PyTorch CUDA build, and access to
the selected Hugging Face model. Judging uses a separate vLLM environment:

```bash
python -m venv .venv-judge
.venv-judge/bin/python -m pip install -r if-tool-use-core/requirements-judge.txt
export JUDGE_PYTHON="$PWD/.venv-judge/bin/python"
```

The pinned judge requirements describe the existing A100 setup. For another
GPU architecture, select a compatible judge environment with `JUDGE_PYTHON`.
See the [shared installation guide](../if-tool-use-core/INSTALL.md) for environment
details and source execution.

Check imports, dataset checksums, evaluation IDs, and training pairs without
loading subject-model weights:

```bash
python -m control_bench.cli.selfcheck --domain instruction_following
```

## Run experiments

Supported method IDs are `unsteered`, `prompted`, `caa`, `lat`,
`linear_probe`, `pca`, and `iti`. A campaign schedules learning, generation,
judging, and scoring; the two prompt baselines skip learning.

Inspect a small experiment before running it:

```bash
python -m control_bench.cli.campaign --domain instruction_following --model Qwen/Qwen3-8B \
  --methods unsteered,prompted,caa,lat --layers 9 --strengths 0.5,1.0 \
  --train-sample 32 --train-seed 42 --sample 16 --seed 42 \
  --gpus 0 --batch 4 --max-new 512 --dry-run
```

Remove `--dry-run` to execute that sampled campaign. This short generation
budget is for a small check, not the full experiment configuration.

A full-data campaign for the non-ITI methods:

```bash
python -m control_bench.cli.campaign --domain instruction_following --model Qwen/Qwen3-8B \
  --methods unsteered,prompted,caa,lat,linear_probe,pca \
  --layers 9,18,27 --strengths 0.5,1.0,1.5,2.0,2.5 \
  --gpus 0 --batch 16 --max-new 3072
```

The model-specific layer grids used by these experiments are:

| Model ID | Zero-based layers |
|---|---|
| `Qwen/Qwen3-8B` | `9,18,27` |
| `google/gemma-2-9b-it` | `11,21,32` |
| `meta-llama/Llama-3.1-8B-Instruct` | `8,16,24` |

Run ITI separately with its head-selection parameters and alpha grid:

```bash
python -m control_bench.cli.campaign --domain instruction_following --model Qwen/Qwen3-8B \
  --methods iti --iti-top-k 48 --iti-seed 42 --strengths 1,5,10,15,20 \
  --gpus 0 --batch 16 --max-new 3072
```

ITI selects heads across all layers, so omit `--layers` for an ITI-only run.
Use `--targets lowercase,no_comma` to select tasks. `--sample` samples evaluation
items per target; `--train-sample` independently samples training pairs per
target. Their seeds and selections must remain consistent when running stages
separately. `--revision` can pin the model and tokenizer revision.

For individual stages, use `python -m control_bench.cli.run learn|generate|judge|score --method ...`
with the same domain, model, targets, and configuration. Baselines start with
`generate`. Rerunning the same campaign reuses compatible completed stages.

## Evaluation and outputs

A successful answer must pass the target's format rule and receive a coherent,
adequate-answer verdict from the blind Qwen3-8B judge. Reports include
`accuracy`, `rule_rate`, `incoherent_rate`, `partial_rate`, `none_rate`, and
`truncation_rate`. Truncation is recorded separately; missing required judge
verdicts prevent final scoring.

Outputs are written relative to the repository root:

```text
data/instruction_following/outputs/repsteer-native-v2/
├── artifacts/   # Fitted steering artifacts
├── runs/        # Generations, metadata, judge verdicts, scores, summaries
└── cache/
```

Runs retain each method, target, layer/head profile, and strength configuration.
Per-configuration files include `generations.jsonl`, `meta.json`,
`scores.jsonl`, and `summary.json`. Campaign logs are under
`logs/instruction_following/`.

To continue an existing experiment, set `STEERING_DATA` to its absolute data
root. Inputs then come from
`$STEERING_DATA/instruction_following/inputs/`, and outputs remain under
`$STEERING_DATA/instruction_following/outputs/repsteer-native-v2/`.
The default shipped inputs are in this domain directory; generated results
are not committed.

## Code and checks

- [Domain implementation](src/instruction_following/): data loading, rendering,
  task rules, and judging/scoring.
- [Shared runtime](../if-tool-use-core/control_bench/): methods, model loading,
  generation, CLI, and campaign scheduling.
- [Offline tests](../if-tool-use-core/tests/): domain checks and small CPU-model
  integration tests; see the [test command](../if-tool-use-core/INSTALL.md#offline-regression-tests).
