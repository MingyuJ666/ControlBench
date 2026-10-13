# Instruction Following

This domain measures whether activation steering helps a model follow a response
format while answering the user's request. Its CLI domain ID is
`instruction_following`.

Training pairs contrast the same prompt with and without the target instruction.
At evaluation time, steering methods and `unsteered` receive the plain request;
`prompted` receives the request with the target instruction appended.

## Targets and data

| Target | Response format |
|---|---|
| `lowercase` | No uppercase or titlecase characters |
| `fixed_ending` | Ends with `Hope this helps!` |
| `question_ending` | Ends with a question |
| `one_sentence` | Exactly one sentence |
| `no_comma` | No commas |

[Task definitions](src/instruction_following/tasks.py) contain the exact
instructions and rule checks. All five targets use the same carrier prompts,
derived from OpenAssistant OASST1/OASST2.

| File under `data/inputs/` | Prompts | Purpose |
|---|---:|---|
| `prompts/extraction.jsonl` | 1,014 | Construct training pairs for each target |
| `prompts/eval.jsonl` | 1,014 | Evaluate each target |

The extraction and evaluation sets are disjoint. [MANIFEST.json](data/inputs/MANIFEST.json)
records their sources, construction, revisions, and checksums.

| `--eval-split` | Evaluation rows | Prompts per target |
|---|---|---:|
| `full` (default) | All rows | 1,014 |
| `dev` | Original first 405 rows | 405 |
| `test` | Original remaining 609 rows | 609 |

Sampling applies within the selected split. Splits have separate run directories
and reuse compatible training artifacts.

## Setup

Run commands from the **ControlBench repository root**, using Python 3.10+.
The generation environment uses CUDA PyTorch and the selected Hugging Face model.

```bash
python -m venv .venv-if-tool-use
source .venv-if-tool-use/bin/activate
python -m pip install -r instruction-following/requirements.txt
```

Create the separate vLLM judge environment:

```bash
python -m venv .venv-judge
.venv-judge/bin/python -m pip install -r instruction-following/requirements-judge.txt
export JUDGE_PYTHON="$PWD/.venv-judge/bin/python"
```

The pinned judge dependencies target A100 GPUs. `JUDGE_PYTHON` selects the judge
interpreter. `bench.py` uses the active Python environment for generation and
sets the domain's import path.

Check imports, input checksums, evaluation IDs, and training pairs:

```bash
python bench.py --domain instruction_following --entry selfcheck -- --out outputs/instruction_following
```

## Run experiments

Available methods: `unsteered`, `prompted`, `caa`, `lat`, `linear_probe`, `pca`,
`iti`, and `sae`. A campaign runs learning, generation, judging, and scoring.
The `unsteered` and `prompted` baselines start at generation.

Preview a sampled campaign; remove `--dry-run` to execute it:

```bash
python bench.py --domain instruction_following -- \
  --out outputs/instruction_following \
  --model Qwen/Qwen3-8B --methods unsteered,prompted,caa,lat \
  --layers 9 --strengths 0.5,1.0 \
  --train-sample 32 --train-seed 42 --sample 16 --seed 42 \
  --gpus 0 --batch 4 --max-new 512 --dry-run
```

Run the full training and evaluation sets:

```bash
python bench.py --domain instruction_following -- \
  --out outputs/instruction_following \
  --model Qwen/Qwen3-8B --methods unsteered,prompted,caa,lat,linear_probe,pca \
  --layers 9,18,27 --strengths 0.5,1.0,1.5,2.0,2.5 \
  --gpus 0 --batch 16 --max-new 3072
```

| Model | Layer grid for CAA, LAT, linear probe, and PCA (zero-based) |
|---|---|
| `Qwen/Qwen3-8B` | `9,18,27` |
| `google/gemma-2-9b-it` | `11,21,32` |
| `meta-llama/Llama-3.1-8B-Instruct` | `8,16,24` |

ITI selects attention heads across all layers. Run it with its head count and
alpha grid:

```bash
python bench.py --domain instruction_following -- \
  --out outputs/instruction_following \
  --model Qwen/Qwen3-8B --methods iti \
  --iti-top-k 48 --iti-seed 42 --strengths 1,5,10,15,20 \
  --gpus 0 --batch 16 --max-new 3072
```

| Option | Meaning | Default |
|---|---|---|
| `--out` | Output directory for this experiment | Domain output path below |
| `--targets` | Comma-separated target IDs | All five targets |
| `--eval-split` | Evaluation partition | `full` |
| `--sample`, `--seed` | Evaluation prompts per target and sampling seed | All prompts; seed 42 when sampling |
| `--train-sample`, `--train-seed` | Training pairs per target and sampling seed | All pairs; seed 42 when sampling |
| `--max-new` | Generated tokens per response | 3,072 |
| `--revision` | Model and tokenizer revision | Model repository default |

Use `--entry run -- <stage>` for a single `learn`, `generate`, `judge`, or `score`
stage, with `--method`, the same `--out`, and the same experiment settings.
Repeating a campaign reuses compatible completed stages.

## Gemma SAE

SAE supports `google/gemma-2-9b-it` at revision
`11c9b309abf73637e4b6f9a3fa1e92e615547819`, using dictionary layers **9, 20, 31**.
The same SAE method is available in both tool-use tracks.

Learning selects one Gemma Scope feature per target and layer from the
positive-minus-negative training activation mean. Generation adds the feature's
decoder direction with the residual-norm strength convention. Feature selection
uses the training pairs. Dictionary files are pinned by revision and SHA-256.

```bash
python bench.py --domain instruction_following -- \
  --out outputs/instruction_following \
  --model google/gemma-2-9b-it \
  --revision 11c9b309abf73637e4b6f9a3fa1e92e615547819 \
  --methods sae --layers 9,20,31 --strengths 0.5,1,1.5,2,2.5 \
  --train-sample 100 --train-seed 42 --sample 20 --seed 42 \
  --gpus 0 --batch 4 --max-new 3072 --dry-run
```

Learning downloads approximately 470 MB of dictionary weights per layer.
`CONTROLBENCH_SAE_CACHE` selects their cache directory; the default is under
the directory selected by `--out`, or IF's default output cache. SAE uses the generation environment's existing dependencies.
For tool use, select `--domain tool_use` or `tool_use_multi_turn` and its
corresponding `--out` directory.

## Evaluation and outputs

Success requires both the target's format rule and a coherent, adequate-answer
verdict from the blind Qwen3-8B judge. Summaries report `accuracy`, `rule_rate`,
`incoherent_rate`, `partial_rate`, `none_rate`, and `truncation_rate`.
Final scoring requires the semantic verdicts for every evaluated response.

`--out DIR` places `artifacts/`, `runs/`, and `cache/` directly under `DIR`.
Campaign logs go to `DIR/logs/<model>/<experiment>/`; `--logs` selects another
log directory. Relative paths use the repository root when launched with
`bench.py`. Use a separate output directory for each domain or track.

Without `--out`, outputs use these paths relative to the repository root:

```text
data/instruction_following/outputs/repsteer-native-v2/
├── artifacts/   # Fitted steering artifacts
├── runs/        # Generations, metadata, verdicts, scores, and summaries
└── cache/
```

Each configuration retains `generations.jsonl`, `meta.json`, `scores.jsonl`,
and `summary.json`. Without `--out`, campaign logs are under `logs/instruction_following/`.

Set `STEERING_DATA` to use an external data root. Inputs then come from
`$STEERING_DATA/instruction_following/inputs/`, and outputs go to
`$STEERING_DATA/instruction_following/outputs/repsteer-native-v2/`.
`--out` takes precedence for outputs and leaves the input location unchanged.

## Code and checks

- [Domain implementation](src/instruction_following/): prompts, targets, rendering, and scoring.
- [Core runtime](core/): methods, model loading, generation, and campaign scheduling.
- [Regression tests](../tests/if_tool_use/): data, methods, judging, and execution checks.

[Test commands and environment details](../README.md#checks-and-troubleshooting)
are in the repository README.
