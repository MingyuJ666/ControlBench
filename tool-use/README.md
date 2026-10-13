# Tool Use

This domain measures whether activation steering changes a model's decision to
call a tool, ask for missing information, or abstain when the required tool is
unavailable.

| CLI domain | Evaluation track |
|---|---|
| `tool_use` | Tool-use decisions with scripted follow-ups |
| `tool_use_multi_turn` | BFCL conversations with executable tool calls |

Both tracks use the targets `call`, `clarify`, and `abstain`. They learn from the
same contrastive pairs and store artifacts and evaluation results separately.

## Targets and data

All inputs are under [data/inputs/](data/inputs/).

| File | Call | Clarify | Abstain | Total |
|---|---:|---:|---:|---:|
| `extraction.jsonl` | 2,700 | 2,700 | 2,700 | 8,100 pairs |
| `eval.jsonl` | 1,226 | 873 | 1,010 | 3,109 items |
| `multi_turn.jsonl` | 200 | 200 | 200 | 600 conversations |

Training pairs come from When2Call preference data. Each pair shares a request
and tool list, with two responses taking different actions. For each target,
900 pairs contrast it with each of the other three actions. The fourth action,
`direct`, is used on the negative side of training pairs.

The decision track uses When2Call test items converted to free-form generation.
Clarify and abstain items each have a scripted follow-up, for 1,883 follow-ups
in total. Call items are evaluated on the initial decision.

The executable track uses BFCL v4 `base`, `miss_param`, and `miss_func`
conversations, corresponding to call, clarify, and abstain. Tool calls run against
stateful BFCL backends, and their results enter the conversation.

[MANIFEST.json](data/inputs/MANIFEST.json) records sources, construction,
revisions, counts, and checksums. [multi_turn_corrections.json](data/inputs/multi_turn_corrections.json)
records the local BFCL correction. Each track evaluates its full target pool by
default; `--sample` selects a reproducible subset. Both tracks use `--eval-split full`.

## Setup

Run commands from the **ControlBench repository root**, using Python 3.10+.
The generation environment uses CUDA PyTorch and the selected Hugging Face model.

```bash
python -m venv .venv-if-tool-use
source .venv-if-tool-use/bin/activate
python -m pip install -r tool-use/requirements.txt
python -m pip install 'bfcl-eval==2026.3.23'
```

`bfcl-eval` supplies the executable backends and official checker. Create the
separate vLLM judge environment:

```bash
python -m venv .venv-judge
.venv-judge/bin/python -m pip install -r tool-use/requirements-judge.txt
export JUDGE_PYTHON="$PWD/.venv-judge/bin/python"
```

The pinned judge dependencies target A100 GPUs. `JUDGE_PYTHON` selects the judge
interpreter. `bench.py` uses the active Python environment for generation and
sets the domain's import path.

Check imports, input checksums, evaluation IDs, and training pairs for both tracks:

```bash
python bench.py --domain tool_use --entry selfcheck -- --out outputs/tool_use
python bench.py --domain tool_use_multi_turn --entry selfcheck -- --out outputs/tool_use_multi_turn
```

## Run experiments

Available methods: `unsteered`, `prompted`, `caa`, `lat`, `linear_probe`, `pca`,
`iti`, and `sae`. A campaign runs learning, generation, judging, and scoring.
The `unsteered` and `prompted` baselines start at generation.

The prompted baseline adds a conditional target instruction throughout the
conversation. For example, it asks for clarification while an argument is
missing, then asks for tool use once the argument has been supplied.

Preview a sampled decision-track campaign; remove `--dry-run` to execute it:

```bash
python bench.py --domain tool_use -- \
  --out outputs/tool_use \
  --model Qwen/Qwen3-8B --methods unsteered,prompted,caa,lat \
  --layers 9 --strengths 1.0 \
  --train-sample 32 --train-seed 42 --sample 8 --seed 42 \
  --gpus 0 --batch 2 --max-new 1024 --dry-run
```

For the executable track, use `--domain tool_use_multi_turn` and
`--out outputs/tool_use_multi_turn` with the same experiment settings.
A full-data decision-track campaign is:

```bash
python bench.py --domain tool_use -- \
  --out outputs/tool_use \
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
python bench.py --domain tool_use_multi_turn -- \
  --out outputs/tool_use_multi_turn \
  --model Qwen/Qwen3-8B --methods iti \
  --iti-top-k 48 --iti-seed 42 --strengths 1,5,10,15,20 \
  --gpus 0 --batch 16 --max-new 3072
```

SAE uses the pinned Gemma-2-9B revision and dictionary layers `9,20,31`.
Follow the [SAE configuration and example](../instruction-following/README.md#gemma-sae)
with `--domain tool_use` or `--domain tool_use_multi_turn`.

| Option | Meaning | Default |
|---|---|---|
| `--out` | Output directory for this experiment | Domain output path below |
| `--targets` | Comma-separated target IDs | `call,clarify,abstain` |
| `--sample`, `--seed` | Evaluation items per target and sampling seed | All items; seed 42 when sampling |
| `--train-sample`, `--train-seed` | Training pairs per target and sampling seed | All pairs; seed 42 when sampling |
| `--max-new` | Generated tokens per assistant reply | 3,072 |
| `--revision` | Model and tokenizer revision | Model repository default |

Use `--entry run -- <stage>` for a single `learn`, `generate`, `judge`, or `score`
stage, with `--method`, the same `--out`, and the same experiment settings.
Repeating a campaign reuses compatible completed stages.

## Evaluation and outputs

**Decision track.** A blind Qwen3-8B judge evaluates action, coherence, and
appropriateness. Call success also requires tool-name and argument checks.
Initial and follow-up responses receive separate judgments. Reports include
`accuracy`, `followup_accuracy`, `recovery_rate`, `sticky_rate`, action rates,
and truncation. Follow-up accuracy requires both a correct initial decision and
a successful later call.

**Executable track.** Scoring combines BFCL's execution and no-call checks.
Clarify and abstain conversations also receive a semantic judgment on a
predeclared missing turn, using its history and tool-execution trace. Reports
include combined `accuracy`, `bfcl_accuracy`, and `decision_accuracy`; the last
covers conversations with a semantic judgment.

Final scoring requires all applicable semantic verdicts. Long-context judge
retries preserve the full input and reuse completed verdicts. The retry window
starts at 40,960 tokens, grows in 4,096-token steps, and supports up to 131,072
tokens with YaRN context extension.

`--out DIR` places `artifacts/`, `runs/`, and `cache/` directly under `DIR`.
Campaign logs go to `DIR/logs/<model>/<experiment>/`; `--logs` selects another
log directory. Relative paths use the repository root when launched with
`bench.py`. Use a separate output directory for each domain or track.

Without `--out`, outputs use these paths relative to the repository root:

```text
data/tool_use/outputs/repsteer-native-v2/{artifacts,runs,cache}/
data/tool_use_multi_turn/outputs/repsteer-native-v2/{artifacts,runs,cache}/
```

Each configuration retains `generations.jsonl`, `meta.json`, judge verdicts,
`scores.jsonl`, and `summary.json`. Multi-turn generations include assistant
turns and tool-execution traces. Without `--out`, campaign logs are under `logs/<domain>/`.

Set `STEERING_DATA` to use an external data root. Both tracks then read
`$STEERING_DATA/tool_use/inputs/` and write to their respective domain output
directories beneath that root. `--out` takes precedence for outputs and leaves
the input location unchanged.

## Code and checks

- [Shared tool-use code](src/tool_use/): training pairs, parsing, and rendering.
- [Decision track](src/tool_use/single_turn/): evaluation items, follow-ups, and scoring.
- [Executable track](src/tool_use/multi_turn/): conversation execution, domain hooks, judge rules, and scoring.
- [Core runtime](core/): methods, model loading, generation, and campaign scheduling.
- [Regression tests](../tests/if_tool_use/): data, methods, judging, and execution checks.

`src/tool_use_multi_turn.py` registers the executable track under its CLI domain ID.
[Test commands and environment details](../README.md#checks-and-troubleshooting)
are in the repository README.
