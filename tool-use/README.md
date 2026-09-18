# Tool Use

This domain evaluates whether activation steering changes a model's decision to
call a tool, ask for missing information, or abstain when no suitable tool is
available. It uses the shared [`control_bench` runtime](../if-tool-use-core/INSTALL.md)
and provides two CLI domain IDs:

| CLI domain | Evaluation |
|---|---|
| `tool_use` | Free-form tool-use decisions, with scripted follow-ups for clarify and abstain items |
| `tool_use_multi_turn` | BFCL conversations that execute tool calls and feed their results back to the model |

Both tracks use the same three targets: `call`, `clarify`, and `abstain`.
They use the same contrastive training inputs and store their fitted artifacts
and evaluation outputs separately.

## Data

All shipped inputs are under [`data/inputs/`](data/inputs/):

| File | Call | Clarify | Abstain | Total |
|---|---:|---:|---:|---:|
| `extraction.jsonl` | 2,700 | 2,700 | 2,700 | 8,100 pairs |
| `eval.jsonl` | 1,226 | 873 | 1,010 | 3,109 items |
| `multi_turn.jsonl` | 200 | 200 | 200 | 600 conversations |

Extraction pairs are derived from When2Call preference-training data.
For each target, 900 pairs contrast it against each of the other three actions;
`direct` is a negative-side action, not a steering target. Training pairs share
the user request and tools, and contrast completed assistant responses.

The decision evaluation uses audited When2Call test items converted to free-form
generation. Every clarify and abstain item has a scripted second turn: 1,883
follow-ups in total. Call items have no follow-up.

Executable evaluation uses BFCL v4 `base`, `miss_param`, and `miss_func`
conversations. Local corrections are retained in
[`multi_turn_corrections.json`](data/inputs/multi_turn_corrections.json).
[`MANIFEST.json`](data/inputs/MANIFEST.json) records source revisions,
construction details, counts, and checksums. The current workflow evaluates
the full selected target set; legacy `split` fields do not select a subset.

## Setup

Keep this directory together with `instruction-following/` and
`if-tool-use-core/`. Run all commands below from the **ControlBench repository
root**, using Python 3.10+:

```bash
python -m venv .venv-if-tool-use
source .venv-if-tool-use/bin/activate
python -m pip install -r if-tool-use-core/requirements.txt
python -m pip install 'bfcl-eval==2026.3.23'
export PYTHONPATH="$PWD/if-tool-use-core${PYTHONPATH:+:$PYTHONPATH}"
```

Commands run directly from the source checkout. Set `PYTHONPATH` again when
opening a new shell, from the repository root.

`bfcl-eval==2026.3.23` supplies the executable backends and official checker.
Generation needs a suitable GPU,
a compatible PyTorch CUDA build, and access to the selected Hugging Face model.
Semantic judging uses a separate vLLM environment:

```bash
python -m venv .venv-judge
.venv-judge/bin/python -m pip install -r if-tool-use-core/requirements-judge.txt
export JUDGE_PYTHON="$PWD/.venv-judge/bin/python"
```

The pinned judge requirements describe the existing A100 setup; select an
appropriate environment with `JUDGE_PYTHON` for other GPU architectures.
See the [shared installation guide](../if-tool-use-core/INSTALL.md) for details.

Check both tracks without loading subject-model weights:

```bash
python -m control_bench.cli.selfcheck --domain tool_use
python -m control_bench.cli.selfcheck --domain tool_use_multi_turn
```

## Run experiments

Both tracks support `unsteered`, `prompted`, `caa`, `lat`,
`linear_probe`, `pca`, and `iti`. The prompted baseline adds a conditional
target instruction throughout the conversation: for example, clarification is
requested while an argument is missing, followed by tool use once it is supplied.

Inspect a small decision-track campaign:

```bash
python -m control_bench.cli.campaign --domain tool_use --model Qwen/Qwen3-8B \
  --methods unsteered,prompted,caa,lat --layers 9 --strengths 1.0 \
  --train-sample 32 --train-seed 42 --sample 8 --seed 42 \
  --gpus 0 --batch 2 --max-new 1024 --dry-run
```

Inspect the equivalent executable-conversation campaign:

```bash
python -m control_bench.cli.campaign --domain tool_use_multi_turn --model Qwen/Qwen3-8B \
  --methods unsteered,prompted,caa,lat --layers 9 --strengths 1.0 \
  --train-sample 32 --train-seed 42 --sample 8 --seed 42 \
  --gpus 0 --batch 2 --max-new 1024 --dry-run
```

Remove `--dry-run` to execute a sampled campaign. For a full-data non-ITI run:

```bash
python -m control_bench.cli.campaign --domain tool_use --model Qwen/Qwen3-8B \
  --methods unsteered,prompted,caa,lat,linear_probe,pca \
  --layers 9,18,27 --strengths 0.5,1.0,1.5,2.0,2.5 \
  --gpus 0 --batch 16 --max-new 3072
```

Use `--domain tool_use_multi_turn` for the full executable track.
For Gemma-2-9B use `--model google/gemma-2-9b-it --layers 11,21,32`;
for Llama-3.1-8B use
`--model meta-llama/Llama-3.1-8B-Instruct --layers 8,16,24`.

Run ITI separately; its strength values are alpha values and it selects heads
across all layers:

```bash
python -m control_bench.cli.campaign --domain tool_use_multi_turn --model Qwen/Qwen3-8B \
  --methods iti --iti-top-k 48 --iti-seed 42 --strengths 1,5,10,15,20 \
  --gpus 0 --batch 16 --max-new 3072
```

Omit `--layers` for ITI-only runs. Use `--targets call,clarify` to select targets.
`--sample` selects evaluation items per target; `--train-sample` independently
selects training pairs per target. `--max-new` limits tokens per assistant
reply. Use `--revision` to pin the model and tokenizer revision.

A campaign runs learning, generation, judging, and scoring, skipping learning
for the two prompt baselines. Individual stages use
`python -m control_bench.cli.run learn|generate|judge|score --method ...`; keep the same domain,
model, targets, and sampling/layer/strength configuration at every stage.
Rerunning the same campaign reuses compatible completed stages.

## Evaluation

**Decision track (`tool_use`).** A blind Qwen3-8B judge checks the response's
action, coherence, and appropriateness. Call success additionally requires
deterministic tool-name and argument checks. First responses and scripted
second responses are judged separately. Reports include first-decision
`accuracy`, `followup_accuracy` (correct first decision and successful later
call), `recovery_rate`, `sticky_rate`, action rates, and truncation.

**Executable track (`tool_use_multi_turn`).** Calls execute against BFCL
backends during generation and the observed results enter later prompts.
Scoring uses the official execution and no-call checks. Clarify and abstain
conversations additionally require a semantic verdict on a predeclared missing
turn, using its recorded conversation and tool-result trace. Call/base
conversations use mechanical checks. Reports separate combined `accuracy`,
`bfcl_accuracy`, and `decision_accuracy`; the last applies to conversations
with a semantic judgment.

Incomplete required judge verdicts prevent final scores in both tracks.

## Outputs and code

Outputs are relative to the repository root, separate from shipped inputs:

```text
data/tool_use/outputs/repsteer-native-v2/{artifacts,runs,cache}/
data/tool_use_multi_turn/outputs/repsteer-native-v2/{artifacts,runs,cache}/
```

Run directories retain each model, method, target, and configuration.
They include `generations.jsonl`, `meta.json`, judge verdicts,
`scores.jsonl`, and `summary.json`. Multi-turn generations also retain
assistant turns and actual tool-execution traces. Campaign logs are under
`logs/<domain>/`.

Set `STEERING_DATA` to an existing absolute data root to reuse its inputs and
outputs. Both tracks then read `$STEERING_DATA/tool_use/inputs/`; their output
directories still use their respective CLI domain IDs. Generated artifacts,
results, caches, and environments are not committed.

- [Shared tool-use data, rendering, and parsing](src/tool_use/).
- [Decision-track follow-ups and scoring](src/tool_use/single_turn/).
- [Executable conversations and scoring](src/tool_use/multi_turn/).
- [Multi-turn domain adapter and semantic judge](src/tool_use_multi_turn/).
- [Shared methods and runtime](../if-tool-use-core/control_bench/).
- [Offline tests and installation instructions](../if-tool-use-core/INSTALL.md#offline-regression-tests).
