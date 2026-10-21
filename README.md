<h1 align="center">RealControlBench: A Benchmark for Interpretability-Informed Fine-Grained Control of LLMs and Agents</h1>

<h3 align="center">Evaluating behavioral control and task performance across five domains of LLMs and agents.</h3>

<p align="center">
  <a href="https://github.com/MingyuJ666/ControlBench">Code</a> · <a href="https://huggingface.co/datasets/Mingyu111/controlbench">🤗 Hugging Face Dataset</a>
</p>

## ❗ Abstract

Activation steering provides a way to control LLM behavior by intervening on
internal representations during inference. We introduce **RealControlBench** to
systematically evaluate steering across **instruction following, agentic tool use,
reasoning performance, reasoning-length control, and trajectory-conditioned
safety awareness**. The benchmark measures both target controllability and the
model's ability to complete the underlying task. It covers response constraints,
tool-use decisions and execution, mathematical reasoning, reasoning compression,
and action-safety judgments. Evaluation extends from individual responses to
stateful multi-turn interactions and transfer across reasoning benchmarks. We
compare six steering methods with unsteered and prompted baselines on three
models. Our experiments show that existing methods often fail to reliably induce
the target behavior and can degrade task performance. Gains in initial tool-use
decisions do not consistently carry over to follow-up execution, reasoning gains
can fail to transfer to new problems, and shorter reasoning can reduce answer
accuracy. RealControlBench provides a testbed for studying how representation-level
interventions translate into useful control of LLMs and agents.

**Methods:** CAA, PCA, LAT, Linear Probe, ITI, and SAE, together with unsteered
and prompted baselines.

**Models:** Qwen3-8B, Llama-3.1-8B-Instruct, and Gemma-2-9B-it. SAE experiments
use Gemma Scope dictionaries on Gemma-2-9B-it.

## 🤗 Environment Setup

Experiments use CUDA GPUs and pretrained model checkpoints. Clone the repository:

```bash
git clone https://github.com/MingyuJ666/ControlBench.git
cd ControlBench
```

Prepare dependencies and model access using the setup guide linked in each
task section below. The examples run from the repository root.

## 🚀 Getting Started

List the available experiment workflows:

```bash
python bench.py --list
```

### 📊 1. Instruction Following

We use OpenAssistant questions to evaluate five response constraints:
**lowercase, one sentence, no comma, question ending, and a fixed ending**.
Steering directions are learned from paired prompts with and without the target
instruction, then applied to new questions. Success requires satisfying the
constraint, producing a coherent response, and adequately answering the question.

Follow the [instruction-following setup](https://github.com/MingyuJ666/ControlBench/blob/main/instruction-following/README.md)
to configure the model and judge environments. Run the five targets with Qwen3-8B:

```bash
python bench.py --domain instruction_following -- \
  --model Qwen/Qwen3-8B \
  --methods unsteered,prompted,caa,lat \
  --targets all --layers 9,18,27 --strengths 0.5,1,1.5,2,2.5 \
  --gpus 0 --out outputs/instruction-following
```

The workflow learns steering directions, generates responses, runs the judge,
and saves scores for each target and configuration.

**Core parameters:**

- `--targets`: `all`, or a comma-separated selection of `lowercase`,
  `one_sentence`, `no_comma`, `question_ending`, and `fixed_ending`.
- `--methods`: methods to compare.
- `--layers` and `--strengths`: intervention layers and steering coefficients.

### 🛠️ 2. Agentic Tool Use

We evaluate three decisions: **call** an available tool, **clarify** missing
arguments, and **abstain** when the required tool is unavailable. The single-turn
track uses When2Call, including follow-up turns that supply missing arguments or
tools. The multi-turn track uses BFCL v4 conversations with executable,
stateful tool interactions. Both tracks learn steering directions from
When2Call contrastive examples.

Complete the [tool-use setup](https://github.com/MingyuJ666/ControlBench/blob/main/tool-use/README.md),
including the judge and BFCL backend, then run either track.

**Single-turn decisions and follow-ups:**

```bash
python bench.py --domain tool_use -- \
  --model Qwen/Qwen3-8B \
  --methods unsteered,prompted,caa,lat \
  --targets all --layers 9,18,27 --strengths 0.5,1,1.5,2,2.5 \
  --gpus 0 --out outputs/tool-use
```

**Multi-turn execution:**

```bash
python bench.py --domain tool_use_multi_turn -- \
  --model Qwen/Qwen3-8B \
  --methods unsteered,prompted,caa,lat \
  --targets all --layers 9,18,27 --strengths 0.5,1,1.5,2,2.5 \
  --gpus 0 --out outputs/tool-use-multi-turn
```

Use `--targets call,clarify,abstain` to select the decision types.
Single-turn results report initial decision and follow-up success. Multi-turn
results combine BFCL execution checks with the required clarification or
abstention behavior.

### 🧠 3. Reasoning Performance

We pair teacher and target-model solutions to the same MATH500 problems to
learn reasoning interventions. The pipeline selects steering configurations on
development data, evaluates held-out MATH500 accuracy, and tests transfer to
**AMC 2023 and AIME 2024**. Final answers are checked for mathematical
equivalence to the references.

Follow the [reasoning setup](https://github.com/MingyuJ666/ControlBench/blob/main/reasoning/README.md)
and register local model checkpoints in `reasoning/models.local.json`.
The input responses and benchmark problems are in `reasoning/data/`.

```bash
python bench.py --domain reasoning -- \
  run --model qwen --methods caa pca lat probe iti \
  --device cuda:0 --output outputs/reasoning-qwen
```

**Core parameters:**

- `--model`: `qwen`, `llama`, or `gemma`.
- `--methods`: steering methods to evaluate; the run also evaluates the plain
  and chain-of-thought baselines.
- `--output`: directory containing learned artifacts, generated answers,
  `results.json`, and `report.md`.

### 📏 4. Reasoning-Length Control

We construct matched long and short solutions to the same mathematical problems
and learn interventions that encourage concise reasoning. Evaluation on
**MATH500** measures reasoning-token length together with answer accuracy.

Use the [long-to-short setup](https://github.com/MingyuJ666/ControlBench/blob/main/long-to-short/README.md)
to configure pair construction, inference, and judging.

**Step 1: Construct long/short pairs.** The GPT-5.1 teacher produces a detailed
solution, extracts its essential intermediate steps, and rewrites it as a concise
derivation. Accepted pairs are saved to `long-to-short/data/l2s/`.

```bash
python bench.py --domain long_to_short --entry pairs --
```

**Step 2: Run steering experiments.** For Qwen3-8B, the following sequence learns
CAA, PCA, LAT, and Linear Probe directions, generates baseline and steered
responses, judges answers, and produces a report:

```bash
for stage in fit baselines grid judge report; do
  python bench.py --domain long_to_short --entry non-iti -- \
    "$stage" --model Qwen3-8B --device cuda:0 || break
done
```

Set `--entry iti` to run the ITI track. Reports contain answer accuracy and
reasoning-length changes relative to the unsteered baseline.

### 🛡️ 5. Safety Awareness

We evaluate whether a model can judge a candidate agent action as **safe or
unsafe** given the task and preceding interaction. Examples come from executable
agent trajectories on benign OS tasks and harmful AdvCLI tasks. Each input
contains the preceding context and the candidate action. Evaluation reports
classification accuracy and unsafe F1.

Follow the [safety setup](https://github.com/MingyuJ666/ControlBench/blob/main/safety/README.md)
and replace `/path/to/gemma-2-9b-it` with the local checkpoint directory.

**Evaluate the unsteered and prompted baselines:**

```bash
python bench.py --domain safety -- \
  all --model gemma2_9b \
  --model-path /path/to/gemma-2-9b-it \
  --config safety/configs/reference.json \
  --out outputs/safety-gemma-reference
```

**Evaluate CAA, PCA, LAT, Linear Probe, and ITI:**

```bash
python bench.py --domain safety -- \
  all --model gemma2_9b \
  --model-path /path/to/gemma-2-9b-it \
  --config safety/configs/steering.json \
  --out outputs/safety-gemma-steering
```

Each run writes predictions and exports `summary.json`, `summary.csv`, and
`summary.md` to its output directory.
