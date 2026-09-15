# CoT / reasoning

A minimal MATH500 runner: normal / CoT prompting and repsteer DiffMean, LAT,
LinearProbe. Four Python files: run.py (CLI/search/evaluation), model.py
(generation/prompts), steering.py (repsteer adapters), data.py (inputs/scoring).

## Setup

Run from the repository root with Python 3.10+:

```bash
python -m venv cot/.venv
source cot/.venv/bin/activate
pip install -r cot/requirements.txt
```

Use a PyTorch build compatible with your GPU. Model IDs or local model directories
are accepted. Tokenizers need a chat template; architectures need repsteer support.
No local credentials or teacher API calls are needed. repsteer is pinned to the
source commit used by the original experiments.

## Prompting

```bash
python cot/run.py eval --model meta-llama/Llama-3.1-8B-Instruct --method normal --output cot/outputs/normal
python cot/run.py eval --model meta-llama/Llama-3.1-8B-Instruct --method cot --output cot/outputs/cot
```

Both use system message: "You are a helpful assistant that solves mathematical problems."

| Method | Instruction appended to the question |
| --- | --- |
| normal | Give a clear mathematical solution. End with the final answer in \boxed{}. |
| cot | Let's think step by step. + the normal instruction |

Normal already requests a solution; this comparison measures the additional
step-by-step instruction. Model-native thinking settings use tokenizer defaults.
All methods use greedy decoding, 2048 new tokens and batch size 4 by default.
Use --max-tokens, --batch-size, --device, --seed and --revision to override defaults.
Pin --revision to a model commit for reproducible downloads.

## Steering

Methods: diffmean (DiffMean), lat (LAT), probe (LinearProbe), all from repsteer.
Capture the last non-padding token of prompt + solution at the chosen residual
block, learn an L2-normalized vector, and add it at generated-token positions
during decoding. Layers are zero-based. Steering uses the normal prompt.
Negative strengths reverse the vector.

Run a predetermined setting:

```bash
python cot/run.py eval --model meta-llama/Llama-3.1-8B-Instruct --method diffmean --layers 16 --strengths 1 --output cot/outputs/diffmean
```

Or search the development split, then load the chosen vector for evaluation:

```bash
python cot/run.py tune --model meta-llama/Llama-3.1-8B-Instruct --method lat --layers 8 11 14 17 20 23 26 --strengths 0.5 1 1.5 2 2.5 3 --output cot/outputs/lat_dev
python cot/run.py eval --model meta-llama/Llama-3.1-8B-Instruct --selection cot/outputs/lat_dev/selected.json --output cot/outputs/lat_eval
```

Repeat with probe or diffmean for other methods. Search uses the same token budget
for all candidates. Ties prefer smaller absolute strength, lower layer, then lower
signed strength. This simple full-dev search differs from the historical runner's
two-stage search. Selection overrides --method and loads the saved artifact
without refitting. Model, revision, input hashes, seed and token budget must match.

## Data

Included data is copied from the existing reasoning experiments:

- data/problems.jsonl: original 500 MATH500 problems, byte-for-byte.
  Source: [MATH500 / PRM800K](https://github.com/openai/prm800k#math-splits).
- data/split.json: original 200 extraction / 20 development / 280 evaluation split.
- data/pairs.jsonl: 56 existing extraction pairs, projected to id, problem,
  positive, negative. Positive is GPT-5.1 teacher text; negative is Llama baseline
  text. These are source groups, not correctness labels. Texts are the existing
  prepared training prefixes, not regenerated full solutions.

The loader checks unique IDs, disjoint splits, and extraction-only training pairs.
The historical dev split has already been used; it is not fresh validation data.
This initial module includes MATH500 only. Supply --data for another directory
with the same schema. IDs are SHA-256 hashes of whitespace-normalized problem text.
Training texts must fit the model's context.

## Outputs and checks

Use a new output directory for each run. Outputs include config/input hashes,
per-question predictions, and accuracy/token/truncation summaries. Steering saves
vectors; tuning also saves scores and selected.json. Interrupted outputs remain
available but this minimal runner does not automatically resume them.

math-verify grades the final balanced \boxed{...}. Missing/incomplete boxes count
as incorrect. Truncation is recorded independently of correctness.

Check inputs without loading a model:

```bash
python cot/run.py eval --model placeholder --method cot --output cot/outputs/check --dry-run
python -m unittest discover -s cot/tests -v
```

Tests use a tiny random local model on CPU. For a generation smoke test add
--limit 2 --max-tokens 32; this is not a benchmark result.
