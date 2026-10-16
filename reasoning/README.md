# Reasoning steering

Build contrastive pairs from teacher and target-model responses to the same math problems,
extract activations, learn steering directions, select parameters on the development set,
and evaluate the plain baseline, CoT, and steering.
Supported models: Llama-3.1-8B-Instruct, Qwen3-8B, and Gemma-2-9B-IT.

This guide follows **prepare positive/negative examples → configure models → run experiments**.
If you already have the complete datasets, skip to Section 2. Reusing teacher responses
does not require an API key. Run all commands from the repository root containing `reasoning/`.

## 1. Prepare positive and negative examples

### Data and splits

| File | Contents |
|---|---|
| `data/original.jsonl` | The 500 original MATH500 problems and reference answers |
| `data/positive.jsonl` | GPT-5.1 teacher responses in the `solution` field |
| `data/negative_llama.jsonl` | Llama responses in the `completion` field |
| `data/negative_qwen.jsonl` | Qwen responses in the `completion` field |
| `data/negative_gemma.jsonl` | Gemma responses in the `completion` field |
| `data/split.json` | Fixed IDs for 200 extraction / 20 dev / 280 test problems |

**Positive and negative labels indicate response sources, not correctness.**
Correct target-model responses still belong to the negative group; pairs are not filtered
by correctness. Each response file must cover all 500 problems with unique IDs.
All three models share the teacher responses and default split, but each uses its own
negative file. Only the extraction set is used to learn directions; the dev set is used
for parameter selection, and the test set for final evaluation.

### Generate positive examples: teacher responses

The simplest option is to reuse `reasoning/data/positive.jsonl`, without making API calls.
To generate your own responses:

1. Read each problem from `original.jsonl` and ask the teacher model for a complete
   solution ending with the final answer in `\boxed{}`. Generating responses with
   GPT-5.1 requires your own API credentials.
2. Save the teacher response verbatim as `solution`, retaining the corresponding
   `problem` and `id`, in JSONL format. Compute IDs with
   `reasoning.datasets.problem_id(problem)`; do not use row numbers.
3. Once all 500 problems are complete, save the file as `reasoning/data/positive.jsonl`.
   A different teacher or generation configuration constitutes a new data version;
   record the actual model, prompts, and generation parameters.

Each line must contain at least these fields (illustrative values only):

```json
{"id": "hash of the problem text", "problem": "original problem text", "solution": "complete teacher solution"}
```

This minimal version **does not include a teacher-response generation command**.
Call the teacher model yourself and export the format above. Matching the schema does
not guarantee identical responses to the existing dataset. Standard steering experiments
only read this file and do not automatically call a teacher API.

### Generate negative examples: target-model responses

Prepare a CUDA-enabled PyTorch environment with Transformers, math-verify, and
huggingface-hub. Download the model weights and tokenizer at the corresponding revision;
the model path must be a local directory. Obtain access from the model hosting platform
first if authorization is required.

| `--family` | Model ID | Pinned revision |
|---|---|---|
| `llama` | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` |
| `qwen` | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |
| `gemma` | `google/gemma-2-9b-it` | `11c9b309abf73637e4b6f9a3fa1e92e615547819` |

For example, generate Gemma responses on a specific GPU:

```bash
python -m reasoning.scripts.generate_negatives worker \
  --family gemma \
  --model-path /path/to/gemma-2-9b-it \
  --device cuda:0 \
  --output reasoning/outputs/v2/gemma_negatives
```

For another model, change `--family`, `--model-path`, and `--output` together.
This command loads local weights, does not read `models.local.json`, and does not wait
for an idle GPU. Choose a GPU with sufficient free memory. The script records the revision
listed above, so ensure that your local weights match it.

Generation uses greedy decoding, seed 42, batch size 2, and up to 8192 output tokens,
limited by the model's remaining context window. Thinking is disabled for Qwen;
Gemma receives the system instruction within the user message.
This data-generation budget is independent of `--max-tokens` in subsequent experiments.

Intermediate results and metadata are saved under `--output`. Once all responses are
complete, they are exported to `reasoning/data/negative_<family>.jsonl`.
Resume an interrupted run with the same command and output directory.
Generation is skipped if the final dataset already exists and is complete. To regenerate,
back up and move the old file first, then use a new output directory.
Negative files include problems, reference answers, model IDs, responses, token counts,
and scores.

Alternatively, prepare all three models in the background and generate responses
sequentially as GPUs become available:

```bash
python -m reasoning.scripts.generate_negatives start \
  --output reasoning/outputs/v2/all_negatives
```

This mode processes all three models and may download weights over the network.
It selects GPUs with at least 32000 MiB of free memory and utilization at or below 10%.
Progress and logs are saved in the specified directory.

## 2. Run experiments

### Environment and model configuration

Experiments use a CUDA GPU, PyTorch, Transformers, and the bundled `vendor/repsteer`.
Install its declared dependencies and the answer-grading dependency:

```bash
python -m pip install -e ./reasoning/vendor/repsteer
python -m pip install math-verify
```

Use a PyTorch CUDA build compatible with your machine. These commands do not define a
fully locked environment; for strict reproducibility, also pin PyTorch, Transformers,
and other software versions.

Create `reasoning/models.local.json` with your local model paths. For Gemma only:

```json
{
  "open_weight_models": [
    {
      "model": "google/gemma-2-9b-it",
      "revision": "11c9b309abf73637e4b6f9a3fa1e92e615547819",
      "local_path": "/path/to/gemma-2-9b-it"
    }
  ]
}
```

Add entries for other models using the table above; `--model` selects the configuration
by model ID. **No API key is required.** This file is ignored by Git; do not commit
private credentials.

### Configuration check and smoke run

Inspect the model and search space without loading weights:

```bash
python -m reasoning dry-run --model gemma --methods lat
```

A `dry-run` does not validate the full datasets or GPU environment.
For a first run, use a small smoke check:

```bash
python -m reasoning run --model gemma --methods lat --smoke \
  --max-tokens 128 --device cuda:0 \
  --output reasoning/outputs/v2/gemma_smoke
```

The smoke run still requires complete input datasets and loads the model, but reduces
the search and evaluation sizes. Do not use its results for formal reporting.

### Full run

```bash
python -m reasoning run --model gemma --methods lat \
  --layers 14 20 26 32 --strengths 0.5 1 1.5 2 2.5 3 \
  --max-tokens 2048 --device cuda:0 \
  --output reasoning/outputs/v2/gemma_lat
```

The pipeline builds `pairs.jsonl` for the selected model, extracts activations, learns
directions, and selects layers and strengths on dev. It then evaluates on the 280
MATH500 test problems, 40 AMC2023 problems, and 30 AIME2024 problems.
The plain baseline, CoT, and steering are evaluated in one run, without depending on
historical experiment outputs.

Run multiple methods together, or launch in the background:

```bash
python -m reasoning start --model llama --methods caa pca lat probe iti \
  --device cuda:0 --output reasoning/outputs/v2/llama_all
python -m reasoning status
```

### Common options

| Option | Meaning |
|---|---|
| `--model` | `llama`, `qwen`, or `gemma`; configure the model first |
| `--methods` | Defaults to `caa pca lat probe iti`; `diffmean` is an alias for `caa` |
| `--layers` / `--strengths` | Supply both to customize the layer and strength search grid |
| `--extraction-position` | `before_answer` (default), `after_answer`, or `both` |
| `--max-tokens` | Shared dev/test generation budget; currently defaults to 2048 |
| `--device` / `--batch-size` | Defaults to `cuda:2` / `4` |
| `--output` | Results and checkpoints; use a new directory when changing settings |

Layer indices start at 0. Default candidate layers are
`[8,11,14,17,20,23,26]` for Llama, `[9,12,15,18,21,24,27]` for Qwen, and
`[14,20,26,32]` for Gemma. Default strengths are `[0.5,1,1.5,2,2.5,3]`.
ITI selects 48 heads and uses its own strengths `[1,5,10,15,20]`.
SAE supports Gemma only: specify `--methods sae` and
`--sae-weights /path/to/params.npz`, with `--sae-layer` and `--sae-strengths`
to configure its parameters.

`before_answer` extracts before the line containing the final boxed answer;
`after_answer` extracts at the end of the complete response.
`both` searches over both positions jointly. Pairs may be excluded if they lack the
required answer boundary, contain empty text, or exceed the length limit.
The IDs actually used are recorded in the outputs.

The default split is `data/split.json`. Set
`REASONING_SPLIT=/absolute/path/split.json` to use another supported split
(200/20/280 or 160/60/280). The code checks that the sets are disjoint, cover all problems,
and that the actual pairs belong to the extraction set.
After changing the split or examples, relearn directions and retune in a new directory.

### Results and resuming

| Output | Contents |
|---|---|
| `report.md` / `results.json` | Summary results |
| `<method>_selected.json` | Configuration selected on dev |
| `artifacts/` | Learned directions or features |
| `pairs.jsonl` / `extraction*.json` | Pair snapshot and actual extraction problems |
| `*.jsonl` | Per-problem responses |
| `status.json` / `run.log` | Progress / background logs |

Resume with the same configuration and output directory.
Resuming is rejected if input or code hashes have changed.
AMC/AIME data live in `data/benchmarks/`. To download them initially, run
`python -m reasoning.datasets.benchmarks`; this download step also requires
`pyarrow` and network access.
