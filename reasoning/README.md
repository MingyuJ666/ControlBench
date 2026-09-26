# Reasoning benchmark

Compares an unsteered baseline, CoT prompting, and representation steering on LLM
mathematical reasoning. Evaluation sets are MATH500, AMC2023, and AIME2024; generation
and interventions use Transformers / repsteer.

## Pipeline

`prepare data → extract activations → learn directions / select features → tune on dev → test with a fixed config`

The extraction, dev, and test sets are kept disjoint. Teacher and target-model answers
form positive/negative pairs; the label indicates the source, not correctness. The default
MATH500 split is 200/20/280; the extended Qwen pipeline uses 160/60/280. The dev set is
used to select settings; the test set is only used to report results.

## Code layout

| Location | Responsibility |
|---|---|
| `__main__.py`, `cli.py` | Command dispatch (`python -m reasoning <command>`) and `status` |
| `experiments/` | Workflows: `tune`, `compare`, `evaluate`, `sample`, `reverse`, `teacher` (teacher-repair), `sae` (SAE feature learning for the suite) |
| `datasets/` | MATH500 loading and IDs, extraction-position text truncation, model-specific pairs (`contrastive.py`), AMC/AIME download and split audit (`benchmarks.py`) |
| `steering.py` | Learners (DiffMean / LAT / LinearProbe), activation extraction, generation-time intervention |
| `models.py` | Transformers and teacher API adapters, chat-template prompts |
| `evaluation.py` | Boxed-answer scoring (`math_verify`), checkpointed evaluation loop, metrics |
| `baseline_cache.py` | Reuse of compatible unsteered completions across runs |
| `common.py` | Paths, default split, prompts, search grids, local config, checkpoints, background workers |
| `scripts/` | `paper_baselines.py` (the `suite` command), `qwen_pipeline.py`, `generate_negatives.py`, `aligned_endings.py`, `recheck_budget.py` |
| `vendor/repsteer/` | Pinned upstream steering algorithms and hook implementations |
| `tests/` | Unit checks; no GPU inference or paid API requests |
| `data/` | Input data (see [Data](#data)) |
| `outputs/` | Experiment artifacts; `v2/` for new runs, `archive/` for historical data |
| `models.local.json` | Local model config and API credentials; never commit or print |

Workflow changes go in `experiments/`, model calls in `models.py`, and steering methods in
`steering.py`. Shared logic lives directly in the relevant module.

## Usage

Run the commands below from the directory that contains `reasoning/`. You need a working
PyTorch / Transformers environment plus local models and data. On this machine the
interpreter is `reasoning/.venv-vllm/bin/python`; the name is historical and every entry
point here uses Transformers, not vLLM.

- **Model**: the first entry of `open_weight_models` in `reasoning/models.local.json`
  (`model`, `revision`, `local_path`). Only `suite` can pick another entry
  (`--model-index`); `evaluate` can override it with `--model-path --model-id --model-revision`.
- **Device**: every command has its own `--device` default (`cuda:2` or `cuda:3`), so the
  `device` field in `models.local.json` is not used by the CLI. Pass `--device` explicitly.
- **Split**: `outputs/archive/data/math500/steering_run/split.json` unless
  `REASONING_SPLIT=/absolute/path/split.json` is set before launch.
- **GPU wait**: runs record `waiting_for_gpu` in `status.json` and block until the device
  has at least 24 GiB free.

```bash
python -m reasoning --help

# Inspect the search grid without running a model
python -m reasoning tune dry-run --method lat

# Search the given layers and strengths; `start` runs in the background, `run` in the foreground
python -m reasoning tune start --method lat \
  --layers 14 20 26 32 --strengths 0.5 1 1.5 2 2.5 3 \
  --device cuda:0 --output reasoning/outputs/v2/lat_run

# Test with the vector and settings selected on dev
python -m reasoning evaluate start \
  --tuning-dirs reasoning/outputs/v2/lat_run \
  --device cuda:0 --output reasoning/outputs/v2/lat_eval

# Unsteered baseline
python -m reasoning evaluate start --baseline-only \
  --device cuda:0 --output reasoning/outputs/v2/baseline

python -m reasoning status
```

## Entry points and tuning

| Entry point | Actions | What it does |
|---|---|---|
| `tune` | `start` `run` `status` `dry-run` | DiffMean / LAT / LinearProbe; layer × strength search on dev in two stages |
| `compare` | `start` `run` `status` | DiffMean / LAT / LinearProbe at one layer (default 16, alphas `-1,0.5,1.5`) on up to 20 dev problems; use `tune` for full search |
| `suite` | `start` `run` `status` `dry-run` | CAA/DiffMean, PCA, LAT, Probe, ITI, SAE; joint search on the full dev set, then test |
| `evaluate` | `start` `run` `status` `dry-run` | Fixed vectors on MATH500 / AMC / AIME, baseline, CoT baseline |
| `sample` | none (runs in the foreground) | Baseline vs. steering on the first test problem; needs a fresh `--output` |
| `reverse` | `start` `run` `status` | Fixed historical DiffMean L16, α = −1 on the 280 test problems at 8192 tokens, with a direction/sign audit |
| `teacher-repair` | none (`--dry-run` only checks) | Retry missing teacher exports once via the API |
| `status` | none | Summarize `status.json` of runs directly under `outputs/` and `outputs/v2/` |

**`tune` stages.** Screen: every grid candidate plus a baseline on the first 8 dev problems.
Confirm: the top 3 screen candidates (ties: smaller |α|, then label) plus a baseline on the
remaining dev problems. Both stages use the same budget (see [Generation budget](#generation-budget)). `selected_steering.json` is the best
steering candidate on confirmation, `selected.json` is the best condition including the
baseline. `--eval-limit N` additionally evaluates baseline and the selected candidate on a
fixed seed-42 sample of N test problems. `--smoke` keeps two candidates and 2 + 2 dev
problems at 32 tokens. All tune candidates use the decode phase (Add on generated tokens).

**`evaluate` conditions.** With `--tuning-dirs`, each directory must be a completed `tune`
run (`config.json` + `selected_steering.json`), one per method; suite outputs are not
accepted. Without `--tuning-dirs` and without `--baseline-only`, it uses the historical
hard-coded vectors in `experiments/evaluate.py::CONDITIONS` (DiffMean L24 α 0.5, LAT L16
α 0.5). `--datasets` selects a subset of `math500 amc2023 aime2024`. A tune directory whose
recorded `confirmation_budget` differs from `--max-tokens` is rejected, so a setting is
never tested at a budget other than the one it was selected at. `evaluate` and `suite` both audit the split against the historical
`outputs/repsteer_tuning/config.json`, which must exist.

In the suite, CAA maps to DiffMean (pass one of `caa` or `diffmean`, not both). SAE uses
Gemma Scope JumpReLU weights and currently supports Gemma-2-9B only.

| What to change | Flag or code location |
|---|---|
| Layer and strength search range | `tune --layers ... --strengths ...` (must be given together) |
| Default search grid | `LAYER_GRID`, `STRENGTH_GRID` in `common.py` |
| Single-layer comparison | `compare --layer 20 --alphas=0.5,1,1.5` |
| ITI strength / number of heads | `STRENGTH_GRID_FOR_ITI` / `fit(top_k=48)` in `scripts/paper_baselines.py` |
| Method-internal parameters | `steering.py::learner`; PCA/ITI in `scripts/paper_baselines.py::fit` |
| Activation extraction position | `--extraction-position before_answer\|after_answer`; `tune` / `suite` also support `both` |
| Intervention position and phase | `steering.py::generate`, the suite's `control_plan`; always decode / generated tokens as launched from the CLI |
| GPU / generation batch size | `--device` / `--batch-size` (default 4) |
| Generation length | `MAX_TOKENS` in `common.py`, or `--max-tokens` on `tune` / `suite` / `evaluate` / `compare` / `sample` (see below) |
| Chat-template thinking | `evaluate --thinking default\|enabled\|disabled`; `suite` and the Qwen pipeline disable Qwen thinking automatically; other entry points use the template default |
| Prompts | `SYSTEM`, `SUFFIX`, `COT_SUFFIX` in `common.py` |
| Data split | `REASONING_SPLIT=/absolute/path/split.json`, set before launch |

### Generation budget

Every stage uses one budget: `MAX_TOKENS` in `common.py` (currently 8192). It is the default
`--max-tokens` of `tune` (screen and confirmation), `suite` (dev selection and tests),
`evaluate`, `compare`, and `sample`, and the Qwen pipeline passes it to the suite. Change it
there once rather than per command; passing `--max-tokens` overrides it for one run and is
recorded in that run's config. `tune --smoke` uses 32 tokens. Two entry points keep their
own budget on purpose: `reverse` replays a historical setting at 8192, and
`scripts/recheck_budget.py` exists to re-test a selection at a different budget.

Gemma has no system role, so its system prompt is prepended to the user message, and its
generation budget is capped at the remaining context window.

### Search grids

Layers are 0-based block indices: `resid_post(20)` is the output of the 21st block. `tune`
and `suite` pick the grid from the model ID:

| Model | Layers |
|---|---|
| Qwen3-8B | 9, 12, 15, 18, 21, 24, 27 |
| Gemma-2-9B | 14, 20, 26, 32 |
| Llama-3.1-8B | 8, 11, 14, 17, 20, 23, 26 |

All models use strengths `[0.5, 1, 1.5, 2, 2.5, 3]`, giving 42 candidates for Qwen/Llama
and 24 for Gemma (doubled with `--extraction-position both`). DiffMean, LAT, LinearProbe,
PCA, and CAA share this grid. ITI selects 48 heads instead of a layer and uses its own
strength grid `[1, 5, 10, 15, 20]`. SAE uses `--sae-layer` and `--sae-strengths`
(default `0.1 0.25 0.5 0.75 1 1.5 2 3 5 10 20 40`).

```bash
python -m reasoning tune start --method lat --device cuda:6 --output reasoning/outputs/v2/lat_grid
python -m reasoning tune start --method probe --device cuda:1 --output reasoning/outputs/v2/probe_grid
# Evaluate the layer, strength, and vector each run selected on dev
python -m reasoning evaluate start --tuning-dirs reasoning/outputs/v2/lat_grid reasoning/outputs/v2/probe_grid \
  --device cuda:6 --output reasoning/outputs/v2/steering_grid_eval
```

### Activation extraction position

`tune`, `compare`, `sample`, `suite`, and the Qwen pipeline support
`--extraction-position`, set independently of the layer (only `tune` and `suite` accept
`both`):

- `before_answer` (default): find the last `\boxed{`, drop that whole line and everything
  after it, strip trailing whitespace, and extract at the last non-padding token of the
  remaining text. Pairs with no boxed answer or empty text after truncation are excluded.
- `after_answer`: keep the full response (including the final answer and anything after
  it), strip trailing whitespace, and extract at the last token. No boxed answer is
  required; empty responses are still excluded. This extracts at the end of the full
  response rather than at the closing brace of the boxed answer.

Both modes keep the same length limit (see [Data](#data)). Extraction position and the
generation-time intervention phase are separate settings. The position is recorded in the
run config and vector metadata, and a saved vector with a different or missing position is
rejected, so switching positions requires a new `--output`.

Before- and after-answer positions can be searched jointly in a single `tune` job:

```bash
python -m reasoning tune dry-run --method lat --extraction-position both
python -m reasoning tune start --method lat --extraction-position both --device cuda:0 --output reasoning/outputs/v2/lat_both
python -m reasoning evaluate start --tuning-dirs reasoning/outputs/v2/lat_both --device cuda:0 --output reasoning/outputs/v2/lat_both_eval
```

One process loads the model once, learns vectors for both positions in turn, and jointly
searches over position × layer × strength. The screen stage ranks candidates from both
positions together and sends the top three to confirmation; a single config is then chosen
on dev. All stages share one baseline. This is a sequential single-process run, not
dual-GPU parallelism or a sum of the two positions' vectors. Vectors are stored under
`artifacts/<position>/L<layer>` (single-position runs: `artifacts/L<layer>`), condition
labels are prefixed by position, and `selected_steering.json` records the chosen position,
which `evaluate --tuning-dirs` reads automatically. Because empty text or the length limit
can drop different pairs per position, the kept pair IDs are written to
`extraction_<position>.json` (single-position runs: `extraction.json`).

### Suite

`suite` accepts Llama-3.1-8B-Instruct, Qwen3-8B, and Gemma-2-9B-IT. It first runs the plain
and CoT baselines on each test set and on dev, then for each method learns vectors, scores
every candidate on the full dev set, and selects the best one (most correct; ties: smaller
α, then lower layer). A steering candidate is always selected and tested on MATH500,
AMC2023, and AIME2024, even when it does not beat the baseline. Baselines are not repeated
per extraction position.

```bash
python -m reasoning suite dry-run --methods caa pca lat probe iti --extraction-position both
python -m reasoning suite start --methods caa pca lat probe iti --extraction-position both --device cuda:0 --output reasoning/outputs/v2/suite_both
```

To include SAE, use `--methods caa pca lat probe iti sae` and pass
`--model-index <index of Gemma in open_weight_models>` and
`--sae-weights /path/to/layer_20/.../params.npz`. `--sae-layer` defaults to 20; if the
weight path contains `layer_<N>`, it must match. SAE is not in the default method list.

Outputs: vectors under `artifacts/<method>/L<layer>` (`artifacts/<method>/<position>/L<layer>`
with `both`), per-candidate dev files `dev_<method>_<label>.jsonl`, `<method>_dev.json`,
`<method>_selected.json`, per-test files `<dataset>_<method>.jsonl`, and `results.json` /
`report.md`. No separate `evaluate` call is needed. Other flags: `--smoke` (one candidate,
2 dev and 1 test problem per set, 8 pairs), `--wait-pid PID` (queue behind another process;
the run refuses to start if the source changed while queued). `python -m
reasoning.scripts.paper_baselines` is the same entry point.

### CoT baseline

The steering baseline runs Transformers/repsteer with interventions off; its prompt already
asks: `Give a clear mathematical solution. End with the final answer in \boxed{}.`
The CoT baseline prepends `Let's think step by step.` to the same instruction as a separate,
unsteered prompt control. Model, problem set, generation budget, batch size, and scoring
are identical, so it measures the gain from the added prompt alone.

```bash
# Check the config only
python -m reasoning evaluate dry-run --baseline-only --cot-baseline --max-tokens 2048 --batch-size 16
# Run the plain and CoT baselines once each (350 problems)
python -m reasoning evaluate start --baseline-only --cot-baseline --max-tokens 2048 --batch-size 16 \
  --device cuda:6 --output reasoning/outputs/v2/cot_baseline_2048
```

In `evaluate`, `--cot-baseline` adds the CoT condition; to run it alongside tuned vectors,
drop `--baseline-only` and pass `--tuning-dirs` (`--baseline-only` cannot be combined with
`--methods` or `--tuning-dirs`). `--cot-only` runs only the CoT condition; gains and losses
are then reported as `—` and must be compared by ID against an existing baseline with the
same settings. `suite` always runs both baselines.

Per-problem files:

| | Plain baseline | CoT baseline |
|---|---|---|
| `evaluate` | `<dataset>_baseline_baseline.jsonl` | `<dataset>_cot_baseline.jsonl` |
| `suite` (test) | `<dataset>_baseline.jsonl` | `<dataset>_cot_baseline.jsonl` |
| `suite` (dev) | `dev_baseline.jsonl` | `dev_cot_baseline.jsonl` |

They are labeled `baseline` and `cot` in reports. The prompt is saved in the run config and
checkpoint metadata, and baseline reuse is keyed by prompt; changing the prompt or code
requires a new output directory.

### Baseline reuse

The unsteered conditions of `evaluate`, `tune`, `suite`, and `recheck_budget` look for
existing completions in `*baseline*.jsonl` files directly under `outputs/` and `outputs/v2/`
or one directory below them, plus `data/negative_llama.jsonl` (below). Every other source
needs a `.meta.json` next to it. A source is used only if
its metadata matches on model, revision, backend, system prompt, prompt suffix, generation
budget, batch size, seed, thinking setting, Gemma prompt/budget policy, and greedy decoding.
Rows are aligned by problem ID, problem text, and reference answer, and reused completions
are re-scored with the current scorer. Only missing problems are generated. When
everything hits, `evaluate --baseline-only` loads no model and does not wait for a GPU.
`compare` does not reuse baselines.

`data/negative_llama.jsonl` has no metadata file; it is accepted only if it matches the
original `outputs/llama31_8b_vllm_math500_8192.jsonl`, whose metadata records the vLLM
backend, so it never matches a Transformers run. Files without metadata or with mismatched
settings are skipped. Each reused row's `baseline_source` records the source path, SHA256,
metadata SHA256, and settings; `Reused` in the evaluate report is the number of reused
problems.

## Data

### Positive/negative pairs

- `data/original.jsonl`: original MATH500, 500 problems, byte-identical to the source.
- `data/positive.jsonl`: GPT-5.1 teacher solutions (`solution` field), 500 problems.
- `data/negative_llama.jsonl`: historical Llama-3.1-8B-Instruct vLLM solutions, 500 entries,
  kept for baseline reuse and reverse validation of old vectors. `data/negative.jsonl` is a
  compatibility symlink to it.
- `data/negative_gemma.jsonl`: Gemma-2-9B-IT solutions, 500 entries.
- `data/negative_qwen.jsonl`: Qwen3-8B solutions with thinking disabled, 500 entries.

The negative files are produced by `python -m reasoning.scripts.generate_negatives`:
`start` downloads the pinned revisions of all three models and generates each on the first
GPU with at least 32000 MiB free and at most 10% utilization; `worker --family
llama|gemma|qwen --model-path ... --device ... --output ...` generates one model directly.
Checkpoints, their `.meta.json`, and `summary.json` stay in the run's `--output`; only the
finished file is copied to `data/negative_<family>.jsonl`. If that file already exists with
all 500 problems from the right model, the worker skips generation; otherwise it refuses and
the old file must be moved away first.

Positive/negative denotes the source group for steering, not whether each answer is correct;
existing fields such as `correct` are kept, and the teacher's historical `negative` label is
stored in `source_label`. `tune`, `compare`, `sample`, and `suite` build pairs from
`positive.jsonl` and the negative file for the configured model. Both files must cover all
500 problems and every negative row must have a `model` field equal to the configured model;
otherwise the run is rejected, with no fallback.

Pairs use every problem in the extraction split (200, or 160 for the Qwen split), not
filtered by correctness. Pairs longer than min(16384, model context) tokens are dropped, so
the actual count can be lower (at least 8 are required). Each run snapshots its pairs to
`pairs.jsonl` + `.meta.json` in the output directory and refuses to resume if the sources
changed; after any data change, relearn vectors and retune in a new output directory.

The archived pairs next to the default split (`outputs/archive/data/math500/steering_run/
pairs.jsonl`, the old 56-problem subset) are still read by `reverse`, by `aligned_endings`,
and by the split audit in `evaluate` / `suite`. Historical API responses, splits, vectors,
and logs live in `outputs/archive/data/math500/`.

`teacher-repair` reads `outputs/archive/data/math500/gpt51_teacher` and `data/positive.jsonl`,
retries missing problems once with 65536 output tokens, and writes `positive.merged.jsonl`
in its `--output` (default `outputs/v2/teacher_repair`). It never modifies
`data/positive.jsonl`. Credentials come from `models.local.json` or `OPENAI_API_KEY` /
`OPENAI_BASE_URL`; `--prompt-key` reads a key from the terminal.

### AMC / AIME

Loading code is in `datasets/`; data is in `data/benchmarks/`. On first use run:

```bash
python -m reasoning.datasets.benchmarks
```

This downloads pinned Hugging Face revisions of AMC 2023 (40 problems, AMC 12A/B) and
AIME 2024 (30 problems, I and II) and saves JSONL, the raw Parquet, and a `manifest.json`
with checksums. If a manifest already exists, it verifies the local data instead. These two
sets are test-only, and exact problem overlap with MATH500 is rejected.

### Qwen extended dev set

```bash
python -m reasoning.scripts.qwen_pipeline start --output <new output dir> --device cuda:N
```

It derives a 160/60/280 split from the default one: the original 20 dev problems plus the
last 40 extraction problems (in their fixed order) become dev; test IDs and order are
unchanged. It writes `data/qwen/split.json` and `data/qwen/pairs.jsonl` (not present until
the pipeline runs), then:

1. waits for `--wait-pid` to exit (the default is a fixed historical PID; pass your own) and
   for the GPU to have at least 32000 MiB free and at most 10% utilization;
2. runs `generate_negatives` for Qwen (skipped when `data/negative_qwen.jsonl` is complete);
3. runs the suite (`caa pca lat probe iti`, `MAX_TOKENS` for dev and test, batch size 4) in
   `<output>/suite` with `REASONING_SPLIT` set to the Qwen split.

Other entry points can use the same split via
`REASONING_SPLIT=<absolute path>/reasoning/data/qwen/split.json`. Changing the split
requires re-extraction and retuning in a new output directory.

## Other scripts

| Script | Purpose |
|---|---|
| `scripts/recheck_budget.py` | Re-run a completed tune selection on its same test IDs at a new `--max-tokens` (default 4096); `--source <tune dir>` |
| `scripts/aligned_endings.py` | Ablation of the pair ending (reasoning prefix without trailing display delimiters) for a fixed DiffMean tune selection; `prepare` previews the alignment |

## Results, verification, and commits

Use a separate `--output` for each experiment; new runs go under `outputs/v2/` and store the
config, artifacts, per-problem JSONL, reports, `status.json`, and `run.log` (background
runs). Resuming requires identical config, including input file hashes and hashes of the
top-level, `datasets/`, and `experiments/` source files (`suite` also hashes its own script);
otherwise the run is refused. After any change, use a new output directory and do not
modify the source of a running job. One worker per output directory is enforced by a lock.

Old scripts, including the old vLLM `baseline` command, have been removed; use
`evaluate start --baseline-only` for the plain baseline. Historical data and results remain
in `outputs/`; the commands in `outputs/archive/data/math500/RUN_INFO.md` are records only,
and some `outputs/v2/*/launch.py` files record how individual runs were launched.

```bash
python -m unittest discover -s reasoning/tests -v
```

Commit source, docs, tests, and vendored dependencies, keeping upstream licenses.
Do not commit `models.local.json`, virtual environments, or `outputs/`; they are excluded
in `.gitignore`. Full reproduction also requires the data and the fixed split; inputs in
historical outputs are not migrated with the code.
