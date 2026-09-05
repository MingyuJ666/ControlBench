# tool_use_steering

A benchmark for **steering the tool-use decision**. At a decision moment an
agent model must either call a tool, ask the user for a missing parameter, or
say that no available tool can help. We measure whether an intervention —
a system prompt, or an activation-level steering method — can move that
decision in a chosen direction *without damaging everything else*: the other
two decisions, the well-formedness of the calls, and the model's ability to
act on the user's answer one turn later.

This is a **method-comparison benchmark**, not a methods paper: many steering
methods exist, each evaluated in its own setting; here they meet on one exam,
one judge, one fairness protocol. It is also not a capability benchmark —
BFCL asks whether a model *can* function-call; we ask whether its *choice* of
when to call can be steered, and at what price.

> **Status (2026-08-14).** Data, evaluation machinery, both tracks and nine
> method arms are built and verified; the single-turn dose grids for dim/caa
> are complete and the remaining methods' grids are running. Numbers in
> "Findings so far" are dev-subsample results (100 items/task) and will be
> finalised on the full pools; nothing has touched the test split.

## The decision setting

One item is one decision moment: a single user request, a list of tool
schemas, one correct action. Three tasks, one per action (the code names them
with words, deliberately — see `schema.py`):

| Task | Situation | Correct first turn |
|---|---|---|
| `call` | request complete, a tool fits | emit the tool call |
| `clarify` | a *required* parameter is missing | ask the user for that parameter |
| `abstain` | no available tool can do it | say so; call nothing |

Every clarify item's withheld parameter is verified to be **required with no
default** in the tool schema (873/873): asking is necessary, not polite.

**The follow-up probe.** A clarify item does not end at the question. A
scripted user turn supplies exactly what was missing, and the model answers
again:

- *clarify*: the user replies "The `{slot}` is `{value}`." — the value taken
  from the item's own gold call. Scoring then asks (a) did the first turn ask
  for **that** parameter, and (b) did the second turn call the tool **with
  the supplied value**.
- *abstain*: the user replies "I've now added a tool called `{name}`…" and
  the withheld tool's schema is restored to the tool list (recovered from
  the pre-removal tool list each record retains). The model should now call it.

The follow-up is what separates "produced something question-shaped" from
"asked for the right thing and then acted on the answer". Steering methods
never see the follow-up; it is evaluation only. Every item plays both turns
regardless of what turn 1 did.

**Pool sizes** (built and audited; the audit is described below):

| Task | Unfiltered | Audited | dev / test | with follow-up |
|---|---|---|---|---|
| `call` | 1,295 | 1,226 | 607 / 619 | — (nothing was missing) |
| `clarify` | 1,062 | 873 | 436 / 437 | 873 |
| `abstain` | 1,295 | 1,010 | 506 / 504 | 1,010 |

## Where the data comes from

**Items** are When2Call's 3,652 test MCQs (NAACL'25, NVIDIA; the only dataset
with per-item call/clarify/abstain gold), converted to plain decision points:
the MCQ options are discarded, the model answers freely. Abstain follow-ups
need the schema of the tool When2Call deleted; each record retains it in its
own pre-removal tool list (`orig_tools`), from which the follow-up restores
it (the same tool's current BFCL copy has drifted on 518 rows).

**Contrast pairs** — the material steering methods learn directions from —
come from When2Call's 9,000-row preference *training* split (disjoint from
the items): one prompt, a chosen and a rejected response, sides labelled by
frozen text recognisers. 3,865 rows become pairs; 3,378 survive audit
(1,680 should_call / 849 should_clarify / 849 should_abstain). Pairs are
stored model-neutral: calls as bare JSON, rendered into each model's own
format at use. Extraction draws balanced piles of 700 per side.

**The audit.** Both sources are synthetic and noisy (When2Call's own paper
says so). Every item pool and pair pile went through LLM-assisted audits
with human adjudication — 1,141 keep/drop decisions in `data/filters/`,
keyed to stable ids. The decisions are the only non-regenerable artifacts in
the project: re-running the audit would not reproduce them.

**Splits.** dev/test is drawn 50/50 over *lineages*, not items: When2Call
derived call, clarify and abstain variants from the same source scenarios,
so an item-level split would leak scenario text from dev to test through a
sibling. Lineages are drawn whole (seed frozen in `data/splits.py`); the
generated `splits.json` *is* the protocol. Methods tune on dev, report on
test — the test split is untouched so far.

**Provenance.** Every raw input is fetched by `fetch` from a pinned
upstream revision and verified by sha256 — a rebuild on a blank machine
either reproduces the benchmark byte for byte or fails loudly. `build`
asserts every pool count against the numbers above.

## How a run works

`eval/loop.py` is the one conversation loop every arm goes through. Per
batch: render the conversation, generate with the method's intervention
open, hand the reply to the track's `next_message`, splice in the scripted
user turn if there is one, generate again.

- **Transport.** Tools go through the subject model's **own chat template**
  (`apply_chat_template(tools=…)`), so every model is asked at the interface
  it was trained on. The template ships with call-leaning framing; that is
  part of deployed reality, and the prompting arm measures how much of it
  one instruction can undo. Thinking is off; decoding is greedy,
  max 160 new tokens. Subject model so far: **Qwen3-8B**.
- **The method contract** (`methods/`): a method exposes
  `system()` (its system prompt, if any), `hooks(model, dose)` (its forward
  hooks), and `logits(dose)`; at dose 0 generation must be byte-identical to
  no method at all. One
  method at one dose is **one fixed, task-blind policy**: it is applied to
  every item of every task, every turn — nobody tells the arm which task an
  input belongs to. That is the deployment reality, and it is where
  unconditional steering is expected to pay for its target-task gains.
  A `--steer-turns first` switch confines the intervention to the model's
  first reply, for the timing ablation the findings below motivate.
- **Records carry raw text only.** What the model said is generated once;
  deciding what it meant happens offline on the stored record — a judge or
  parser change costs a re-read, never a GPU re-run.

## How answers are scored

Two organs, with a fixed division of labour:

- **Structural** (`eval/calls.py`): text that parses as a tool call is
  checked mechanically, following BFCL's AST evaluation — name offered → no
  hallucinated parameters → required parameters present — plus
  `name_match` against gold. The checker is frozen under a regression gate
  of 352 real generations with checked labels.
- **Semantic** (`eval/judge.py`): everything else is read by a pinned
  Gemma-3-27B judge with a certified rubric (.941 agreement against 204
  human-labelled responses): `tool_call` / `clarify` / `abstain` /
  `direct_or_other`. A second rubric, `asks_for`, decides whether a
  clarifying question asks for **the withheld parameter** (not yet
  certified; a hand-checked sample is scheduled before test).
- **Follow-up**: `used_value` checks the second-turn call actually uses the
  value the user supplied; abstain recovery checks the restored tool is
  called.

**Per task, two headline numbers.** `decision` — turn 1 was right (for
clarify that includes asking for *the* withheld slot). `followup` — turn 1
was right **and** turn 2 acted on the answer. A wrong first turn fails
followup no matter how good turn 2 is; recovering after a wrong decision
(`recovery`) and deciding right then failing to act (`sticky`) are kept as
diagnostics. Dose–response curves are reported rather than a single point;
`call` doubles as the guard axis. The declared operating-point rule:
maximise target-task followup subject to call decision ≥ baseline − .10,
ties to the smaller dose. For the one-shot test run the protocol adds
Wilson 95% CIs and McNemar against baseline on identical items.

## Methods under comparison

Two reference arms, always present: **baseline** (no intervention) and
**prompting** (a behaviour-matched system instruction per axis — the
cheapest thing that could work, and the bar any representation method must
beat).

The steering roster — every named method implemented from its original
paper *and* official code, never a summary; where our setting forced an
adaptation, the table says so and the paper must name it:

| Method | Direction / signal | Applied | Fidelity notes |
|---|---|---|---|
| dim | mean difference of situation piles, last prompt token | all turns, all positions | last-token task adaptation of difference-in-means (Arditi et al. style) |
| caa | mean difference of paired responses, first response token | all turns, last position of each forward (final prompt token + generated tokens) | Rimsky et al.: read at the token where the reply commits (their answer letter), inject from the end of the instruction onward, per the official code; contrast is full replies rather than A/B letters |
| probe | logistic-regression weights on situations | all turns, all positions | discriminative counterpart to dim; same piles, same injection |
| cast | pairwise-PCA condition + behavior vectors; condition gates the push | gated; behavior band of layers | IBM official-code port; condition threshold calibrated on train pairs (F1 .67–.70) |
| iti | per-head probes (2-fold), top-K heads, mass-mean shift, σ-scaled | every step, last position | Li et al. official recipe; K=48 |
| sadi | amplify the input's own top-differing heads | once, at the last prompt token | official head-variant; dose is the paper's δ (×(1+δ)) |
| sae | top-separating feature of Qwen's official Qwen3-8B SAEs (TopK-100, 64K features) | all turns, all positions | decoder column unit-normalised; top features separate the piles cleanly on the instruct model (firing 51–72% vs <3%) |
| (asa) | router + probe-gated signed vectors — nearest in-domain prior work | — | deferred: no official code released; treated as related work |

Every axis has exactly one artifact per method ("toward calling" is
axis-relative — a model can fail to call by asking, by declining, or by
answering from memory; the first two have audited pairs, the third is future
work — the pooled call direction is the shared component of the two, verified
cos .9993 against their average). Fairness is equal dev budget, identical
piles, and one shared injection implementation for all residual-stream arms.

## The multi-turn track

The decision track above is single-decision with a scripted probe. The
second track asks whether an intervention **survives inside a real executing
agent**: BFCL v4 multi-turn, where the model's calls are actually executed
against stateful backends and the result feeds its next step, up to 20 steps
per user turn. Entries stay in BFCL's own format; only the backends, the
executor and the state checker are borrowed (bfcl-eval, pinned to the data
snapshot date). The conversation runs through **our** loop — native FC
transport, identical steering hooks — which BFCL's own local-model path
cannot do (it forces multi-turn onto prompting transport).

Its three matched categories map onto our axes — `base` → call,
`miss_param` → clarify, `miss_func` → abstain — split dev 50 / test 150 by
*index*, since the categories are index-matched variants of one scenario.
Scoring keeps two ledgers: BFCL's official state check (task success), and
per-turn diagnostics on the *blocked* turns that the official score skips —
premature calls and asked questions, which is exactly where steering should
show. Operating points come from the single-turn dev grid; nothing is tuned
on this track.

## Findings so far (dev, 100 items/task, preliminary)

Baseline Qwen3-8B: call .93/.93, clarify .20/.14, abstain .53/.51
(decision/followup) — the model over-calls; the decision is nonetheless
linearly decodable from layer ~6 on (probe accuracy ≈ 1.0), a
representation–behaviour gap.

1. **Clarify steering is real.** dim@L24 lifts clarify followup .14 → .42
   (dose 48) → .61 (dose 72); the rule-compliant operating point is dose 24
   (+.13 followup at −.03 call).
2. **Abstain steering is an illusion that the followup metric exposes.**
   Decision reaches 1.00 while followup falls to .00; across 18
   configurations none beats the .51 baseline followup. The push produces
   indiscriminate refusal, not judgment — at high dose the model refuses
   call and clarify items too. Single-turn evaluation alone would report
   this axis as a solved success.
3. **The same trade-off transfers to real execution.** On BFCL miss_param,
   caa@L24 cuts blocked-turn premature calls .76 → .24 and raises asking
   .22 → .45, but no configuration improves task pass-rate — whole-
   trajectory steering wins the blocked turn and loses the acting turns.
   This motivates the intervention-timing ablation (`--steer-turns first`),
   not yet run.
4. **Layer selection is robust.** L24 (of 36) is the only usable layer for
   the restraint axes across methods and both tracks; shallow layers
   collapse the model at moderate doses, deep layers are inert.

Caveats: dev subsample, ±~.10 CIs; the remaining methods' grids are running.

## Command line

```
python -m tool_use_steering fetch      # upstream snapshots -> raw/, sha256-gated
python -m tool_use_steering build      # raw/ -> audited pools, counts asserted
python -m tool_use_steering splits     # dev/test tables (explicit act; voids dev results)
python -m tool_use_steering extract --method dim --axis clarify        # artifacts
python -m tool_use_steering eval --method dim --axis clarify \
    --vector <artifact.pt> --layer 24 --dose 0,24,48 --split dev       # single-turn
python -m tool_use_steering eval --track multi_turn --method none --dose 0 --split dev
python -m tool_use_steering sweep --grid <data>/runs/sweep100/grid.json --phase generate --shard 0/4
python -m tool_use_steering sweep --grid <data>/runs/sweep100/grid.json --phase judge
```

`eval` loads the subject model once across all doses, then judges everything
in one judge pass; every dose lands in `runs/<tag>/<split>/dose<K>/` with
`raw.jsonl` (model text, never edited), `decisions.jsonl`, `summary.json`
and a `meta.json` pinning argv, code commit and config. `--n`, `--tasks`,
`--steer-turns`, `--categories` narrow a run; `--n 5 --dose 0` is a
two-minute smoke test.

`sweep` is `eval` over a grid file: the sweep's name, split, item cap, batch
size and every configuration (name, method, config, doses) live in
`runs/<sweep>/grid.json`, next to the results it produces, and each
configuration lands in `runs/<sweep>/<name>/` with the same files and meta
as an `eval` run. Generation shards over GPUs (`--shard i/n`, one process
each); judging runs once. Both phases skip what is already on disk, so the
grid file is the record of the sweep and a configuration added to it is the
only thing the next run evaluates.

## Layout

```
tool_use_steering/
  schema.py      the data contract: Item, Pair, actions, tasks
  paths.py       where data lives (TOOLSTEER_DATA overrides)
  data/          fetch.py (pinned raw), build.py (asserted pools),
                 splits.py (lineage/index dev-test), sources/ (converters)
  eval/          loop.py (the one conversation loop), prompt.py (rendering),
                 calls.py (structural checker), judge.py (semantic judge)
  methods/       one folder or file per method + shared/ mechanics
  tracks/        single_turn.py (decision + follow-up), multi_turn.py (BFCL)
```

Code lives in git; data, vectors and results do not. The data root is
`/common/users/bg654/steering/tool_use_steering/` (override with
`TOOLSTEER_DATA`):

```
raw/        upstream snapshots, hash-verified, never modified
data/       built pools, pairs, filters, splits
artifacts/  what methods precompute, one folder per method
runs/       one directory per evaluation run, raw text records; a sweep's
            grid.json sits in its own directory
```

To rebuild from nothing: `fetch`, then `build`, then `splits`. Fetch fails
loudly on any hash mismatch; build fails on any count drift; regenerating
splits is deliberately a separate, explicit act because it voids every dev
result produced under the old split.
