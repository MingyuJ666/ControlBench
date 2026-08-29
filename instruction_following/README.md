A benchmark for activation steering on *hidden* instruction following. The
model is asked an ordinary question with no instruction attached, and a
steering vector — not the prompt — has to make it obey a formatting rule it
was never told. Five rule-checkable tasks, one model, several methods under
one protocol and one budget.

    lowercase          write in lowercase only
    fixed_ending       end with the exact sentence "Hope this helps!"
    question_ending    end on a question
    one_sentence       answer in exactly one sentence
    no_comma           use no commas

The questions are first-turn English prompts from OpenAssistant, drawn from
`OpenAssistant/oasst1` at `fdf72ae` and `OpenAssistant/oasst2` at `179dd21`
and screened down to a shared pool of 2028, redistributed here under OASST's
Apache-2.0 licence. They are split once into 1014 for extraction, 405 for
development and 609 for the frozen test; a question is in exactly one split,
and every task uses the same questions, so a number in one column is
comparable to a number in the next. The splits are in `splits/`, one JSONL
each, with `splits_manifest.json` recording the seed, the fractions and a
sha256 per file — the same hashes the pipeline reads them under.

Each split has one job. Extraction pairs a question carrying the instruction
against the same question without it, and a method turns the model's
activations on those pairs into its steering artifact. Development selects:
each method searches nine configurations per task, and every configuration
generates on the same fixed 150-question subset of dev, drawn once at seed
20260801 and identical across methods and tasks. Dev's job is to rank nine
configurations, not to estimate an accuracy — the gaps between
configurations run to tens of points while 150 items put a 95% interval at
roughly ±8. The winner — highest
accuracy, ties to the weaker strength and then the lower site, keeping the
least invasive configuration among equals — is frozen and run exactly once
on the 609 test questions.

Evaluation shows the model the bare question; the instruction is present
only in the steering. One measurement everywhere, called accuracy: the
task's rule passes *and* the judge model, shown the answer under a hash
instead of a method name, finds it coherent and responsive *and* the answer
has not collapsed into repetition; truncation at the generation budget is a
failure. Two reference conditions bound the range without sharing its
measurement: `unsteered` generates from the bare question, `prompted` writes
the instruction into the prompt, and both run on the full split and report
the rule-pass rate alone, with no judge.

So, in the tables below: a steered dev cell is the best of nine
configurations, each measured on the same 150 dev questions; a steered test
cell is that one frozen configuration measured on all 609 test questions;
and the `unsteered` and `prompted` rows are rule-pass rates on the full 405
or 609.

Development results on `Qwen/Qwen3-8B`:

| | lowercase | fixed_ending | question_ending | one_sentence | no_comma |
|---|---|---|---|---|---|
| unsteered | 0.0% | 0.0% | 14.3% | 0.0% | 0.2% |
| caa | 70.7% | 0.0% | 16.0% | **71.3%** | 55.3% |
| probe | 68.7% | 0.0% | 17.3% | 69.3% | **58.0%** |
| lat_pca | 70.7% | 0.0% | 18.0% | 68.7% | 56.0% |
| sae | **74.7%** | 0.0% | 14.0% | 0.7% | 0.7% |
| iti | 9.3% | 0.0% | 17.3% | 23.3% | 24.0% |
| prompted | 97.8% | 100.0% | 100.0% | 99.3% | 94.6% |

The frozen test:

| | lowercase | fixed_ending | question_ending | one_sentence | no_comma |
|---|---|---|---|---|---|
| unsteered | 0.0% | 0.0% | 12.5% | 0.0% | 0.0% |
| caa | 66.3% | 0.0% | 10.7% | 69.5% | 48.6% |
| probe | 68.0% | 0.0% | 12.0% | 69.1% | **51.6%** |
| lat_pca | 66.8% | 0.0% | 12.3% | **70.6%** | 48.9% |
| sae | **78.0%** | 0.0% | 12.5% | 0.2% | 0.2% |
| iti | 8.4% | 0.0% | 15.8% | 24.8% | 24.6% |
| prompted | 98.9% | 100.0% | 99.7% | 98.9% | 94.9% |

Dev chose well: on the three tasks steering moves, test costs each method two
to seven points and reorders nothing, and the SAE's `lowercase` number *rises*
to the highest cell in the benchmark. Both null results hold at n=609.

Five findings so far, each with the measurement behind it.

Steering carries diffuse style but not a literal string. Three tasks move from
essentially zero to 50–78%. `fixed_ending` stays at 0.0% for every method and
every configuration — close to ten thousand generations across dev and test,
and "Hope this helps" never once lands at the end — while the prompted control
reaches 100%. The model can do it; no direction we can find induces it.

`question_ending` is not steered at all. The unsteered model already ends on a
question 14.3% of the time on dev and 12.5% on test, and no method beats that
by more than the interval around it — on test, two methods land below it. A
task needs its unsteered baseline printed beside it, or its number means
nothing.

The three residual-stream methods are one method here. Their directions agree
to a cosine of 0.9999, they select the same physical dose, and a third of their
generations are identical token for token. The first
principal component of the extraction differences explains 80–87% of the
variance, so "an instruction was added" is close to a single direction, and a
mean difference, a probe normal and a principal axis all find it. Whatever
discrimination this benchmark offers has to come from methods that do not
estimate a residual-stream direction.

Localising the intervention to a few heads is a different method, and a
worse one here. `iti` reaches 9.3% where the dense methods reach 70% on
`lowercase`, and its best cell anywhere is 24.0%. Two of its configurations
pass the rule on every single item and score zero: steering 128 heads at 30
sigma collapses the output into repetition, which contains no commas and ends
on no sentence. It is the clearest case in the benchmark for scoring a rule
and a judge together rather than a rule alone.

Sparse selection succeeds or fails on whether the dictionary holds the right
unit. The SAE beats every dense method on `lowercase` at a *smaller* dose, and
scores 0.7% on `one_sentence` and `no_comma` — nine configurations, all near
zero. Its ranking picked the same feature for four of the five tasks at layer
33, a feature that reports that an instruction is present rather than which
one. `sae_disc` and `iti_disc` are registered to test whether a discriminative
ranking recovers what the naive one loses.

What is fixed across every run, so that the numbers compare:

    generation budget    3072 new tokens, greedy, thinking disabled
    search               one joint grid, three sites by three strengths
    budget               9 configurations per baseline per task
    selection            a fixed 150-prompt subset of dev, seed 20260801
    truncation           counts as a failure, on every task
    judge                gemma-3-27b-it, shown a hash instead of the config
    dose                 recorded per configuration, and as a ratio against
                         the residual stream it is added to

The generation budget was measured: the unsteered model's
longest dev answer is 2010 new tokens, p99 1857, and nothing reaches a 4096
ceiling, so 3072 leaves room for a more verbose model without letting broken
steering run much further than it already does. Under it the unsteered control
truncates 0.2% of the split, which is what makes counting truncation as a
failure a statement about the steering rather than about the budget.

Setting up on a new machine. One virtual environment, pinned dependencies:

    python -m venv .venv && .venv/bin/pip install -r requirements.txt

The judge, `google/gemma-3-27b-it`, is a gated model: accept its licence on
Hugging Face once, and authenticate with `hf auth login` (or set HF_TOKEN)
before the first judged stage. Weights download on first use, about 16GB for
the target model and 54GB for the judge; set HF_HOME to an existing cache to
reuse one, otherwise they land in `hf_cache/` next to the output root. A
fresh clone needs no data setup: the splits ship with the repository, and
outputs go to `data/`, created on first use (set IF_DATA_ROOT to put them on
a data disk instead). Before spending GPU time, `python -m pytest tests/ -q`
should pass, and `python run.py status` should print the five tasks with
every stage pending.

Running it. Every stage is a subprocess, so a stage can fail and be re-run
alone, and repeating a command resumes what is already on disk.

    python run.py status                                  where each task stands
    python run.py produce   --baseline caa --task all     extract the artifacts
    python run.py calibrate --baseline caa --gpus 0,1     search dev, then judge
    python run.py test      --baseline caa --gpus 0,1     the frozen test run
    python run.py report                                  the cross-method table

The judge needs about 54GB, so give it two 40GB cards or one 80GB. Generation
needs one. A grid is named for the axis it varies, which is the baseline's own:

    python run.py calibrate --task no_comma --grid layer=9,21,33 --grid alpha=1,2,4
    python run.py calibrate --task no_comma --grid heads_k=16,48,128    # iti

The SAE baselines need their reconstruction gate first. The Qwen-Scope
dictionary was trained on `Qwen3-8B-Base` and is applied here to the
post-trained model, so only layers whose reconstruction passes a
pre-registered threshold may be searched:

    CUDA_VISIBLE_DEVICES=0 python src/baselines/sae/gate_runner.py

On this model it admits layers 21, 27 and 33; the fraction of variance the
dictionary fails to explain runs 0.85 at layer 9 and 0.007 at layer 33.

The controls and the dose denominator are separate one-off measurements:

    python src/kernel/reference_conditions.py --split dev
    python src/kernel/residual_scale.py

Where things are written. The released input is versioned with the code and
the outputs are not, so the repository holds no generations:

    splits/                                         released benchmark input
    <data-root>/                                    ./data, or $IF_DATA_ROOT
    ├── baselines/<model>/<baseline>/               extracted artifacts
    └── runs/
        ├── reference_conditions/<model>/           unsteered and prompted
        ├── calibration/<model>/<baseline>/<task>/  the dev search
        └── test/<model>/<baseline>/<task>/<config>/

Baselines are a registry, and each declares what it needs: an artifact, a
loader, an intervention, a search plan, dose metadata and a producer. Adding
one means writing those six things and registering a key; nothing in the
pipeline knows what a layer or a head count is.

    caa        difference in means, raw magnitude
    probe      logistic-regression normal, sigma units
    lat_pca    first principal component of sign-alternated differences
    iti        per-head mass-mean shift on the heads a probe ranks highest
    sae        one Qwen-Scope feature per admitted layer, activation units
    iti_disc   iti, ranked by specificity to this instruction
    sae_disc   sae, ranked by specificity to this instruction
    reft       specified, not implemented

Tests. Unit and fake-model tests run in about twenty seconds and should pass
before any change; the model smokes are opt-in because they load the 8B model.

    python -m pytest tests/ -q
    RUN_MODEL_SMOKE=1 python -m pytest -q tests/test_model_smokes.py -s

Caveats as of this writing. Every `iti` configuration chose the lowest
strength on offer, which means its grid is probably positioned too high and
its numbers are a lower bound. The dose window is narrow — one step of alpha
takes `lowercase` from 71% to 3% on dev — so a finer grid would likely find
better configurations for every method, and the choices here should be read
as the best of nine rather than as an optimum. `sae_disc` and `iti_disc` have
artifacts but no search yet, so the discriminative question is still open.
