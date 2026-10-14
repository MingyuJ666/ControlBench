# ControlBench

ControlBench evaluates activation steering across five domains, comparing methods
including CAA, PCA, LAT, linear probing, ITI, and SAE with baseline generation.

## Domains

Each domain guide covers setup, data, experiment commands, and metrics.

| Domain | CLI ID | Task |
|---|---|---|
| [Instruction following](instruction-following/README.md) | `instruction_following` | Follow five response-format constraints |
| [Tool use](tool-use/README.md) | `tool_use`, `tool_use_multi_turn` | Decide when to call, clarify, or abstain; execute multi-turn tool conversations |
| [Reasoning](reasoning/README.md) | `reasoning` | Solve MATH500, AMC2023, and AIME2024 problems |
| [Safety](safety/README.md) | `safety` | Classify candidate actions as safe or unsafe |
| [Long-to-short](long-to-short/README.md) | `long_to_short` | Shorten mathematical reasoning while preserving accuracy |

## Usage

Set up the selected domain's environment, then run from the repository root:

```bash
python bench.py --list
python bench.py --domain reasoning -- --help
```

All domains use the same launcher:

```text
python bench.py --domain DOMAIN [--entry ENTRY] [--python EXECUTABLE] -- DOMAIN_ARGUMENTS
```

- `--entry` selects a workflow; `--list` shows the choices, with the default first.
- `--python` selects the domain's interpreter; it defaults to the active Python.
- `--preview` prints the launch command without running it.
- Arguments after `--` go to the domain's CLI.

For example, preview an IF campaign in its installed environment:

```bash
python bench.py --domain instruction_following -- \
  --model Qwen/Qwen3-8B --methods unsteered,caa \
  --layers 9 --strengths 0.5,1.0 --sample 16 \
  --gpus 0 --out outputs/if --dry-run
```

Remove `--dry-run` to execute. Full experiment commands are in the domain guides above.

## Outputs

Pass the output option after `--`:

| Domain | Output option |
|---|---|
| Instruction following | `--out DIR` |
| Tool use (both tracks) | `--out DIR` |
| Reasoning | `--output DIR` |
| Safety | `--out DIR` (required) |
| Long-to-short inference | `--root DIR` |

Use separate directories for each domain or track, and the same directory when
resuming a run. Relative output paths use the repository root, except for
long-to-short, which runs from `long-to-short/`. The domain guides list defaults
and output contents.
