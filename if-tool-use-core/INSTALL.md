# Instruction following and tool use

This shared runtime serves only `instruction-following/` and `tool-use/`.
The `long-to-short/` project keeps its own installation and execution workflow.
The Python package remains `control_bench`; the three CLI domain IDs remain
`instruction_following`, `tool_use`, and `tool_use_multi_turn`.

Imported from `aslcai/control_benchmark` at
`199d062495e93ea634cf1dc64fe5026f14245a9f`.
The import changes package discovery and filesystem locations. Training,
generation, prompts, parsing, scoring, input files, and protocol versions are
preserved. The original combined README is intentionally omitted. Domain-specific guides
are available for [instruction following](../instruction-following/README.md)
and [tool use](../tool-use/README.md).

## Install from the repository root

Use Python 3.10+ and a CUDA build of PyTorch appropriate for the machine.

```bash
python -m venv .venv-if-tool-use
source .venv-if-tool-use/bin/activate
python -m pip install -r if-tool-use-core/requirements.txt
python -m pip install -e '.[tool-use]'
```

The tool-use extra pins the BFCL backend used by executable multi-turn
evaluation. The judge uses a separate environment:

```bash
python -m venv .venv-judge
.venv-judge/bin/python -m pip install -r if-tool-use-core/requirements-judge.txt
export JUDGE_PYTHON="$PWD/.venv-judge/bin/python"
```

The judge requirements describe the existing A100 setup. A different GPU
architecture may require a compatible judge environment selected through
`JUDGE_PYTHON`. Model access and Hugging Face cache configuration are external
to this repository.

## Data and commands

Shipped inputs and their checksum manifests are in
`instruction-following/data/inputs/` and `tool-use/data/inputs/`.
Both tool-use tracks use the latter directory.

```bash
cbench-selfcheck --domain instruction_following
cbench-selfcheck --domain tool_use
cbench-selfcheck --domain tool_use_multi_turn

cbench-run generate --domain instruction_following --method unsteered --sample 4 --dry-run
cbench-run generate --domain tool_use --method unsteered --sample 4 --dry-run
cbench-run generate --domain tool_use_multi_turn --method unsteered --sample 4 --dry-run
cbench-campaign --help
```

Existing CLI arguments, method names and stage commands are unchanged.
Outputs retain `data/<domain>/outputs/repsteer-native-v2/` relative to this
repository. To reuse an existing data disk, set `STEERING_DATA` to its absolute
data root. With that override, inputs are read from
`$STEERING_DATA/<domain>/inputs/` and outputs use the existing domain layout.
Multi-turn inputs still come from `tool_use/inputs/`, while its outputs remain
under `tool_use_multi_turn/`.

For source execution without installing this package, add the absolute
`if-tool-use-core/` directory to `PYTHONPATH` and use
`python -m control_bench.cli.run`. Keep all three project directories together.
Editable installation is the supported checkout workflow; a wheel alone does
not include the input datasets and needs `STEERING_DATA`.

## Offline regression tests

```bash
python -m pip install -e '.[test]'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
```

The suite uses shipped data, mocked judges, and small CPU models; it does not
download subject-model weights. Generated outputs, environments and logs are
not included in this import.
