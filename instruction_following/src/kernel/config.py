"""Paths and models. Defined once, imported everywhere — no module hard-codes
a path.

The released inputs (the splits) live in the repository, beside the code that
reads them. Everything a run PRODUCES lives outside: outputs are large and
rebuildable from the code plus the released splits. Where "outside" is depends
on the machine, so it is resolved in this order:

  1. $IF_DATA_ROOT
  2. <repo>/data          (a directory, or a symlink to wherever the data is)

The default is created on first use, so a fresh clone runs with no setup;
point the variable or a symlink at a data disk when outputs should live
elsewhere. No absolute path from one machine ever reaches the repository.

The HF cache: $HF_HOME if set, otherwise the sibling directory
<data>/../hf_cache, which is where model weights actually live here.
"""
import os
from pathlib import Path

# REPO / DATA_ROOT / HF_HOME are implementation detail: everything else is
# derived from them and downstream modules import the derived paths, not these.
# src/kernel/config.py -> src -> repository root
REPO = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(os.environ.get("IF_DATA_ROOT", REPO / "data")).resolve()
# The HF cache sits BESIDE the data root, not inside it: it is shared with the
# sibling project on this machine and holds ~250GB of weights that have nothing
# to do with this benchmark's data. Defaulting it inside DATA_ROOT pointed at an
# empty directory, and the only symptom would have been silently re-downloading
# a 16GB model.
HF_HOME = os.environ.get("HF_HOME") or str(DATA_ROOT.parent / "hf_cache")
# Applied, not merely computed: without HF_HOME in the environment the
# HuggingFace libraries fall back to ~/.cache and re-download models that are
# already on disk. Importing config is enough, so a script run directly behaves
# the same as one launched through run.py.
os.environ["HF_HOME"] = HF_HOME

RAW = DATA_ROOT / "raw"
# Everything a run writes. Not "interim": the frozen test outputs under
# it are the benchmark's final numbers, not a step on the way to them.
RUNS = DATA_ROOT / "runs"
# The released benchmark input, versioned with the code that reads it: a fresh
# clone runs without data setup, and rebuilding the splits shows up as a git
# diff rather than as silently different inputs.
SPLITS = REPO / "splits"
# Conclusions, not run outputs: the cross-baseline tables a reader of the
# paper would want. Everything under runs/ is evidence for what lands here.
RESULTS = DATA_ROOT / "results"
BASELINES = DATA_ROOT / "baselines"

# The frozen evidence chain from data construction: written once by the
# data_construction/ scripts, read afterwards only by provenance checks.
DATA_CONSTRUCTION = RUNS / "data_construction"

REFERENCE_CONDITIONS = RUNS / "reference_conditions"
CALIBRATION = RUNS / "calibration"
TEST = RUNS / "test"

TARGET_MODEL = os.environ.get("IF_MODEL", "Qwen/Qwen3-8B")
JUDGE_MODEL = os.environ.get("IF_JUDGE", "google/gemma-3-27b-it")


def slug(model: str) -> str:
    return model.split("/")[-1].lower()


def artifact_path(model_slug: str, method: str, task: str) -> Path:
    """Return the artifact path for one model-specific steering direction."""
    return BASELINES / model_slug / method / f"{task}.pt"



def calibration_dir(model_slug: str, method: str, task: str) -> Path:
    """Dev-search outputs. The method segment keeps runs from mixing: two
    methods can never write configurations into one directory."""
    return CALIBRATION / model_slug / method / task


def test_dir(model_slug: str, method: str, task: str, config: str) -> Path:
    """Frozen test outputs for one configuration of one method."""
    return TEST / model_slug / method / task / config


def env() -> dict:
    """Environment for child processes: the HF cache plus whatever GPUs the
    caller selected."""
    e = dict(os.environ)
    e["HF_HOME"] = HF_HOME
    e["IF_DATA_ROOT"] = str(DATA_ROOT)
    return e


# Outputs only. Everything under DATA_ROOT is produced by runs, so a missing
# root on a fresh clone is normal and is created rather than reported.
DATA_ROOT.mkdir(parents=True, exist_ok=True)
