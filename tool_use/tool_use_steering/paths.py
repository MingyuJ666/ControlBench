"""Where things live.

The benchmark's inputs ship in the repo and are read from here: the committed,
directly-read data under this domain's raw/ and data/, resolved relative to
this file so a clone runs with no path set. What a run PRODUCES -- the vectors
and probes methods precompute, and the run records -- stays off git on a data
disk, pointed anywhere with TOOLSTEER_DATA.

    raw/       upstream snapshots the eval reads as-is (the multi-turn data)
    data/      the split, audited items, the split tables, the contrast pairs
    artifacts/ what methods precompute (vectors, probes, feature picks),
               one subdirectory per method
    runs/      one directory per evaluation run
"""

import os

#: Committed inputs live in the repo, one level up from this package.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(_ROOT, "raw")
DATA = os.path.join(_ROOT, "data")

ITEMS = os.path.join(DATA, "items")
PAIRS = os.path.join(DATA, "pairs.jsonl")  # one file: the sides are named by action
SPLITS = os.path.join(DATA, "splits.json")
#: Build intermediates the eval never reads; only the (separate) data pipeline
#: writes them, so they are absent from this release.
FILTERS = os.path.join(DATA, "filters")
UNFILTERED = os.path.join(DATA, "unfiltered")

#: What a method computes ahead of evaluation and needs at run time -- a
#: steering vector, trained low-rank weights, a feature selection -- and the
#: run records: produced, large, off git. One subdirectory per method.
OUT_ROOT = os.environ.get(
    "TOOLSTEER_DATA", "/common/users/bg654/steering/tool_use_steering")
ARTIFACTS = os.path.join(OUT_ROOT, "artifacts")
RUNS = os.path.join(OUT_ROOT, "runs")
