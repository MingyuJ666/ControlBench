"""The extraction split: contrastive pairs for learning directions.

inputs/extraction.jsonl holds 8,100 pairs, 2,700 per target. A row is one user
request with a tool list and two candidate responses that take different
actions:

    pair_id            1-based line number in NVIDIA's when2call_train_pref.jsonl
    target             what the positive response does: call, clarify or abstain
    tools, messages    the tool list and the conversation up to the request
    positive           the response the situation calls for
    negative           the other response, which does something else
    negative_action    call, clarify, abstain or direct

Both sides of a pair share a prompt, so what separates them is the response.
Each target draws 900 pairs against each of the three other actions, so its
negative pile is exactly one third each and no counterpart can dominate the
direction by weight.

Both tracks learn from this file. The multi-turn track reuses the directions
the single-turn data produced rather than learning its own, so the pairs are
shared and the evaluation items are not.

The file is frozen and its sha256, provenance and construction are recorded in
inputs/MANIFEST.json; the script that drew it is not part of this package.
Tool calls are stored as a bare JSON list of {name, arguments}, so the file is
model-neutral; the subject model's own call format is applied when a pair is
rendered, which is render.py's job.

    python -m control_bench.domains.tool_use.pairs      # counts, as a check
"""

import json
from collections import Counter

from .schema import ACTIONS, INPUTS, TARGETS

EXTRACTION = INPUTS / "extraction.jsonl"


def load_extraction() -> list[dict]:
    """Every pair, with the file's invariants checked as it is read."""
    rows, seen = [], set()
    with EXTRACTION.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            row = json.loads(line)
            pair_id = row["pair_id"]
            if pair_id in seen:
                raise ValueError(f"duplicate pair_id {pair_id} in {EXTRACTION} at line {number}")
            seen.add(pair_id)
            if row["target"] not in TARGETS:
                raise ValueError(f"pair {pair_id}: unknown target {row['target']!r}")
            if row["negative_action"] not in ACTIONS:
                raise ValueError(f"pair {pair_id}: unknown negative action "
                                 f"{row['negative_action']!r}")
            if row["negative_action"] == row["target"]:
                raise ValueError(f"pair {pair_id}: both sides are {row['target']!r}; "
                                 "a pair needs a contrast")
            rows.append(row)
    return rows


def extraction_pairs(target: str) -> list[dict]:
    """The pairs of one target, in pair_id order."""
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    return [row for row in load_extraction() if row["target"] == target]


def main() -> None:
    print(f"{len(load_extraction())} pairs <- {EXTRACTION}")
    for target in TARGETS:
        pairs = extraction_pairs(target)
        counts = Counter(row["negative_action"] for row in pairs)
        shares = "  ".join(f"{a} {counts[a]}" for a in ACTIONS if counts[a])
        distinct = len({row["messages"][-1]["content"] for row in pairs})
        print(f"  {target:8s} {len(pairs):5d} pairs   negatives: {shares}"
              f"   distinct requests: {distinct}")


if __name__ == "__main__":
    main()
