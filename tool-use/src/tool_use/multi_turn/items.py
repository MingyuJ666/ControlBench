"""The multi-turn evaluation set.

inputs/multi_turn.jsonl holds 600 conversations, 200 per target, from BFCL v4's
multi_turn categories. A row is one conversation:

    item_id       BFCL's entry id, e.g. multi_turn_miss_func_0
    target        call, clarify or abstain
    category      BFCL's own name: base, miss_param, miss_func
    tools         the tool list the conversation starts with
    injections    turn index -> the function documentation restored at it
    answer        BFCL's per-turn reference calls; an empty list means the turn
                  calls for no tool at all, which is where clarify and abstain
                  behaviour shows
    entry         the BFCL entry consumed by the official checker; any local
                  correction is recorded in inputs/multi_turn_corrections.json

The file is frozen and its sha256 and provenance are in inputs/MANIFEST.json.

    python -m core.domains.tool_use.multi_turn.items
"""

import json

from ..schema import INPUTS, TARGETS

MULTI_TURN = INPUTS / "multi_turn.jsonl"


def load_items() -> list[dict]:
    """Every conversation, with the file's invariants checked as it is read."""
    rows, seen = [], set()
    with MULTI_TURN.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            row = json.loads(line)
            item_id = row["item_id"]
            if item_id in seen:
                raise ValueError(f"duplicate item_id {item_id!r} in {MULTI_TURN} at line {number}")
            seen.add(item_id)
            if row["target"] not in TARGETS:
                raise ValueError(f"{item_id}: unknown target {row['target']!r}")
            if len(row["answer"]) != len(row["entry"]["question"]):
                raise ValueError(f"{item_id}: {len(row['answer'])} reference turns for "
                                 f"{len(row['entry']['question'])} user turns")
            rows.append(row)
    return rows


def eval_items(target: str) -> list[dict]:
    """The conversations of one target, in file order."""
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    return [row for row in load_items() if row["target"] == target]


def main() -> None:
    rows = load_items()
    print(f"{len(rows)} conversations <- {MULTI_TURN}")
    for target in TARGETS:
        items = eval_items(target)
        turns = [len(item["entry"]["question"]) for item in items]
        blocked = sum(1 for item in items for gt in item["answer"] if not gt)
        injections = sum(len(item["injections"]) for item in items)
        print(f"  {target:8s} {len(items):4d} conversations   turns {min(turns)}-{max(turns)}"
              f"   blocked turns {blocked}   restored functions {injections}")


if __name__ == "__main__":
    main()
