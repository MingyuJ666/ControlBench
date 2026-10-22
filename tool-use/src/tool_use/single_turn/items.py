"""The single-turn evaluation set.

inputs/eval.jsonl holds 3,109 items, with 1,549 original dev items and 1,560
original test items. Full evaluation remains the default; selecting dev/test
uses the recorded lineage-level partition without changing file order.

    item_id            When2Call's uuid, prefixed w2c_
    target             the action the situation calls for
    split              dev or test in the original lineage-level 50/50 division
    lineage            the upstream scenario; the division was drawn over these
    tools, messages    the tool list and the conversation, ending on the user
    gold               tool_names, and where one could be built, followup

`gold.followup` scripts a second turn: for a clarify item the user supplies the
withheld argument, for an abstain item the missing tool comes back. Call items
have none -- they are complete at the decision. 1,883 of the 3,109 items carry
one (873 clarify, 1,010 abstain), played by the domain's next_turn hook.

    python -m core.domains.tool_use.single_turn.items
"""

import json

from ..schema import EVAL_SPLIT, EVAL_SPLITS, INPUTS, TARGETS

EVAL = INPUTS / "eval.jsonl"


def load_items() -> list[dict]:
    """Every evaluation item, with the file's invariants checked as it is read."""
    rows, seen = [], set()
    with EVAL.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            row = json.loads(line)
            item_id = row["item_id"]
            if item_id in seen:
                raise ValueError(f"duplicate item_id {item_id!r} in {EVAL} at line {number}")
            seen.add(item_id)
            if row["target"] not in TARGETS:
                raise ValueError(f"{item_id}: unknown target {row['target']!r}")
            if not row["messages"] or row["messages"][-1]["role"] != "user":
                raise ValueError(f"{item_id}: an item must end on a user message")
            followup = row["gold"].get("followup")
            if followup is not None and (followup.get("slot") is None) == (
                    followup.get("restored_tool") is None):
                raise ValueError(f"{item_id}: a follow-up needs slot/value or "
                                 "restored_tool, not both and not neither")
            rows.append(row)
    return rows


def eval_items(target: str, split: str = EVAL_SPLIT) -> list[dict]:
    """The selected evaluation items of one target, in original file order."""
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    if split not in EVAL_SPLITS:
        raise ValueError(f"choose an evaluation split from {EVAL_SPLITS}, got {split!r}")
    rows = load_items()
    if split == "full":
        return [row for row in rows if row["target"] == target]
    lineage_splits = {}
    for row in rows:
        if row.get("split") not in ("dev", "test"):
            raise ValueError(f"{row['item_id']}: split must be dev or test")
        if lineage_splits.setdefault(row["lineage"], row["split"]) != row["split"]:
            raise ValueError(f"{row['item_id']}: lineage appears in both dev and test")
    return [row for row in rows if row["target"] == target and row["split"] == split]


def main() -> None:
    print(f"{len(load_items())} items <- {EVAL}")
    for target in TARGETS:
        items = eval_items(target)
        followups = sum(1 for item in items if item["gold"].get("followup"))
        print(f"  {target:8s} {len(items):5d} items   with follow-up {followups}")


if __name__ == "__main__":
    main()
