"""Carrier prompts and the task views derived from them.

inputs/prompts/extraction.jsonl and eval.jsonl hold {"prompt_id", "source", "text"} rows;
inputs/MANIFEST.json records sizes, hashes and provenance. Task-specific
pairs and evaluation items are built at load time so the instruction wording
lives in one place (tasks.py).
"""

import json

from ...paths import input_root
from .tasks import TASKS, make_pair, make_prompted_prompt

NAME = "instruction_following"
INPUTS = input_root(NAME)
EVAL_SPLIT = "full"
EVAL_SPLITS = ("full", "dev", "test")
# MANIFEST.json construction: eval concatenates the original 405 dev rows,
# then 609 test rows, without reshuffling.
DEV_SIZE = 405


def load_prompts(split: str) -> list[dict]:
    path = INPUTS / "prompts" / f"{split}.jsonl"
    rows, seen = [], set()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if row["prompt_id"] in seen:
                raise ValueError(f"duplicate prompt_id {row['prompt_id']!r} in {path} at line {number}")
            seen.add(row["prompt_id"])
            rows.append(row)
    return rows


def extraction_pairs(task: str) -> list[dict]:
    """Positive/negative prompt pairs differing by exactly the appended instruction."""
    out = []
    for r in load_prompts("extraction"):
        pos, neg = make_pair(r["text"], task)
        out.append({"task": task, "pair_id": f"{task}_ext_{r['prompt_id']}", "prompt_id": r["prompt_id"],
                    "base_prompt": neg, "positive": pos, "negative": neg})
    return out


def eval_items(task: str, split: str = "full") -> list[dict]:
    """Shared carrier prompts from the full pool or its original dev/test split."""
    if split not in EVAL_SPLITS:
        raise ValueError(f"choose an evaluation split from {EVAL_SPLITS}, got {split!r}")
    prompts = load_prompts("eval")
    if split == "dev":
        prompts = prompts[:DEV_SIZE]
    elif split == "test":
        prompts = prompts[DEV_SIZE:]
    out = []
    for row in prompts:
        base = row["text"].strip()
        out.append({"task": task, "item_id": f"{task}_{split}_{row['prompt_id']}", "prompt_id": row["prompt_id"],
                    "base_prompt": base, "prompted_prompt": make_prompted_prompt(base, task),
                    "verifier": TASKS[task]["verifier"]})
    return out
