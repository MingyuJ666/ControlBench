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
    """The complete evaluation set shared by all targets."""
    if split != EVAL_SPLIT:
        raise ValueError(f"evaluation split is {EVAL_SPLIT!r}, got {split!r}")
    prompts = load_prompts("eval")
    out = []
    for row in prompts:
        base = row["text"].strip()
        out.append({"task": task, "item_id": f"{task}_full_{row['prompt_id']}", "prompt_id": row["prompt_id"],
                    "base_prompt": base, "prompted_prompt": make_prompted_prompt(base, task),
                    "verifier": TASKS[task]["verifier"]})
    return out
