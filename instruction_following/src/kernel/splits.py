"""Load published carrier splits and derive task-specific prompt views.

On disk each split is one list of carrier prompts:

    splits/<split>/prompts.jsonl   {"prompt_id", "source", "text"} x N

Task-specific extraction pairs and prompted evaluation inputs are derived at
load time so instruction wording remains canonical in tasks.py.

Peek at what a task's data looks like:
    python src/kernel/splits.py --show no_comma
"""
import json

# Run directly as a script as well as imported: the package root
# is this file's ancestor, not its own directory.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernel.config import SPLITS
from kernel.tasks import TASKS, make_pair, make_prompted_prompt  # noqa: E402

EVAL_SPLITS = ("dev", "test")


def _load_prompts(split: str) -> list[dict]:
    """Load carrier prompts while rejecting duplicate join keys."""
    path = SPLITS / split / "prompts.jsonl"
    rows = []
    seen_prompt_ids = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            prompt_id = row["prompt_id"]
            if prompt_id in seen_prompt_ids:
                raise ValueError(
                    f"Duplicate prompt_id {prompt_id!r} in {path} "
                    f"at line {line_number}"
                )
            seen_prompt_ids.add(prompt_id)
            rows.append(row)
    return rows


def extraction_pairs(task: str) -> list[dict]:
    """Positive/negative pairs for vector extraction.

    Positive and negative differ by exactly the appended instruction, which is
    what makes the activation difference attributable to the instruction
    alone."""
    out = []
    for r in _load_prompts("extraction"):
        pos, neg = make_pair(r["text"], task)
        out.append({"task": task, "pair_id": f"{task}_ext_{r['prompt_id']}",
                    "prompt_id": r["prompt_id"], "base_prompt": neg,
                    "positive": pos, "negative": neg})
    return out


# The dev split's job is to rank nine configurations, not to estimate an
# accuracy — the published number comes from test. Ranking is much the easier
# task: the gaps between configurations run to tens of points, while 150 items
# put a 95% interval at roughly +/-8. Scoring all 405 spent three times the
# generation for a decision the smaller sample already makes.
CALIBRATION_ITEMS = 150
CALIBRATION_SEED = 20260801


def calibration_items(task: str) -> list[dict]:
    """The fixed dev subset every calibration searches over.

    Drawn once with a fixed seed and kept in file order, so it is the same
    prompts for every task, every baseline and every re-run — a subset that
    moved between baselines would make their dev numbers incomparable.
    """
    import random

    items = eval_items(task, "dev")
    if len(items) <= CALIBRATION_ITEMS:
        return items
    chosen = random.Random(CALIBRATION_SEED).sample(
        range(len(items)), CALIBRATION_ITEMS
    )
    return [items[index] for index in sorted(chosen)]


def eval_items(task: str, split: str) -> list[dict]:
    """Build task-specific items for a dev or test evaluation."""
    if split not in EVAL_SPLITS:
        raise ValueError(
            f"Evaluation split must be one of {EVAL_SPLITS}, got {split!r}"
        )
    spec = TASKS[task]
    # item_id is derived from prompt_id, never from position in the file.
    # A positional id silently remaps every record if the split file is ever
    # reordered or rebuilt, and these ids are the join key across conditions,
    # configs and judge verdicts — a remap would not error, it would quietly
    # pair one prompt's unsteered output with another's steered output.
    out = []
    for r in _load_prompts(split):
        base_prompt = r["text"].strip()
        out.append({
            "task": task,
            "item_id": f"{task}_{split}_{r['prompt_id']}",
            "prompt_id": r["prompt_id"],
            "base_prompt": base_prompt,
            "prompted_prompt": make_prompted_prompt(base_prompt, task),
            "verifier": spec["verifier"],
        })
    return out


def _show(task: str, n: int = 2):
    print(f"task={task}  canonical instruction: {TASKS[task]['instruction']}\n")
    for split in ("extraction", "dev", "test"):
        rows = _load_prompts(split)
        print(f"-- {split}: {len(rows)} prompts")
    print("\n-- extraction pairs (sample)")
    for p in extraction_pairs(task)[:n]:
        print(f"    +  {p['positive'][:100]}")
        print(f"    -  {p['negative'][:100]}")
    print("\n-- test items (sample)")
    for it in eval_items(task, "test")[:n]:
        print(f"    unsteered: {it['base_prompt'][:100]}")
        print(f"    prompted : {it['prompted_prompt'][:100]}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", default="no_comma", choices=list(TASKS))
    _show(ap.parse_args().show)
