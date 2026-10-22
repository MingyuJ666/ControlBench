"""Tool-use decision steering as a control_benchmark domain.

Given a user request and a tool list, an agent must call the right tool, ask
for the argument it is missing, or decline when no tool fits. Those three
actions are the targets. A steering method has to move the decision without
being told which decision to make. The prompted baseline instead supplies a
conditional instruction for the target action throughout the conversation.

What the shared layer takes from a domain, and where it is here:

  pairs(target)        2,700 contrastive pairs: one prompt, two responses that
                       take different actions
  rendering()          chat template, the assistant turn included, thinking off
  render(tok, item)    the prompt for one item, ending where the model writes
  prompted_items(...) generation-only copies with a target instruction
  items(target)        the whole evaluation set, 3,109 items over three targets
  clean(text)          drop a stray think block
  next_turn(...)       the scripted second turn, or None when an item is done
  prepare_judging,     the blind judge: what it is asked, and whether it has
  judge, metrics       answered (single_turn/scoring.py, judge_core.py)

Tools reach every model as text, in BFCL's prompting convention, so a result
compares across models rather than across template conventions; render.py says
why. The chat template still supplies each model's own turn markers.

Scoring judges every response once for action, coherence and appropriateness;
submitted calls also undergo deterministic argument validation. Clarify and
abstain items have a separately judged scripted second response. Call items
have no follow-up.
"""

import re
from pathlib import Path

from repsteer.data import ContrastivePairs

from ...paths import PACKAGE_ROOT, run_dir
from ...runtime.rendering import Rendering
from ...runtime.search import read_jsonl, write_json, write_jsonl
from .pairs import extraction_pairs
from .prompting import prompted_items                                      # noqa: F401  domain hook
from .render import PROMPT_VERSION, item_messages, pair_messages, render_chat
from .schema import EVAL_SPLIT, EVAL_SPLITS, INPUTS, NAME, TARGETS
from .single_turn import (build_records, eval_items, judging_complete as _complete,
                          load_verdicts, merge, next_turn, prepare_shards, summarize)

THINK_BLOCK = re.compile(r"<think>.*?(?:</think>\s*|$)", re.S)

#: Bumped when a scored number could change without the generations changing.
SCORING_VERSION = 6
PROTOCOL_VERSION = PROMPT_VERSION
JUDGE_BATCHES = 4
JUDGE_SEED = 11


def pairs(target: str) -> ContrastivePairs:
    """Positive and negative conversations, each ending on a completed
    assistant turn, so the read point is the token after the reply."""
    rows = extraction_pairs(target)
    sides = [pair_messages(row) for row in rows]
    return ContrastivePairs.from_pairs(
        [positive for positive, _ in sides], [negative for _, negative in sides],
        metadata={"domain": NAME, "target": target, "pairs": len(rows)})


def rendering() -> Rendering:
    """The assistant turn is part of the input, so no generation prompt."""
    return Rendering.chat(add_generation_prompt=False, enable_thinking=False)


def render(tokenizer, item: dict) -> str:
    return render_chat(tokenizer, item_messages(item))


def items(target: str, split: str = EVAL_SPLIT) -> list[dict]:
    """Evaluation items of this target in the full pool or its fixed dev/test split."""
    return eval_items(target, split)


def clean(text: str) -> str:
    """Drop a complete or unfinished think block if the model emitted one."""
    return THINK_BLOCK.sub("", text).strip()


# ---- scoring -------------------------------------------------------------------

def _records(target, split, items_, rows, name, metadata):
    by_id = {row["item_id"]: row for row in rows}
    outputs = []
    for item in items_:
        row = by_id[item["item_id"]]
        output = {key: row[key] for key in ("output", "truncated", "n_new_tokens")}
        if "turns" in row:
            output["turns"] = row["turns"]
        outputs.append(output)
    return build_records(target, split, items_, outputs, name, metadata)


def judging_dir(model: str, experiment: str, method: str, target: str) -> Path:
    return run_dir(NAME, model, experiment, method, target, "judging")


def judging_complete(model, experiment, method, target) -> bool:
    return _complete(judging_dir(model, experiment, method, target))


def prepare_judging(model, experiment, method, target, names, items_, split) -> int:
    records = []
    for name in names:
        rows = read_jsonl(run_dir(NAME, model, experiment, method, target, name) / "generations.jsonl")
        records += _records(target, split, items_, rows, name, {"method": method, "config": name})
    return prepare_shards(records, judging_dir(model, experiment, method, target),
                          JUDGE_BATCHES, JUDGE_SEED)


JUDGE_SCRIPT = PACKAGE_ROOT / "cli" / "judge_vllm.py"


def judge_python() -> str:
    """The interpreter that has vLLM: JUDGE_PYTHON, else <repo>/.venv-judge/bin/python."""
    import os
    import sys
    from ...paths import REPO_ROOT
    configured = os.environ.get("JUDGE_PYTHON")
    if configured:
        return configured
    candidate = REPO_ROOT / ".venv-judge" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def judge(dirs: list, force: bool = False) -> None:
    """Judge several directories in one judge process (the model is loaded once)."""
    import subprocess
    command = ([judge_python(), str(JUDGE_SCRIPT), "--domain", NAME, *map(str, dirs)]
               + (["--force"] if force else []))
    subprocess.run(command, check=True)


def metrics(model, experiment, method, target, name, items_, split, extra=None) -> dict:
    out = run_dir(NAME, model, experiment, method, target, name)
    records = _records(target, split, items_, read_jsonl(out / "generations.jsonl"), name,
                       {"method": method, "config": name})
    merge(records, load_verdicts(judging_dir(model, experiment, method, target)))
    summary = summarize(records)
    write_jsonl(out / "scores.jsonl", records)
    result = {**summary, "n": float(summary["n_total"]), **(extra or {})}
    write_json(out / "summary.json", {**result, "n": summary["n_total"], "config": name,
                                      "method": method, "target": target, "split": split})
    return result
