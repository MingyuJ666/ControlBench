"""Instruction following as a control_benchmark domain.

Five tasks, one instruction each, over a shared pool of OASST carrier prompts
(data/instruction_following/inputs). A steering method must make the model
obey the instruction without it being in the prompt. The prompted baseline
appends that instruction without an activation intervention.

What the shared layer needs from a domain, all here:
  pairs(target)            1014 chat pairs: prompt with the instruction / without
  rendering()              chat template, generation prompt on, thinking off
  messages(item)           chat messages using the selected generation prompt
  render(tok, item)        the same selected prompt rendered as a string
  prompted_items(items)    generation-only copies with the instruction appended
  items(target, split)     full = 1014; original dev = 405, test = 609
  clean(text)              drop a stray think block
  scoring                  rule + blind judge (Qwen3-8B through vLLM, its own environment)
"""

from pathlib import Path

from repsteer.data import ContrastivePairs

from ...paths import PACKAGE_ROOT, run_dir
from ...runtime.rendering import Rendering
from ...runtime.search import read_jsonl, write_json, write_jsonl
from .data import NAME, EVAL_SPLIT, EVAL_SPLITS, INPUTS, eval_items, extraction_pairs
from .render import render_chat, strip_think_blocks
from .scoring import build_records, load_verdicts, merge, prepare_shards, summarize, judging_complete as _judging_complete
from .tasks import TASKS

TARGETS = tuple(TASKS)
JUDGE_BATCHES = 4
JUDGE_SEED = 11
SCORING_VERSION = 6


def _user(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def pairs(target: str) -> ContrastivePairs:
    rows = extraction_pairs(target)
    return ContrastivePairs.from_pairs([_user(r["positive"]) for r in rows], [_user(r["negative"]) for r in rows],
                                       metadata={"domain": NAME, "target": target, "pairs": len(rows)})


def rendering() -> Rendering:
    return Rendering.chat(add_generation_prompt=True, enable_thinking=False)


def messages(item: dict) -> list[dict]:
    """Only the prompted baseline overrides the plain user message."""
    return _user(item.get("generation_prompt", item["base_prompt"]))


def render(tokenizer, item: dict) -> str:
    return render_chat(tokenizer, item.get("generation_prompt", item["base_prompt"]))


def prompted_items(items_: list[dict]) -> list[dict]:
    """Select instructed generation inputs, retaining original questions for scoring."""
    return [{**item, "generation_prompt": item["prompted_prompt"]} for item in items_]


def items(target: str, split: str = "full") -> list[dict]:
    return eval_items(target, split)


def clean(text: str) -> str:
    return strip_think_blocks(text)


# ---- scoring ------------------------------------------------------------------

def _records(target, split, items_, rows, name, metadata):
    by_id = {r["item_id"]: r for r in rows}
    outputs = [{k: by_id[it["item_id"]][k] for k in ("output", "truncated", "n_new_tokens")} for it in items_]
    return build_records(target, split, items_, outputs, name, metadata)


def judging_dir(model: str, experiment: str, method: str, target: str) -> Path:
    return run_dir(NAME, model, experiment, method, target, "judging")


def judging_complete(model, experiment, method, target) -> bool:
    return _judging_complete(judging_dir(model, experiment, method, target))


def prepare_judging(model, experiment, method, target, names, items_, split) -> int:
    records = []
    for name in names:
        rows = read_jsonl(run_dir(NAME, model, experiment, method, target, name) / "generations.jsonl")
        records += _records(target, split, items_, rows, name, {"method": method, "config": name})
    return prepare_shards(records, judging_dir(model, experiment, method, target), JUDGE_BATCHES, JUDGE_SEED)


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
    # json.dumps escapes every newline in the candidate answer into a literal
    # backslash-n, which breaks the tokenizer's long-run merges: one real answer
    # went from 1,336 tokens to 9,924 and overran the 8,192 default. Measured on
    # 3,751 records, 32,768 costs nothing (1,290 s vs 1,331 s) and leaves every
    # scored verdict field identical.
    cmd = ([judge_python(), str(JUDGE_SCRIPT), "--domain", NAME,
            "--max-model-len", "32768", *map(str, dirs)]
           + (["--force"] if force else []))
    subprocess.run(cmd, check=True)


def metrics(model, experiment, method, target, name, items_, split, extra=None) -> dict:
    out = run_dir(NAME, model, experiment, method, target, name)
    records = _records(target, split, items_, read_jsonl(out / "generations.jsonl"), name,
                       {"method": method, "config": name})
    merge(records, load_verdicts(judging_dir(model, experiment, method, target)))
    summary = summarize(records)
    write_jsonl(out / "scores.jsonl", records)
    result = {"accuracy": summary["accuracy"], "rule_rate": summary["rule_pass_rate"],
              "incoherent_rate": summary["incoherent_rate"], "partial_rate": summary["partial_rate"],
              "none_rate": summary["none_rate"], "truncation_rate": summary["truncation_rate"],
              "n": float(summary["n_total"]), **(extra or {})}
    write_json(out / "summary.json", {**result, "n": summary["n_total"], "config": name, "method": method,
                                      "target": target, "split": split})
    return result
