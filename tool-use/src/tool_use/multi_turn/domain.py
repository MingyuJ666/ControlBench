"""Multi-turn domain hooks: generation, judging, and metric output.

Registered as ``tool_use_multi_turn`` through a compatibility module. The
execution engine and judge rubric live alongside this module. Training pairs
remain shared with the decision track; artifacts and outputs keep their
original, separate ``tool_use_multi_turn`` paths.
"""
import json

import subprocess
from pathlib import Path

from ....paths import PACKAGE_ROOT, run_dir
from ....runtime.search import read_jsonl, write_json, write_jsonl
from .. import (PROTOCOL_VERSION, clean, judge_python, pairs,             # noqa: F401  shared with the single-turn domain
                        rendering)
from . import build_records, eval_items, next_turn, summarize
from .driver import generation_record
from . import judging
from .scoring import merge
from ..prompting import prompted_items                                # noqa: F401  domain hook
from ..render import conversation, render_chat
from ..schema import EVAL_SPLIT, INPUTS, TARGETS                    # noqa: F401

NAME = "tool_use_multi_turn"

#: Bumped when a scored number could change without the generations changing.
SCORING_VERSION = 4

#: Parsing determines which tools execute and therefore subsequent model inputs.
GENERATION_VERSION = 2


def render(tokenizer, item: dict) -> str:
    """The conversation's opening turn, ending where the model writes."""
    return render_chat(tokenizer, conversation(
        item["entry"]["question"][0], item["tools"],
        instruction=item.get("generation_instruction")))


def items(target: str, split: str = EVAL_SPLIT) -> list[dict]:
    """Every conversation of this target. `split` is accepted and ignored:
    each configuration is evaluated on all of them and reported on its own."""
    return eval_items(target)


# ---- scoring ----

def judging_dir(model: str, experiment: str, method: str, target: str):
    return run_dir(NAME, model, experiment, method, target, "judging")


def _outputs(model, experiment, method, target, name, items_):
    rows = read_jsonl(run_dir(NAME, model, experiment, method, target, name) / "generations.jsonl")
    by_id = {row["item_id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != {item["item_id"] for item in items_}:
        raise ValueError("multi-turn generation coverage differs from requested items")
    return [by_id[item["item_id"]] for item in items_]


def prepare_judging(model, experiment, method, target, names, items_, split) -> int:
    rows = []
    for name in names:
        outputs = _outputs(model, experiment, method, target, name, items_)
        rows.extend(judging.judge_rows(target, split, items_, outputs, name))
    return judging.prepare_shards(rows, judging_dir(model, experiment, method, target),
                                  mechanical=(target == "call"))


def judging_complete(model, experiment, method, target) -> bool:
    return judging.judging_complete(judging_dir(model, experiment, method, target))


JUDGE_SCRIPT = PACKAGE_ROOT / "cli" / "judge_vllm.py"


def judge(dirs: list, force: bool = False) -> None:
    pending = [p for p in dirs if not judging.preparation(p)["mechanical_only"]]
    if pending:
        command = ([judge_python(), str(JUDGE_SCRIPT), "--domain", NAME, *map(str, pending)]
                   + (["--force"] if force else []))
        # Keep the original full prompt and completed verdicts. Retry only missing
        # records with a supported longer context; never trim or discard inputs.
        def overflow_capacity():
            needed = []
            for directory in pending:
                for path in directory.glob("judge_unparsed_*.jsonl"):
                    for line in path.read_text().splitlines():
                        row = json.loads(line)
                        if row.get("error") == "context_length_exceeded":
                            needed.append(row["required_model_len"])
            return max(needed, default=0)
        required = overflow_capacity()
        if not required:
            result = subprocess.run(command)
            if result.returncode == 0: return
            required = overflow_capacity()
            if not required: result.check_returncode()
        if required > 131072:
            raise RuntimeError(f"Judge input needs {required} tokens, beyond supported 131072; no truncation applied")
        capacity = max(40960, ((required + 4095) // 4096) * 4096)
        retry = [x for x in command if x != "--force"]
        retry += ["--long-context", "--max-model-len", str(capacity)]
        subprocess.run(retry, check=True)


def metrics(model, experiment, method, target, name, items_, split, extra=None) -> dict:
    out = run_dir(NAME, model, experiment, method, target, name)
    outputs = _outputs(model, experiment, method, target, name, items_)
    records = build_records(target, split, items_, outputs, name,
                            {"method": method, "config": name})
    merge(records, judging.load_verdicts(judging_dir(model, experiment, method, target)))
    summary = summarize(records)
    write_jsonl(out / "scores.jsonl", records)
    result = {**summary, "n": float(summary["n_total"]), **(extra or {})}
    write_json(out / "summary.json", {**result, "n": summary["n_total"], "config": name,
                                      "method": method, "target": target, "split": split})
    return result
