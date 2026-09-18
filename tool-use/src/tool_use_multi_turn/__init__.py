"""Tool-use decision steering, on conversations whose calls are executed.

The same three targets as the tool_use domain, and the same directions: this
domain learns from the same contrastive pairs and only evaluates differently.
It is registered separately because a domain is one (pairs, items, metrics)
triple to the shared layer, and these items are conversations rather than
single decisions. The code lives with its sibling, in
control_bench.domains.tool_use.multi_turn.

    cbench-run generate --domain tool_use_multi_turn --method caa \
        --layers 20 --strengths 1.0

Call/base conversations use BFCL execution and no-call checks. Missing-value
and missing-tool conversations also receive one blind semantic judgment on
their predeclared missing turn, with its complete observed context.

Inputs and outputs sit in different places, on purpose: the conversations are
read from data/tool_use/inputs/multi_turn.jsonl, next to the pairs both
domains learn from, while runs are written under data/tool_use_multi_turn/, so
one evaluation's results never land among the other's.
"""

import subprocess
from pathlib import Path

from ...paths import PACKAGE_ROOT, run_dir
from ...runtime.search import read_jsonl, write_json, write_jsonl
from ..tool_use import (PROTOCOL_VERSION, clean, judge_python, pairs,             # noqa: F401  shared with the single-turn domain
                        rendering)
from ..tool_use.multi_turn import build_records, eval_items, next_turn, summarize
from ..tool_use.multi_turn.driver import generation_record
from ..tool_use.multi_turn import judging
from ..tool_use.multi_turn.scoring import merge
from ..tool_use.prompting import prompted_items                                # noqa: F401  domain hook
from ..tool_use.render import conversation, render_chat
from ..tool_use.schema import EVAL_SPLIT, INPUTS, TARGETS                    # noqa: F401

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
        subprocess.run(command, check=True)


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
