"""The multi-turn track: BFCL v4 multi_turn against real executable backends.

Here the model's calls are executed -- a cd() changes a real directory
object, a trade moves a real balance -- and the results feed back into the
conversation, up to 20 model steps per user turn. Entries stay in BFCL's own
format (they carry initial state, involved backend classes and per-turn
reference calls; our Item cannot hold that and should not try). Only three
things are borrowed from the bfcl-eval package: the backend classes, the
executor, and the state checker. The conversation itself runs through OUR
loop -- native FC format, thinking off, steering hooks identical to the
single-turn track -- which BFCL's own local-model path cannot do (it forces
multi-turn onto prompting transport).

The three categories mirror the single-turn tasks:

    base        call     -- everything needed is on hand
    miss_param  clarify  -- a turn lacks a required value; the next user
                            turn supplies it
    miss_func   abstain  -- a turn lacks the needed tool; it is restored a
                            turn later with a canned announcement

Scoring is two-ledger, on purpose:
  official     bfcl's multi_turn_checker: backend end-state and execution
               results must match the reference. Objective; no judge.
  diagnostics  per-turn behaviour the official score ignores -- BFCL skips
               blocked turns entirely (its irrelevance checker is dead
               code), so premature calls and asked-questions on those turns
               are recorded here. This is where steering should show.
"""

from __future__ import annotations

import json
import os
import re

from ..paths import RAW
from ..eval import calls as calls_mod

MT_RAW = os.path.join(RAW, "bfcl_multi_turn")
CATEGORIES = {"base": "call", "miss_param": "clarify", "miss_func": "abstain"}
STEP_LIMIT = 20  # bfcl MAXIMUM_STEP_LIMIT

#: BFCL's canned line announcing restored functions (miss_func's empty turn).
RESTORED_LINE = "I have updated some more functions you can choose from. What about now?"


class Entry:
    """A BFCL entry wearing just enough interface for eval/loop.py."""

    def __init__(self, raw: dict, answer: list, tools: list[dict],
                 injections: dict[int, list[dict]]):
        self.id = raw["id"]
        # id looks like "multi_turn_miss_func_17"
        self.task = CATEGORIES[
            raw["id"].removeprefix("multi_turn_").rsplit("_", 1)[0]]
        self.raw = raw
        self.answer = answer                     # per-turn reference calls
        self.messages = list(raw["question"][0])  # turn 0
        self.tools = tools                        # turn-0 tool list
        self.injections = injections              # turn idx -> tools to add


def _normalize(node):
    from ..schema import normalize_types
    return normalize_types(node)


def entries(category: str, limit: int | None = None,
            split: str | None = None) -> list[Entry]:
    """BFCL entries of one category; `split` filters by the index-grouped
    dev/test table (matched variants share an index, so the i-th entry of
    every category sits on the same side)."""
    from bfcl_eval.constants.executable_backend_config import (
        MULTI_TURN_FUNC_DOC_FILE_MAPPING)
    docs = {}
    for cls, fname in MULTI_TURN_FUNC_DOC_FILE_MAPPING.items():
        path = os.path.join(MT_RAW, "func_doc", fname)
        if os.path.exists(path):
            docs[cls] = [json.loads(line) for line in open(path)]

    answers = {row["id"]: row["ground_truth"] for row in
               map(json.loads, open(os.path.join(MT_RAW, f"answer_{category}.json")))}
    table = None
    if split is not None:
        table = json.load(open(os.path.join(
            os.path.dirname(MT_RAW.rstrip("/")).replace("raw", "data"),
            "mt_splits.json")))
    out = []
    for line in open(os.path.join(MT_RAW, f"{category}.json")):
        raw = json.loads(line)
        if table is not None:
            index = raw["id"].rsplit("_", 1)[1]
            if table[index] != split:
                continue
        funcs = [f for cls in raw["involved_classes"] for f in docs[cls]]
        held: dict[int, list[str]] = {int(t): names for t, names in
                                      (raw.get("missed_function") or {}).items()}
        held_names = {n for names in held.values() for n in names}
        tools = [_normalize(f) for f in funcs if f["name"] not in held_names]
        injections = {t: [_normalize(f) for f in funcs if f["name"] in names]
                      for t, names in held.items()}
        out.append(Entry(raw, answers[raw["id"]], tools, injections))
        if limit and len(out) >= limit:
            break
    return out


def _py_literal(value) -> str:
    """JSON value -> the Python literal the bfcl executor expects."""
    return repr(value)


def to_call_strings(parsed: list[dict]) -> list[str]:
    return [f"{c['name']}({', '.join(f'{k}={_py_literal(v)}' for k, v in c['args'].items())})"
            for c in parsed]


class Driver:
    """Per-run state + the next_message the loop needs.

    `run_name` isolates backend state: the bfcl executor caches instances
    keyed by (model_name, entry id), so distinct run names mean distinct
    worlds and a repeated run starts clean.
    """

    def __init__(self, run_name: str, family: str):
        self.run_name = re.sub(r"[^A-Za-z0-9_]", "_", run_name)
        self.family = family
        self.state: dict[str, dict] = {}

    def _st(self, entry: Entry) -> dict:
        return self.state.setdefault(entry.id, {
            "turn": 0, "step": 0, "turn_calls": [[]], "force_quit": False,
            "turn_texts": [[]]})

    def next_message(self, entry: Entry, history: list[dict]) -> dict | None:
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
            execute_multi_turn_func_call)
        st = self._st(entry)
        text = history[-1]["content"]
        st["turn_texts"][-1].append(text)
        attempt, parsed, _ = calls_mod._parse(text, self.family)

        if attempt and parsed and st["step"] < STEP_LIMIT:
            call_strings = to_call_strings(parsed)
            st["turn_calls"][-1].append(call_strings)
            results, _ = execute_multi_turn_func_call(
                call_strings, entry.raw["initial_config"],
                entry.raw["involved_classes"], self.run_name, entry.id,
                long_context=False)
            st["step"] += 1
            return {"messages": [{"role": "tool", "content": str(r)}
                                 for r in results]}

        if attempt and parsed:
            st["force_quit"] = True          # hit the step limit mid-flow

        # advance to the next user turn
        st["turn"] += 1
        st["step"] = 0
        if st["turn"] >= len(entry.raw["question"]):
            return None
        st["turn_calls"].append([])
        st["turn_texts"].append([])
        step: dict = {}
        if st["turn"] in entry.injections:
            step["tools"] = entry.tools + [
                tool for turn, tools in sorted(entry.injections.items())
                if turn <= st["turn"] for tool in tools]
        turn_msgs = entry.raw["question"][st["turn"]]
        step["messages"] = list(turn_msgs) or [
            {"role": "user", "content": RESTORED_LINE}]
        return step


def score(driver: Driver, items: list[Entry]) -> tuple[list[dict], dict]:
    """Official pass/fail per entry + our per-turn diagnostics."""
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
        multi_turn_checker)
    rows = []
    for entry in items:
        st = driver.state[entry.id]
        category = "multi_turn_" + entry.raw["id"].removeprefix("multi_turn_").rsplit("_", 1)[0]
        try:
            official = multi_turn_checker(
                st["turn_calls"], entry.answer, entry.raw, category,
                f"{driver.run_name}_eval")
            passed, error = bool(official["valid"]), official.get("error_type")
        except Exception as exc:               # checker can raise on odd output
            passed, error = False, f"checker_exception:{type(exc).__name__}"

        blocked = [t for t, gt in enumerate(entry.answer) if not gt]
        diag = {"premature_call_turns": [], "asked_turns": []}
        for t in blocked:
            if t < len(st["turn_calls"]) and any(st["turn_calls"][t]):
                diag["premature_call_turns"].append(t)
            if t < len(st["turn_texts"]) and any(
                    x.strip().endswith("?") for x in st["turn_texts"][t]):
                diag["asked_turns"].append(t)
        rows.append({"id": entry.id, "task": entry.task, "passed": passed,
                     "error": error, "force_quit": st["force_quit"],
                     "n_turns_played": len(st["turn_calls"]),
                     "blocked_turns": blocked, **diag})

    n = len(rows)
    blocked_total = sum(len(r["blocked_turns"]) for r in rows)
    summary = {
        "n": n,
        "pass_rate": sum(r["passed"] for r in rows) / n if n else None,
        "force_quit_rate": sum(r["force_quit"] for r in rows) / n if n else None,
        "premature_call_rate": (sum(len(r["premature_call_turns"]) for r in rows)
                                / blocked_total) if blocked_total else None,
        "asked_rate": (sum(len(r["asked_turns"]) for r in rows)
                       / blocked_total) if blocked_total else None,
    }
    return rows, summary
