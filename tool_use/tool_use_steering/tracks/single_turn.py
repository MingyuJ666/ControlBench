"""The single-turn track: one decision, played out over two turns.

The model sees a request and a tool list and decides -- call, ask, or
decline. Then a scripted user supplies what was missing (the withheld value,
the removed tool) and the model acts.

Scoring routes by output type, never by task: output that parses as a call is
read structurally (calls.py); text is read by the judge. On top of the
verdicts sit the two reported numbers per task:

  decision   turn 1 was right
      call      a valid call to the gold tool
      clarify   the reply is a question AND it asks for the withheld slot
      abstain   the reply declines, with no call attempted
  followup   turn 1 was right AND turn 2 acted correctly
      call      = decision (there is no second turn)
      clarify   a valid call to the gold tool whose slot argument is the
                value the user just supplied
      abstain   a valid call to the restored tool

A wrong turn 1 fails followup no matter how good turn 2 is; recovering after
a wrong decision is recorded (recovery) but is not success. sticky --
deciding right and then failing to act on the answer -- is likewise kept as
a diagnostic.

Two artifacts per run, split on purpose: generate() writes raw.jsonl once
(model text, never edited); score() reads it plus the judge and writes
decisions.jsonl and summary.json. A judge change re-runs score(), not the
GPU.
"""

from __future__ import annotations

import json
import os
from collections import Counter

from ..paths import ITEMS, SPLITS
from ..schema import Item, read_items
from ..eval import calls
from ..eval.loop import run as run_loop

TASKS = ("call", "clarify", "abstain")


def pool(split: str, limit: int | None = None,
         tasks: tuple[str, ...] = TASKS) -> list[Item]:
    """The audited items of one split, split by the table.

    `tasks` selects which pools to run (each is always reported separately);
    `limit` caps every selected task at the same count -- the pools are
    different sizes, so equal caps are how a user runs a balanced or a cheap
    slice of dev or test without touching the data.
    """
    table = json.load(open(SPLITS))
    items = []
    for task in tasks:
        rows = [i for i in read_items(os.path.join(ITEMS, f"{task}.jsonl"))
                if table[i.id] == split]
        items += rows[:limit] if limit else rows
    return items


def next_message(item: Item, history: list[dict]) -> dict | None:
    """The scripted user: answer once, then stop."""
    model_turns = sum(1 for m in history if m["role"] == "assistant")
    fu = item.gold.followup
    if fu is None or model_turns >= 2:
        return None
    step = {"message": {"role": "user", "content": fu.user_reply}}
    if fu.restored_tool is not None:
        step["tools"] = item.tools + [fu.restored_tool]
    return step


def generate(model, tokenizer, items: list[Item], method, dose: float,
             batch_size: int = 8, steer_turns: str = "all") -> list[dict]:
    return run_loop(model, tokenizer, items, method, dose, next_message,
                    batch_size=batch_size, steer_turns=steer_turns)


# --- scoring ----------------------------------------------------------------

def _slot_description(item: Item) -> str:
    fu = item.gold.followup
    tool = next((t for t in item.tools
                 if t["name"] == (fu.tool_names or [None])[0]), None)
    props = ((tool or {}).get("parameters") or {}).get("properties") or {}
    return props.get(fu.slot, {}).get("description") or "(no description)"


def score(records: list[dict], items: list[Item], judge_model, judge_tokenizer,
          family: str, judge_batch: int = 16) -> tuple[list[dict], dict]:
    """records + items -> (per-item decisions, summary). `family` names the
    call-format parser for the generating model (calls.family_for)."""
    from ..eval import judge as judge_mod

    by_id = {i.id: i for i in items}
    rows = []
    for record in records:
        item = by_id[record["id"]]
        turns = record["turns"]
        fu = item.gold.followup
        row = {"id": item.id, "task": item.task, "n_turns": len(turns)}

        # -- turn 1, structural
        turn2_tools = item.tools
        if fu is not None and fu.restored_tool is not None:
            turn2_tools = item.tools + [fu.restored_tool]
        v1 = calls.check(turns[0], item.tools, item.gold.tool_names,
                         family=family)
        row["turn1_call"] = v1
        # -- turn 2, structural (when played)
        v2 = None
        if len(turns) > 1:
            v2 = calls.check(turns[1], turn2_tools,
                             fu.tool_names if fu else None, family=family)
            row["turn2_call"] = v2
            if fu is not None and fu.slot is not None:
                row["turn2_used_value"] = calls.used_value(v2, fu.slot, fu.value)
        rows.append(row)

    # -- the judge reads every turn that is not a call attempt
    need_label = [(i, t) for i, row in enumerate(rows)
                  for t in range(row["n_turns"])
                  if not rows[i].get(f"turn{t+1}_call", {}).get("attempt")]
    if need_label:
        judge_rows = []
        for i, t in need_label:
            item = by_id[rows[i]["id"]]
            judge_rows.append({
                "question": item.messages[-1]["content"],
                "tool_names": [tool["name"] for tool in item.tools],
                "text": records[i]["turns"][t],
            })
        labels = judge_mod.classify(judge_model, judge_tokenizer, judge_rows,
                                    judge_batch)
        for (i, t), label in zip(need_label, labels):
            rows[i][f"turn{t+1}_label"] = label

    # -- clarify decisions additionally need the slot question
    need_slot = [i for i, row in enumerate(rows)
                 if row["task"] == "clarify"
                 and row.get("turn1_label") == "clarify"
                 and by_id[row["id"]].gold.followup is not None]
    if need_slot:
        slot_rows = []
        for i in need_slot:
            item = by_id[rows[i]["id"]]
            slot_rows.append({
                "question": item.messages[-1]["content"],
                "slot": item.gold.followup.slot,
                "slot_description": _slot_description(item),
                "text": records[i]["turns"][0],
            })
        hits = judge_mod.asks_for(judge_model, judge_tokenizer, slot_rows,
                                  judge_batch)
        for i, hit in zip(need_slot, hits):
            rows[i]["turn1_asks_slot"] = hit

    # -- decisions and the summary
    summary: dict = {}
    for row in rows:
        task = row["task"]
        v1, v2 = row["turn1_call"], row.get("turn2_call")
        if task == "call":
            decision = bool(v1["tool_call_success"])
            followup = decision
        elif task == "clarify":
            decision = (row.get("turn1_label") == "clarify"
                        and row.get("turn1_asks_slot", False))
            acted = bool(v2 and v2["tool_call_success"]
                         and row.get("turn2_used_value"))
            followup = decision and acted
            row["recovery"] = (not decision) and acted
            row["sticky"] = decision and not acted
        else:  # abstain
            decision = (not v1["attempt"]
                        and row.get("turn1_label") == "abstain")
            acted = bool(v2 and v2["tool_call_success"])
            followup = decision and acted
            row["recovery"] = (not decision) and acted
            row["sticky"] = decision and not acted
        row["decision"], row["followup"] = decision, followup

    present = [t for t in TASKS if any(r["task"] == t for r in rows)]
    for task in present:
        sub = [r for r in rows if r["task"] == task]
        n = len(sub)
        entry = {
            "n": n,
            "decision": sum(r["decision"] for r in sub) / n,
            "followup": sum(r["followup"] for r in sub) / n,
        }
        if task != "call":
            entry["recovery"] = sum(r.get("recovery", False) for r in sub) / n
            entry["sticky"] = sum(r.get("sticky", False) for r in sub) / n
        entry["turn1_labels"] = dict(Counter(
            r.get("turn1_label", "tool_call") for r in sub))
        summary[task] = entry
    return rows, summary
