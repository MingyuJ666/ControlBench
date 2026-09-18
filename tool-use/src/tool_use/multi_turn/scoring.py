"""BFCL execution and no-call checks, plus one semantic verdict on the missing turn.

The missing turn is annotated before generation; it is never selected from the
model's mistakes. Call/base conversations need no LLM verdict. For missing-value
or missing-tool conversations, whole-conversation success additionally requires
the target action, coherent behavior and appropriate content on that turn.
"""
import uuid

from .. import calls as calls_mod
from .decision import decision_turn
from .driver import call_strings, step_limit


def group_turns(replies: list[str]) -> tuple[list[list[list[str]]], list[list[str]]]:
    """Replies -> (calls per user turn per step, replies per user turn).

    The same rule the driver played by: a parsed call keeps the turn open and
    counts a step, anything else closes the turn.
    """
    per_turn_calls: list[list[list[str]]] = [[]]
    per_turn_texts: list[list[str]] = [[]]
    step = 0
    for reply in replies:
        per_turn_texts[-1].append(reply)
        attempt, parsed, _, _ = calls_mod.parse(reply)
        if attempt and parsed and step < step_limit():
            per_turn_calls[-1].append(call_strings(parsed))
            step += 1
            continue
        per_turn_calls.append([])
        per_turn_texts.append([])
        step = 0
    if per_turn_texts and not per_turn_texts[-1]:      # the last close opened an empty turn
        per_turn_calls.pop()
        per_turn_texts.pop()
    return per_turn_calls, per_turn_texts


def score_item(item: dict, replies: list[str]) -> dict:
    from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils as backend
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
        multi_turn_checker, multi_turn_irrelevance_checker)

    selected = decision_turn(item)
    per_turn_calls, per_turn_texts = group_turns(replies)
    namespace = "cb_score_" + uuid.uuid4().hex
    if len(per_turn_calls) != len(item["answer"]):
        execution = irrelevance = {"valid": False, "error_type": "multi_turn:turn_count_mismatch"}
    else:
        # BFCL's execution checker skips empty reference turns. Its separate
        # irrelevance checker must also pass; otherwise premature calls escape.
        irrelevance = multi_turn_irrelevance_checker(per_turn_calls, item["answer"])
        try:
            execution = multi_turn_checker(per_turn_calls, item["answer"], item["entry"],
                                           f"multi_turn_{item['category']}", namespace)
        except Exception as exc:
            execution = {"valid": False, "error_type": f"checker_exception:{type(exc).__name__}"}
        finally:
            for name in list(vars(backend)):
                if name.startswith(namespace + "_") and name.endswith("_instance"):
                    delattr(backend, name)

    blocked = [index for index, reference in enumerate(item["answer"]) if not reference]
    premature = [index for index in blocked
                 if index < len(per_turn_calls) and any(per_turn_calls[index])]
    asked = [index for index in blocked
             if index < len(per_turn_texts)
             and any(text.strip().endswith("?") for text in per_turn_texts[index])]
    bfcl_passed = bool(execution["valid"] and irrelevance["valid"])
    return {"item_id": item["item_id"], "target": item["target"], "category": item["category"],
            "execution_passed": bool(execution["valid"]),
            "irrelevance_passed": bool(irrelevance["valid"]), "bfcl_passed": bfcl_passed,
            "passed": bfcl_passed if selected is None else None,
            "error": execution.get("error_type") or irrelevance.get("error_type"),
            "execution_error": execution.get("error_type"),
            "irrelevance_error": irrelevance.get("error_type"),
            "judge_turn_index": selected, "decision": None,
            "decision_has_response": (bool(selected < len(per_turn_texts)
                                           and any(text.strip() for text in per_turn_texts[selected]))
                                      if selected is not None else None),
            "n_turns_played": len(per_turn_calls), "n_turns": len(item["answer"]),
            "blocked_turns": blocked, "premature_call_turns": premature, "asked_turns": asked}


def build_records(target: str, split: str, items: list[dict], outputs: list[dict],
                  config: str, metadata: dict | None = None) -> list[dict]:
    records = []
    for generated, item in zip(outputs, items, strict=True):
        if item["target"] != target or generated.get("item_id", item["item_id"]) != item["item_id"]:
            raise ValueError("multi-turn outputs and items are misaligned")
        played = generated.get("turns") or [generated]
        replies = [turn["output"] for turn in played]
        records.append({**(metadata or {}),
                        "uid": f"{target}|{split}|{config}|{item['item_id']}",
                        "split": split, "n_replies": len(replies),
                        "truncated": any(turn["truncated"] for turn in played),
                        **score_item(item, replies)})
    return records


def merge(records: list[dict], verdicts: dict[str, dict]) -> None:
    from ...tool_use_multi_turn.judge_core import _fields
    for record in records:
        index = record["judge_turn_index"]
        if index is None:
            continue
        judgment = verdicts.get(f"{record['uid']}|turn_{index}")
        if judgment is not None and _fields(judgment) is None:
            raise ValueError("incomplete multi-turn judge verdict")
        for field in ("action", "coherent", "appropriate"):
            record[field] = (judgment or {}).get(field)
        record["judge_note"] = (judgment or {}).get("note")
        record["decision"] = (bool(judgment["action"] == record["target"]
                                   and judgment["coherent"] and judgment["appropriate"]
                                   and record["decision_has_response"]
                                   and index not in record["premature_call_turns"])
                              if judgment is not None else None)
        record["passed"] = (bool(record["bfcl_passed"] and record["decision"])
                            if judgment is not None else None)


def summarize(records: list[dict]) -> dict:
    n = len(records)
    if not n:
        raise RuntimeError("no conversations to score")
    judged = [r for r in records if r["judge_turn_index"] is not None]
    if any(r["decision"] is None for r in judged):
        raise RuntimeError("missing multi-turn judge verdicts; refusing incomplete scores")
    blocked = sum(len(record["blocked_turns"]) for record in records)
    return {
        "accuracy": sum(record["passed"] for record in records) / n,
        "bfcl_accuracy": sum(record["bfcl_passed"] for record in records) / n,
        "decision_accuracy": (sum(r["decision"] for r in judged) / len(judged) if judged else None),
        "coherent_rate": (sum(r["coherent"] for r in judged) / len(judged) if judged else None),
        "appropriate_rate": (sum(r["appropriate"] for r in judged) / len(judged) if judged else None),
        "premature_call_rate": (sum(len(r["premature_call_turns"]) for r in records) / blocked
                                if blocked else None),
        "asked_rate": (sum(len(r["asked_turns"]) for r in records) / blocked if blocked else None),
        "truncation_rate": sum(bool(r["truncated"]) for r in records) / n,
        "n_blocked_turns": blocked, "n_judged": len(judged), "n_total": n,
    }
