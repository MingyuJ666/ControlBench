"""One blind judgment per response, plus deterministic call validation.

Success requires the target action, coherent output and appropriate content.
Submitted calls additionally require a valid invocation of the expected tool.
Only clarify/abstain items can script a second response. That response has its
own judgment using the actual history and the tools available after follow-up.
A missing verdict stays unknown; it is never silently counted as failure.
"""

import json
import random
from pathlib import Path

from .. import calls
from .followup import check_second
from ..judge_core import FIRST_INSTRUCTIONS, FOLLOWUP_INSTRUCTIONS, JUDGE_MODEL, _fields, run_spec


def build_records(target: str, split: str, items: list[dict], outputs: list[dict],
                  config: str, metadata: dict | None = None) -> list[dict]:
    """Join saved outputs to inputs without rerendering or regenerating replies."""
    records = []
    for generated, item in zip(outputs, items, strict=True):
        followup = item["gold"].get("followup")
        if target == "call" and followup is not None:
            raise ValueError(f"{item['item_id']}: call items cannot have a follow-up")
        played = generated.get("turns") or []
        expected_turns = 2 if followup is not None else 1
        if played and (len(played) != expected_turns or played[0]["output"] != generated["output"]):
            raise ValueError(f"{item['item_id']}: saved turns do not match the scripted conversation")
        if followup is not None and len(played) != 2:
            raise ValueError(f"{item['item_id']}: missing scripted follow-up response")
        second_text = played[1]["output"] if followup is not None else None
        second = check_second(item, second_text) if second_text is not None else None
        current_tools = item["tools"]
        if followup and followup.get("restored_tool"):
            current_tools = current_tools + [followup["restored_tool"]]
        history = (item["messages"] + [
            {"role": "assistant", "content": generated["output"]},
            {"role": "user", "content": followup["user_reply"]}]
            if followup is not None else None)
        slot = (followup or {}).get("slot")
        records.append({
            **(metadata or {}), **generated,
            "uid": f"{target}|{split}|{config}|{item['item_id']}",
            "item_id": item["item_id"], "target": target, "split": split,
            "question": item["messages"][-1]["content"],
            "messages": item["messages"], "tools": item["tools"],
            "tool_names": [tool.get("function", tool)["name"] for tool in item["tools"]],
            "call": calls.check(generated["output"], item["tools"], item["gold"].get("tool_names")),
            "second": second, "second_output": second_text,
            "followup_messages": history, "followup_tools": current_tools if history is not None else None,
            # Diagnostic only: semantic equivalence is assessed against the full history by the judge.
            "second_used_value": (calls.used_value(second, slot, followup["value"])
                                  if second is not None and slot else None),
        })
    return records


def judge_rows(records: list[dict]) -> list[dict]:
    """Exactly one row per assistant response; no target or hidden gold in prompts."""
    rows = []
    for record in records:
        rows.append({"uid": f"{record['uid']}|first", "kind": "first",
                     "messages": record["messages"], "tools": record["tools"],
                     "answer": record["output"]})
        if record["second"] is not None:
            rows.append({"uid": f"{record['uid']}|followup", "kind": "followup",
                         "messages": record["followup_messages"], "tools": record["followup_tools"],
                         "answer": record["second_output"]})
    return rows


def prepare_shards(records: list[dict], out_dir: Path, n_batches: int = 4, seed: int = 11) -> int:
    """Write the questions as shuffled blind shards; returns how many were sent."""
    rows = judge_rows(records)
    random.Random(seed).shuffle(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = {f"judge_in_{k}.jsonl": "".join(json.dumps(row, ensure_ascii=False) + "\n"
                                             for row in rows[k::n_batches]) for k in range(n_batches)}
    instructions = FIRST_INSTRUCTIONS + "\n\n" + FOLLOWUP_INSTRUCTIONS
    old = {path.name: path.read_text(encoding="utf-8") for path in out_dir.glob("judge_in_*.jsonl")}
    written = out_dir / "judge_instructions.txt"
    if old == shards and written.exists() and written.read_text(encoding="utf-8") == instructions:
        try:
            validate_run(out_dir)
        except (OSError, ValueError, TypeError):
            pass
        else:
            return len(rows)
    for pattern in ("judge_in_*.jsonl", "judge_out_*.jsonl", "judge_unparsed_*.jsonl"):
        for path in out_dir.glob(pattern):
            path.unlink()
    (out_dir / "judge_run.json").unlink(missing_ok=True)
    for name, text in shards.items():
        path = out_dir / name
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    written.write_text(instructions, encoding="utf-8")
    return len(rows)


def validate_run(out_dir: Path) -> None:
    shards = [path.name for path in sorted(out_dir.glob("judge_in_*.jsonl"))]
    if not shards:
        raise ValueError(f"no judge inputs under {out_dir}")
    actual = json.loads((out_dir / "judge_run.json").read_text())
    expected = run_spec(shards, JUDGE_MODEL, "vllm", {})
    if not isinstance(actual, dict):
        raise ValueError(f"invalid judge manifest under {out_dir}")
    mismatched = [key for key, value in expected.items() if actual.get(key) != value]
    if mismatched:
        raise ValueError(f"{out_dir}: judge settings differ: {', '.join(mismatched)}; rerun judge")


def load_verdicts(out_dir: Path) -> dict[str, dict]:
    validate_run(out_dir)
    verdicts: dict[str, dict] = {}
    for path in sorted(out_dir.glob("judge_out_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"invalid verdict in {path} line {number}")
                uid = row.get("uid")
                if not isinstance(uid, str) or not uid:
                    raise ValueError(f"verdict in {path} line {number} has no uid")
                if uid in verdicts:
                    raise ValueError(f"duplicate judge verdict uid {uid!r}")
                if uid.rsplit("|", 1)[-1] not in ("first", "followup"):
                    raise ValueError(f"verdict {uid!r} has no known response kind")
                fields = _fields(row)
                if fields is None:
                    raise ValueError(f"verdict {uid!r} requires action, coherent, appropriate and note")
                verdicts[uid] = fields
    return verdicts


def judging_complete(out_dir: Path) -> bool:
    """Require current judge settings and exactly one valid verdict per question."""
    try:
        verdicts = load_verdicts(out_dir)
        expected = []
        for path in sorted(out_dir.glob("judge_in_*.jsonl")):
            ids = [json.loads(line)["uid"] for line in path.read_text().splitlines()]
            out = path.with_name(path.name.replace("judge_in_", "judge_out_"))
            actual = [json.loads(line)["uid"] for line in out.read_text().splitlines()]
            if len(ids) != len(actual) or set(ids) != set(actual):
                return False
            expected.extend(ids)
        return len(expected) == len(set(expected)) and set(expected) == set(verdicts)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _success(target: str, structural: dict, judge: dict | None) -> bool | None:
    if judge is None:
        return None
    if _fields(judge) is None:
        raise ValueError("incomplete tool-use judge verdict")
    return bool(judge["action"] == target and judge["coherent"] and judge["appropriate"]
                and (target != "call" or structural["success"]))


def merge(records: list[dict], verdicts: dict[str, dict]) -> None:
    """Score each response independently, then compute paired outcomes."""
    for record in records:
        first = verdicts.get(f"{record['uid']}|first")
        second = verdicts.get(f"{record['uid']}|followup") if record["second"] is not None else None
        for prefix, judgment in (("", first), ("second_", second)):
            for field in ("action", "coherent", "appropriate"):
                record[prefix + field] = (judgment or {}).get(field)
            record[prefix + "judge_note"] = (judgment or {}).get("note")
        record["decision"] = _success(record["target"], record["call"], first)
        record["acted"] = (_success("call", record["second"], second)
                           if record["second"] is not None else None)
        if record["acted"] is None or record["decision"] is None:
            record["followup"] = record["recovery"] = record["sticky"] = None
        else:
            record["followup"] = record["decision"] and record["acted"]
            record["recovery"] = (not record["decision"]) and record["acted"]
            record["sticky"] = record["decision"] and not record["acted"]


def summarize(records: list[dict]) -> dict:
    """Refuse incomplete judgments in either round. Keep paired denominators."""
    n = len(records)
    if not n:
        raise RuntimeError("no outputs to score")
    scripted = [r for r in records if r["second"] is not None]
    unjudged = (sum(r.get("decision") is None for r in records)
                + sum(r.get("acted") is None for r in scripted))
    if unjudged:
        raise RuntimeError(f"{unjudged} replies have no judge verdict")
    attempts = [r for r in records if r["action"] == "call"]
    return {
        "n_followup": len(scripted),
        "followup_accuracy": (sum(r["followup"] is True for r in scripted) / len(scripted)
                              if scripted else None),
        "recovery_rate": (sum(r["recovery"] is True for r in scripted) / len(scripted)
                          if scripted else None),
        "sticky_rate": (sum(r["sticky"] is True for r in scripted) / len(scripted)
                        if scripted else None),
        "accuracy": sum(r["decision"] is True for r in records) / n,
        "coherent_rate": sum(r["coherent"] for r in records) / n,
        "appropriate_rate": sum(r["appropriate"] for r in records) / n,
        "call_rate": len(attempts) / n,
        "valid_call_rate": sum(r["call"]["valid"] for r in attempts) / n,
        "clarify_rate": sum(r["action"] == "clarify" for r in records) / n,
        "abstain_rate": sum(r["action"] == "abstain" for r in records) / n,
        "direct_or_other_rate": sum(r["action"] == "direct_or_other" for r in records) / n,
        "bfcl_syntax_rate": (sum(r["call"]["syntax"] == "bfcl" for r in attempts) / len(attempts)
                             if attempts else None),
        "truncation_rate": sum(bool(r["truncated"]) for r in records) / n,
        "n_total": n,
    }
