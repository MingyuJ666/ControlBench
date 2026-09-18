"""Records, rule summaries, blind judge shards and the final success score.

    success = rule_pass AND coherent AND answer_status == "adequate"

The rule is code (tasks.py). coherent and answer_status come from the blind
judge (judge_core.py), which also catches looping and word salad. A rule
failure is final without a verdict; a rule pass needs one, and a missing
verdict leaves success unknown, so a report can refuse to publish an
incomplete number.
"""

import json
import random
from pathlib import Path

from .tasks import TASKS, score

from .judge_core import JUDGE_INSTRUCTIONS, JUDGE_MODEL, retry_spec, run_spec


def build_records(task: str, split: str, items: list[dict], outputs: list[dict],
                  uid_namespace: str, metadata: dict | None = None) -> list[dict]:
    """Join generated outputs to items with one stable schema; uid = task|split|config|item."""
    verifier = TASKS[task]["verifier"]
    extra = metadata or {}
    return [{
        **extra, **generated, **score(generated["output"], verifier, generated["truncated"]),
        "uid": f"{task}|{split}|{uid_namespace}|{item['item_id']}",
        "item_id": item["item_id"], "prompt_id": item["prompt_id"], "task": task, "split": split,
        "question": item["base_prompt"],
    } for generated, item in zip(outputs, items, strict=True)]


def summarize_rules(records: list[dict]) -> dict:
    n = len(records)
    n_pass = sum(r["rule_pass"] is True for r in records)
    n_trunc = sum(bool(r["truncated"]) for r in records)
    return {"n_total": n, "n_rule_scored": n, "n_rule_pass": n_pass, "n_truncated": n_trunc,
            "rule_pass_rate": n_pass / n if n else None, "truncation_rate": n_trunc / n if n else None}


def prepare_shards(records: list[dict], out_dir: Path, n_batches: int = 4, seed: int = 11) -> int:
    """Write the rule-pass outputs as shuffled blind judge shards; returns how many were sent."""
    passers = [r for r in records if r["rule_pass"] is True]
    rows = [{"uid": r["uid"], "question": r["question"], "answer": r["output"],
             "n_words": len(r["output"].split())} for r in passers]
    random.Random(seed).shuffle(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = {f"judge_in_{k}.jsonl": "".join(json.dumps(row, ensure_ascii=False) + "\n"
                                           for row in rows[k::n_batches]) for k in range(n_batches)}
    old_inputs = {path.name: path.read_text(encoding="utf-8") for path in out_dir.glob("judge_in_*.jsonl")}
    instructions = out_dir / "judge_instructions.txt"
    if old_inputs == shards and instructions.exists() and instructions.read_text(encoding="utf-8") == JUDGE_INSTRUCTIONS:
        try:
            validate_run(out_dir)
        except (OSError, ValueError, TypeError):
            pass
        else:
            return len(passers)
    for pattern in ("judge_in_*.jsonl", "judge_out_*.jsonl", "judge_unparsed_*.jsonl"):
        for path in out_dir.glob(pattern):
            path.unlink()
    (out_dir / "judge_run.json").unlink(missing_ok=True)
    for name, text in shards.items():
        path = out_dir / name
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    (out_dir / "judge_instructions.txt").write_text(JUDGE_INSTRUCTIONS, encoding="utf-8")
    return len(passers)


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
    verdicts = {}
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
                if row.get("judge_mode") and row.get("retry_spec") != retry_spec():
                    raise ValueError(f"verdict {uid!r} has outdated retry settings; rerun judge")
                if type(row.get("coherent")) is not bool:
                    raise ValueError(f"verdict {uid!r} field 'coherent' must be a JSON boolean")
                if row.get("answer_status") not in ("adequate", "partial", "none"):
                    raise ValueError(f"verdict {uid!r} field 'answer_status' must be adequate, partial or none")
                verdicts[uid] = {"coherent": row["coherent"], "answer_status": row["answer_status"],
                                 "note": row.get("note", "")}
    return verdicts


def judging_complete(out_dir: Path) -> bool:
    """Require current judge settings and exactly one valid verdict per input."""
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


def merge(records: list[dict], verdicts: dict[str, dict]) -> None:
    """Attach verdicts and the final success decision to every record in place."""
    for record in records:
        verdict = verdicts.get(record["uid"])
        record["coherent"] = verdict["coherent"] if verdict else None
        record["answer_status"] = verdict["answer_status"] if verdict else None
        record["judge_note"] = verdict.get("note", "") if verdict else None
        if not record["rule_pass"]:
            record["success"] = False
        elif verdict is None:
            record["success"] = None
        else:
            record["success"] = verdict["coherent"] and verdict["answer_status"] == "adequate"


def summarize(records: list[dict]) -> dict:
    """Final numbers; raises if any rule-pass output is still unjudged."""
    rules = summarize_rules(records)
    n = rules["n_rule_scored"]
    if not n:
        raise RuntimeError("no outputs to score")
    unjudged = sum(r["rule_pass"] is True and r.get("success") is None for r in records)
    if unjudged:
        raise RuntimeError(f"{unjudged} rule-pass outputs have no judge verdict")
    passed = [r for r in records if r["rule_pass"] is True]
    return {
        **rules,
        "accuracy": sum(r["success"] is True for r in records) / n,
        "incoherent_rate": sum(r["coherent"] is False for r in passed) / n,
        "partial_rate": sum(r["answer_status"] == "partial" for r in passed) / n,
        "none_rate": sum(r["answer_status"] == "none" for r in passed) / n,
    }
