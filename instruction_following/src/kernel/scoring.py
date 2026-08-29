"""Shared instruction-following scoring for one or more generated configs.

Steering methods own generation.  This module owns the benchmark contract
after generation: stable records, rule denominators, blind judge inputs and
the final rule/coherence/answering score.
"""

from pathlib import Path

from kernel.judge import load_verdicts, merge, prepare
from kernel.tasks import TASKS
from kernel.verifiers import is_degenerate, score


def build_records(task: str, split: str, items: list[dict], outputs: list[dict],
                  uid_namespace: str, metadata: dict | None = None) -> list[dict]:
    """Join generated outputs to benchmark items with one stable schema."""
    verifier = TASKS[task]["verifier"]
    extra = {} if metadata is None else metadata
    records = []
    for generated, item in zip(outputs, items, strict=True):
        records.append({
            **extra,
            **generated,
            **score(generated["output"], verifier, generated["truncated"]),
            "uid": f"{task}|{split}|{uid_namespace}|{item['item_id']}",
            "item_id": item["item_id"],
            "prompt_id": item["prompt_id"],
            "task": task,
            "split": split,
            "question": item["base_prompt"],
        })
    return records


def summarize_rules(records: list[dict]) -> dict:
    """Summarize rules over every record.

    The denominator is the whole item set for every configuration, which is
    what makes two configurations comparable. Truncation is reported beside
    the rate rather than removed from it.
    """
    n = len(records)
    n_rule_pass = sum(row["rule_pass"] is True for row in records)
    n_truncated = sum(bool(row["truncated"]) for row in records)
    return {
        "n_total": n,
        "n_rule_scored": n,
        "n_rule_pass": n_rule_pass,
        "n_truncated": n_truncated,
        "rule_pass_rate": n_rule_pass / n if n else None,
        "truncation_rate": n_truncated / n if n else None,
    }


def prepare_rule_passers(records: list[dict], out_dir: Path,
                         n_batches: int = 4, seed: int = 11) -> int:
    """Only rule-pass outputs need semantic judgments for final accuracy."""
    passers = [row for row in records if row["rule_pass"] is True]
    prepare(passers, out_dir, n_batches=n_batches, seed=seed)
    return len(passers)


def load_all_verdicts(judging_dirs: list[Path]) -> dict[str, dict]:
    """Combine judge stages while rejecting duplicate record identities."""
    combined = {}
    for judging_dir in judging_dirs:
        for uid, verdict in load_verdicts(judging_dir).items():
            if uid in combined:
                raise ValueError(f"Duplicate judge verdict uid {uid!r}")
            combined[uid] = verdict
    return combined


def merge_judgments(records: list[dict], judging_dirs: list[Path]) -> dict:
    """Attach verdicts and return the shared completeness diagnostics."""
    return merge(records, load_all_verdicts(judging_dirs))


def summarize_merged(records: list[dict]) -> dict:
    """Summarize records after ``merge_judgments`` attached final decisions."""
    rules = summarize_rules(records)
    n = rules["n_rule_scored"]
    if not n:
        raise RuntimeError("No outputs are scorable under the rule verifier")
    unjudged = sum(
        row["rule_pass"] is True and row.get("success") is None
        for row in records
    )
    if unjudged:
        raise RuntimeError(
            f"{unjudged} rule-pass outputs have no judge verdict"
        )
    successes = sum(row.get("success") is True for row in records)
    incoherent = sum(
        row["rule_pass"] is True and row.get("coherent") is False
        for row in records
    )
    not_answering = sum(
        row["rule_pass"] is True and row.get("answers_question") is False
        for row in records
    )
    degenerate = sum(
        row["rule_pass"] is True and is_degenerate(row["output"])
        for row in records
    )
    judge_suspect = sum(
        bool(row.get("coherent")) and is_degenerate(row["output"])
        for row in records
    )
    return {
        **rules,
        "accuracy": successes / n,
        "n_success": successes,
        "n_incoherent": incoherent,
        "incoherent_rate": incoherent / n,
        "n_not_answering": not_answering,
        "not_answering_rate": not_answering / n,
        "n_degenerate": degenerate,
        "degenerate_rate": degenerate / n,
        "judge_suspect": judge_suspect,
    }
