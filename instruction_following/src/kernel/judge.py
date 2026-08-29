"""Prepare blind judge batches and combine judge verdicts with rule scores.

Rule verifiers decide whether an output satisfies the requested surface form.
The judge supplies the two semantic decisions that rules cannot make:

    success = rule_pass AND coherent AND answers_question AND NOT degenerate

The degeneracy term is a rule, not a judge decision: an output that collapses
into repeated text is not a valid answer, and the judge was observed calling
such outputs coherent when a sound opening paragraph preceded the collapse.
Rule failures are final without a judge verdict. Rule-pass outputs require a
verdict; if one is missing, ``merge()`` leaves their success unknown so report
callers can refuse to publish an incomplete score.
"""

import json
import random
from pathlib import Path

from kernel.verifiers import is_degenerate


JUDGE_INSTRUCTIONS = """\
You are auditing text produced by a language model. For each item you get the
user's question and the model's full answer. Judge exactly two things.

coherent — is this real, well-formed text?
  false only for: looping or repeated phrases, word salad, strings of filler
  characters or digits, empty output.

answers_question — does it make a genuine attempt to answer the user's
question?
  true for a complete or partial answer, including one that drifts off part-way
  false if it only restates the question, refuses an answerable question,
  answers a different question, or invents facts wholesale

NOT defects — never mark these down:
  - unusual punctuation, including text with no commas at all
  - unusual capitalisation, including all-lowercase text
  - brevity: a single-sentence or few-word answer can be complete
  - an answer that stops mid-sentence at the end (a length cap, not a failure);
    judge the visible text on its merits
  - plain, stilted or repetitive-sounding phrasing that is still grammatical

Write one JSON object per input line:
{"id": "...", "coherent": true/false, "answers_question": true/false, "note": "<optional short reason>"}
Exactly one output line per input line."""


def prepare(records: list[dict], out_dir: Path, n_batches: int = 3,
            seed: int = 0) -> list[Path]:
    """Replace stale judge artifacts with deterministic blind input batches.

    Each record must contain ``uid``, ``question``, and ``output``. Questions
    and answers are both kept in full because either can contain information
    needed to decide whether the output answers the question.
    """
    rows = [{"uid": r["uid"], "question": r["question"],
             "answer": r["output"], "n_words": len(r["output"].split())}
            for r in records]
    random.Random(seed).shuffle(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    for pattern in (
        "judge_in_*.jsonl", "judge_out_*.jsonl", "judge_unparsed_*.jsonl",
    ):
        for path in out_dir.glob(pattern):
            path.unlink()
    (out_dir / "judge_run.json").unlink(missing_ok=True)

    paths = []
    for k in range(n_batches):
        path = out_dir / f"judge_in_{k}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows[k::n_batches]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        paths.append(path)
    (out_dir / "judge_instructions.txt").write_text(
        JUDGE_INSTRUCTIONS, encoding="utf-8"
    )
    return paths


def load_verdicts(out_dir: Path) -> dict[str, dict]:
    """Load verdict files while rejecting score-changing schema errors."""
    verdicts = {}
    for path in sorted(out_dir.glob("judge_out_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {path} at line {line_number}"
                    ) from error
                if not isinstance(row, dict):
                    raise ValueError(
                        f"Verdict in {path} at line {line_number} "
                        "must be a JSON object"
                    )

                uid = row.get("uid")
                if not isinstance(uid, str) or not uid:
                    raise ValueError(
                        f"Verdict in {path} at line {line_number} "
                        "has no non-empty string uid"
                    )
                if uid in verdicts:
                    raise ValueError(f"Duplicate judge verdict uid {uid!r}")
                for field in ("coherent", "answers_question"):
                    if type(row.get(field)) is not bool:
                        raise ValueError(
                            f"Verdict {uid!r} field {field!r} must be a "
                            "JSON boolean"
                        )

                note = row.get(
                    "note", row.get("raw_response", row.get("raw", ""))
                )
                if not isinstance(note, str):
                    raise ValueError(
                        f"Verdict {uid!r} field 'note' must be a string"
                    )
                verdicts[uid] = {
                    "coherent": row["coherent"],
                    "answers_question": row["answers_question"],
                    "note": note,
                }
    return verdicts


def write_scored_records(records: list[dict], path: Path) -> Path:
    """Atomically save the per-item records produced by ``merge()``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)
    return path


def merge(records: list[dict], verdicts: dict[str, dict]) -> dict:
    """Attach final per-record decisions and return scoring diagnostics.

    Accuracy is available only when every rule-pass output has a verdict.
    Rule failures remain failures without judging — including outputs that ran
    into the generation cap, which the rule scorer fails outright.
    """
    n_rule_pass = 0
    n_rule_fail = 0
    n_judged = 0
    n_success = 0

    n_degenerate_blocked = 0

    for record in records:
        verdict = verdicts.get(record["uid"])
        record["degenerate"] = is_degenerate(record["output"])
        record["coherent"] = verdict["coherent"] if verdict else None
        record["answers_question"] = (
            verdict["answers_question"] if verdict else None
        )
        record["judge_note"] = verdict.get("note", "") if verdict else None

        if not record["rule_pass"]:
            record["success"] = False
            n_rule_fail += 1
        else:
            n_rule_pass += 1
            if verdict is None:
                record["success"] = None
                continue
            n_judged += 1
            judged_good = verdict["coherent"] and verdict["answers_question"]
            record["success"] = judged_good and not record["degenerate"]
            n_degenerate_blocked += judged_good and record["degenerate"]
            n_success += record["success"]

    n_scored = n_rule_pass + n_rule_fail
    n_unjudged = n_rule_pass - n_judged
    complete = n_unjudged == 0
    return {
        "n_records": len(records),
        "n_scored": n_scored,
        "n_rule_pass": n_rule_pass,
        "n_rule_fail": n_rule_fail,
        "n_judged": n_judged,
        "n_unjudged": n_unjudged,
        "n_success": n_success,
        "n_degenerate_blocked": n_degenerate_blocked,
        "accuracy": n_success / n_scored if complete and n_scored else None,
        "rule_only_rate": n_rule_pass / n_scored if n_scored else None,
        "judge_coverage": n_judged / n_rule_pass if n_rule_pass else 1.0,
    }
