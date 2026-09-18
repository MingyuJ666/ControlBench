"""Answer parsing and collapse metrics; the judge only handles answer equivalence."""

import re

from l2s.matching import (
    answers_match,
    normalize_problem,
    repetition_collapsed,
    split_trace,
)
from l2s.sources import AMC, MATH500

from .settings import EVAL_SIZES, eval_sets_for


def accepted_letters(answer):
    return set(re.findall(r"\b[A-E]\b", str(answer)))


def parse_letter(answer):
    from l2s.matching import normalize_answer

    normalized = normalize_answer(answer).strip("(){}").upper()
    return normalized if normalized in ("A", "B", "C", "D", "E") else None


def evaluate_output(output, item, tokenizer):
    reasoning, parsed = split_trace(output["text"])
    tokens = tokenizer.encode(reasoning, add_special_tokens=False)
    reasons = []
    if not output["text"].strip():
        reasons.append("empty")
    if output["truncated"]:
        reasons.append("truncated")
    if repetition_collapsed(output["token_ids"]):
        reasons.append("repetition")
    if len(tokens) < 32:
        reasons.append("short_reasoning")
    if not reasoning.strip() and parsed:
        reasons.append("answer_only")
    if item["eval_set"] == "amc12":
        parsed = parse_letter(parsed)
        correct = parsed in accepted_letters(item["gold_answer"])
        needs_judge = False
    else:
        exact = answers_match(parsed, item["gold_answer"])
        needs_judge = bool(parsed) and not exact
        correct = True if exact else None if needs_judge else False
    return dict(
        n_tokens=len(tokens),
        generated_n_tokens=len(output["token_ids"]),
        answer_parsed=parsed,
        correct=correct,
        needs_judge=needs_judge,
        collapsed=bool(reasons),
        collapse_reason=reasons,
    )


def load_eval(pairs, eval_sets=None):
    from datasets import load_dataset

    train_texts = {normalize_problem(row["problem"]) for row in pairs}
    result = []
    selected = eval_sets_for(eval_sets)
    for name, repo, split in [("math500", MATH500, "test"), ("amc12", AMC, "train")]:
        if name not in selected:
            continue
        rows = load_dataset(repo, split=split)
        if len(rows) != EVAL_SIZES[name]:
            raise RuntimeError(f"Unexpected size for {name}")
        ids = set()
        for index, row in enumerate(rows):
            problem = row["problem"] if name == "math500" else row["question"]
            if normalize_problem(problem) in train_texts:
                raise RuntimeError("Evaluation/extraction leakage detected")
            problem_id = (
                row["unique_id"]
                if name == "math500"
                else str(row.get("problem_id") or row.get("id") or f"amc12/{index:04d}")
            )
            if problem_id in ids:
                raise RuntimeError("Duplicate evaluation ID")
            ids.add(problem_id)
            if name == "math500":
                difficulty = int(re.search(r"[1-5]", str(row["level"]))[0])
            else:
                difficulty = int(row["difficulty"])
                if difficulty not in (2, 3, 4) or not accepted_letters(row["answer"]):
                    raise RuntimeError("Unexpected AMC difficulty or answer schema")
            result.append(
                dict(
                    problem_id=problem_id,
                    eval_set=name,
                    problem=problem,
                    gold_answer=row["answer"],
                    difficulty=difficulty,
                    has_figure="[asy]" in problem or "\\begin{asy}" in problem,
                )
            )
    return result
