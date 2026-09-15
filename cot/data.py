"""JSONL inputs, fixed split checks, and final boxed-answer scoring."""
import hashlib
import json
from pathlib import Path

DEFAULT_DATA = Path(__file__).resolve().parent / "data"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def problem_id(problem):
    return hashlib.sha256(" ".join(problem.split()).encode()).hexdigest()


def load(directory):
    directory = Path(directory)
    rows = read_jsonl(directory / "problems.jsonl")
    rows = [dict(row, id=problem_id(row["problem"])) for row in rows]
    lookup = {row["id"]: row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate problems")
    split = json.loads((directory / "split.json").read_text())
    seen = set()
    for name in ("extraction", "dev", "eval"):
        ids = split[name]
        if len(ids) != len(set(ids)) or seen.intersection(ids) or not set(ids) <= lookup.keys():
            raise ValueError(f"Invalid or overlapping {name} split")
        seen.update(ids)
    pairs = read_jsonl(directory / "pairs.jsonl")
    if not pairs or len({p["id"] for p in pairs}) != len(pairs):
        raise ValueError("Training pairs must be nonempty and unique")
    for row in pairs:
        if (row["id"] not in split["extraction"] or
                problem_id(row["problem"]) != row["id"] or
                not row["positive"].strip() or not row["negative"].strip()):
            raise ValueError("Invalid training pair or training/evaluation overlap")
    return {name: [lookup[i] for i in split[name]] for name in ("extraction", "dev", "eval")}, pairs


def boxed(text):
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    begin, depth = start + len("\\boxed{"), 1
    for i in range(begin, len(text)):
        if text[i] == "{" and text[i - 1] != "\\":
            depth += 1
        elif text[i] == "}" and text[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[begin:i].strip() or None
    return None


def score(completion, gold):
    from math_verify import LatexExtractionConfig, parse, verify

    answer = boxed(completion)
    if answer is None:
        return {"correct": False, "parsed_answer": None, "scorable": False}
    config = [LatexExtractionConfig()]
    target = parse("\\boxed{" + str(gold) + "}", extraction_config=config)
    prediction = parse("\\boxed{" + answer + "}", extraction_config=config)
    scorable = bool(target and prediction)
    return {"correct": bool(scorable and verify(target, prediction)),
            "parsed_answer": answer, "scorable": scorable}
