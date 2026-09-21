"""Read-only validation of supplied Safety data; never executes actions."""

from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import re

MODELS = {
    "qwen3_8b": "Qwen/Qwen3-8B",
    "gemma2_9b": "google/gemma-2-9b-it",
    "llama31_8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}
INPUTS = Path("inputs")


def snapshot_key(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_new(path, value):
    """Atomic, no-clobber publication; partial temporary files are not checkpoints."""
    import os
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def snapshot(row):
    return {k: row[k] for k in ("task_description", "input", "output")}


def load(root, model, split):
    root = Path(root)
    manifest_path = root / INPUTS / "MANIFEST.json"
    manifest = read(manifest_path)
    if model not in MODELS or set(manifest["models"]) != set(MODELS):
        raise ValueError("Unknown or incomplete model manifest")
    spec = manifest["models"][model]
    inner = split in ("fit", "head_validation")
    if inner:
        entry = spec["iti"][split]
        extraction, path = load(root, model, "extraction")
        by_id = {row["sample_id"]: row for row in extraction}
        ids = entry["sample_ids"]
        if len(ids) != len(set(ids)) or set(ids) - set(by_id):
            raise ValueError("Invalid ITI membership")
        rows = [by_id[sample_id] for sample_id in ids]
    else:
        if split not in ("extraction", "eval"):
            raise ValueError("Unknown split")
        entry = spec["files"][split]
        path = root / INPUTS / entry["path"]
        rows = read_jsonl(path)
    if len(rows) != entry["rows"]:
        raise ValueError("Wrong row count")
    ids = set()
    for row in rows:
        if type(row["label"]) is not int or row["label"] not in (0, 1):
            raise ValueError("Invalid gold")
        if row["model"] != MODELS[model]:
            raise ValueError("Wrong model")
        if row["dataset_split"] != ("extraction" if inner else split):
            raise ValueError("Wrong split")
        if row["sample_id"] in ids:
            raise ValueError("Duplicate sample ID")
        ids.add(row["sample_id"])
        snapshot(row)
    counts = Counter(r["label"] for r in rows)
    if (counts[0], counts[1]) != (entry["safe"], entry["unsafe"]):
        raise ValueError("Label counts differ")
    return rows, path


def audit(root):
    root = Path(root)
    manifest = read(root / INPUTS / "MANIFEST.json")
    if set(manifest) != {"models"} or set(manifest["models"]) != set(MODELS):
        raise ValueError("Invalid manifest")
    report = {"schema": 1, "checks": [], "models": {}}
    for model in MODELS:
        splits = {}
        summary = {}
        for split in ("extraction", "eval", "fit", "head_validation"):
            rows, path = load(root, model, split)
            splits[split] = rows
            lengths = sorted(
                len(json.dumps(snapshot(r), ensure_ascii=False)) for r in rows
            )
            summary[split] = {
                "n": len(rows),
                "safe": sum(r["label"] == 0 for r in rows),
                "unsafe": sum(r["label"] == 1 for r in rows),
                "task_groups": len({r["task_group_id"] for r in rows}),
                "snapshot_characters": {
                    "max": max(lengths),
                    "p50": lengths[len(lengths) // 2],
                    "p95": lengths[int(0.95 * (len(lengths) - 1))],
                },
                "mixed_label_task_groups": sum(
                    len({r["label"] for r in rows if r["task_group_id"] == g}) > 1
                    for g in {r["task_group_id"] for r in rows}
                ),
            }
        key_functions = {
            "sample": lambda r: r["sample_id"],
            "task_group": lambda r: r["task_group_id"],
            "raw_task": lambda r: (r["source_file"], str(r["task_id"])),
            "trajectory": lambda r: r["trajectory_id"],
            "normalized_task": lambda r: re.sub(
                r"[^\w]+", " ", r["task_description"].lower()
            ).strip(),
            "snapshot": lambda r: snapshot_key(snapshot(r)),
        }
        for name, key in key_functions.items():
            if {key(r) for r in splits["extraction"]} & {
                key(r) for r in splits["eval"]
            }:
                raise ValueError(f"{model}: {name} overlap extraction/eval")
        task_registry = {}
        for row in splits["extraction"] + splits["eval"]:
            key = (row["source_file"], str(row["task_id"]))
            previous = task_registry.setdefault(key, row["task_group_id"])
            if previous != row["task_group_id"]:
                raise ValueError("Task registry mismatch")
        if {r["task_group_id"] for r in splits["fit"]} & {
            r["task_group_id"] for r in splits["head_validation"]
        }:
            raise ValueError("ITI group overlap")
        extraction = {r["sample_id"]: r for r in splits["extraction"]}
        inner = {r["sample_id"]: r for r in splits["fit"] + splits["head_validation"]}
        if extraction.keys() != inner.keys() or len(inner) != len(splits["fit"]) + len(
            splits["head_validation"]
        ):
            raise ValueError("ITI union")
        for sid, r in inner.items():
            for key in extraction[sid]:
                if r[key] != extraction[sid][key]:
                    raise ValueError(f"ITI record changed: {sid} {key}")
        report["models"][model] = summary
    if (
        sum(
            v[s]["n"]
            for v in report["models"].values()
            for s in ("extraction", "eval")
        )
        != 4729
    ):
        raise ValueError("Total mismatch")
    report["checks"] = [
        "12 split counts and labels",
        "model and binary labels",
        "task registry consistency",
        "6 kinds of extraction/eval disjointness",
        "ITI disjoint groups and unchanged union",
        "complete input manifest",
    ]
    report["status"] = "passed"
    report["annotation"] = "Model annotations, not human gold"
    return report
