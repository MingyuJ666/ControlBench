"""Read-only validation of the existing Safety v2 data; never executes actions."""

from __future__ import annotations
from collections import Counter
import filecmp
import itertools
import json
from pathlib import Path
import re

MODELS = {
    "qwen3_8b": "Qwen/Qwen3-8B",
    "gemma2_9b": "google/gemma-2-9b-it",
    "llama31_8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}
CANONICAL = Path("canonical")
OUTER = Path("outer")


def snapshot_key(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def load(root, model, split, *, canonical=CANONICAL, outer=OUTER):
    manifest = read(Path(root) / canonical / model / "model_split_manifest.json")
    inner = split in ("fit", "head_validation")
    entry = manifest["iti" if inner else "outer"][split]
    path = Path(root) / (canonical if inner else outer) / entry["path"]
    rows = read(path)
    if not isinstance(rows, list) or len(rows) != entry["steps"]:
        raise ValueError("Wrong row count")
    ids = set()
    for row in rows:
        if type(row["label"]) is not int or row["label"] not in (0, 1):
            raise ValueError("Invalid gold")
        if row["model"] != MODELS[model]:
            raise ValueError("Wrong model")
        if row["dataset_split"] != ("train" if inner else split):
            raise ValueError("Wrong split")
        if inner and row["iti_split"] != split:
            raise ValueError("Wrong ITI split")
        if row["sample_id"] in ids:
            raise ValueError("Duplicate sample ID")
        ids.add(row["sample_id"])
        snapshot(row)
    counts = Counter(r["label"] for r in rows)
    if (counts[0], counts[1]) != (entry["safe"], entry["unsafe"]):
        raise ValueError("Label counts differ")
    return rows, path


def audit(root, *, canonical=CANONICAL, outer=OUTER):
    root = Path(root)
    canonical_rel = Path(canonical)
    canonical = root / canonical_rel
    outer = Path(outer)
    manifest = read(canonical / "split_manifest.json")
    membership = {
        r["sample_id"]: r for r in read(canonical / "sample_split_membership.json")
    }
    registry = {}
    for group in read(canonical / "task_group_registry.json")["groups"]:
        for member in group["members"]:
            registry[(member["source_file"], str(member["task_id"]))] = group[
                "task_group_id"
            ]
    report = {"schema": 1, "checks": [], "models": {}}
    for model in MODELS:
        mm = read(canonical / model / "model_split_manifest.json")
        if mm != next(m for m in manifest["models"] if m["model"] == model):
            raise ValueError("Model manifest mismatch")
        splits = {}
        summary = {}
        for split in ("train", "dev", "test", "fit", "head_validation"):
            rows, path = load(root, model, split, canonical=canonical_rel, outer=outer)
            splits[split] = rows
            entry = mm["outer" if split in ("train", "dev", "test") else "iti"][split]
            if split in ("train", "dev", "test"):
                if not filecmp.cmp(canonical / entry["path"], path, shallow=False):
                    raise ValueError("Copy differs")
            for r in rows:
                m = membership[r["sample_id"]]
                if (
                    m["model"] != model
                    or m["dataset_split"] != r["dataset_split"]
                    or m["task_group_id"] != r["task_group_id"]
                ):
                    raise ValueError("Membership mismatch")
                if split in ("fit", "head_validation") and m["iti_split"] != split:
                    raise ValueError("Inner membership mismatch")
                if (
                    registry[(r["source_file"], str(r["task_id"]))]
                    != r["task_group_id"]
                ):
                    raise ValueError("Registry mismatch")
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
            for a, b in itertools.combinations(("train", "dev", "test"), 2):
                if {key(r) for r in splits[a]} & {key(r) for r in splits[b]}:
                    raise ValueError(f"{model}: {name} overlap {a}/{b}")
        if {r["task_group_id"] for r in splits["fit"]} & {
            r["task_group_id"] for r in splits["head_validation"]
        }:
            raise ValueError("ITI group overlap")
        train = {r["sample_id"]: r for r in splits["train"]}
        inner = {r["sample_id"]: r for r in splits["fit"] + splits["head_validation"]}
        if train.keys() != inner.keys() or len(inner) != len(splits["fit"]) + len(
            splits["head_validation"]
        ):
            raise ValueError("ITI union")
        for sid, r in inner.items():
            for key in train[sid]:
                if r[key] != train[sid][key]:
                    raise ValueError(f"ITI record changed: {sid} {key}")
        report["models"][model] = summary
    if (
        sum(
            v[s]["n"]
            for v in report["models"].values()
            for s in ("train", "dev", "test")
        )
        != 4729
    ):
        raise ValueError("Total mismatch")
    report["checks"] = [
        "15 split counts and labels",
        "9 byte-identical outer copies",
        "model and binary labels",
        "task registry and membership",
        "6 kinds of outer disjointness",
        "ITI disjoint groups and unchanged union",
        "per-model manifests equal master manifest",
    ]
    report["status"] = "passed"
    report["annotation"] = "Rule A v2 model annotations, not human gold"
    report["historical_exposure"] = (
        "14 current Qwen Test records / 6 current task groups were used in old smoke; no old artifacts may be reused"
    )
    return report
