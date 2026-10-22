"""Reproducible, scenario-grouped partitions of the frozen BFCL conversations."""

import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

SEED = 20261022
GROUPING = ("Keep all BFCL category variants of a scenario together, and merge "
            "scenarios sharing a whitespace-normalized initial user request.")


def scenario_groups(rows: list[dict]) -> list[list[str]]:
    """Connect category variants and duplicate initial requests before splitting."""
    parents = {row["item_id"]: row["item_id"] for row in rows}
    if len(parents) != len(rows):
        raise ValueError("duplicate multi-turn item_id")

    def root(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    by_scenario, by_request = {}, {}
    for row in rows:
        item_id = row["item_id"]
        match = re.fullmatch(r"multi_turn_(base|miss_param|miss_func)_(\d+)", item_id)
        if match is None:
            raise ValueError(f"unexpected BFCL item_id: {item_id!r}")
        request = " ".join(" ".join(message["content"] for message in row["entry"]["question"][0]
                                   if message["role"] == "user").split())
        if not request:
            raise ValueError(f"{item_id}: missing initial user request")
        for mapping, key in ((by_scenario, match[2]), (by_request, request)):
            other = mapping.setdefault(key, item_id)
            parents[root(item_id)] = root(other)
    groups = defaultdict(list)
    for item_id in sorted(parents):
        groups[root(item_id)].append(item_id)
    return sorted(groups.values())


def build_split(rows: list[dict], source_sha256: str, seed: int = SEED) -> dict:
    """Assign whole balanced scenario groups to equal dev and test partitions."""
    groups = scenario_groups(rows)
    by_id = {row["item_id"]: row for row in rows}
    for group in groups:
        counts = [sum(by_id[key]["target"] == target for key in group)
                  for target in ("call", "clarify", "abstain")]
        if not counts[0] or len(set(counts)) != 1:
            raise ValueError("each scenario group must balance the three targets")
    random.Random(seed).shuffle(groups)
    half = len(rows) // 2
    choices = {0: []}
    for index, group in enumerate(groups):
        for size, selected in list(choices.items()):
            new_size = size + len(group)
            if new_size <= half and new_size not in choices:
                choices[new_size] = [*selected, index]
        if half in choices:
            break
    if len(rows) % 2 or half not in choices:
        raise ValueError("cannot divide whole scenario groups into equal halves")
    dev = {key for index in choices[half] for key in groups[index]}
    return {"schema_version": 1, "seed": seed, "grouping": GROUPING,
            "source_sha256": source_sha256,
            "dev": [row["item_id"] for row in rows if row["item_id"] in dev],
            "test": [row["item_id"] for row in rows if row["item_id"] not in dev]}


def load_split(rows: list[dict], path: Path, source: Path) -> dict[str, set[str]]:
    """Reject stale, incomplete, overlapping, or scenario-leaking partitions."""
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported split schema")
    if record.get("source_sha256") != hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError(f"{path}: split was built for different multi-turn data")
    partitions = {}
    for split in ("dev", "test"):
        ids = record[split]
        if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
            raise ValueError(f"{path}: {split} must contain unique item IDs")
        partitions[split] = set(ids)
    dev, test = partitions["dev"], partitions["test"]
    if dev & test or dev | test != {row["item_id"] for row in rows}:
        raise ValueError(f"{path}: dev/test must partition every multi-turn item exactly once")
    for group in scenario_groups(rows):
        if dev.intersection(group) and test.intersection(group):
            raise ValueError(f"{path}: related scenarios occur in both dev and test")
    return partitions
