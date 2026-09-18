"""Resumable output primitives without persisted run metadata or digest checks."""

import json
import os
from pathlib import Path


def load_pairs(directory):
    """Load the prepared extraction pairs directly from their JSONL file."""
    rows = read_jsonl(Path(directory) / "extraction.jsonl")
    if len(rows) != 64:
        raise RuntimeError(f"Expected 64 extraction pairs, found {len(rows)}")
    if len({row.get("pair_id") for row in rows}) != 64:
        raise RuntimeError("Extraction pairs must have 64 distinct pair IDs")
    return rows


def save_tensors(path, tensors):
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    save_file({key: value.detach().cpu().contiguous() for key, value in tensors.items()}, str(temporary))
    temporary.replace(path)


def save_learned_artifact(artifact, path):
    """Publish a learned artifact only after its complete save succeeds."""
    import shutil

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    artifact.save(temporary)
    temporary.replace(path)


def load_tensors(path):
    from safetensors.torch import load_file

    return load_file(str(path))


def repair_jsonl_tail(path):
    """Recover only an interrupted final JSONL write under the stage lock."""
    path = Path(path)
    if not path.exists():
        return
    raw = path.read_bytes()
    if not raw or raw.endswith(b"\n"):
        return
    boundary = raw.rfind(b"\n") + 1
    for line in raw[:boundary].splitlines():
        json.loads(line)
    tail = raw[boundary:]
    try:
        json.loads(tail)
    except (ValueError, UnicodeDecodeError):
        backup = path.with_name(path.name + ".partial.bak")
        if not backup.exists():
            with backup.open("xb") as handle:
                handle.write(tail)
                handle.flush()
                os.fsync(handle.fileno())
        with path.open("r+b") as handle:
            handle.truncate(boundary)
            handle.flush()
            os.fsync(handle.fileno())
        print(f"Resume: backed up incomplete final row to {backup.name}", flush=True)
    else:
        with path.open("ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(f"Resume: restored final newline in {path.name}", flush=True)


def read_jsonl(path, *, repair_tail=False):
    """Read JSONL while refusing complete-line corruption."""
    path = Path(path)
    if repair_tail:
        repair_jsonl_tail(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise RuntimeError(f"Incomplete last JSONL line in {path}; repair before resuming")
    return [json.loads(line) for line in raw.splitlines()]


def append_jsonl(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
