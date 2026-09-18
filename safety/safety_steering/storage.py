"""Immutable tensor artifacts and resumable, provenance-checked checkpoints."""

import os
from pathlib import Path
import tempfile
import torch
import base64
import uuid


def save_tensor(path, value):
    value = {**value, "artifact_id": uuid.uuid4().hex}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(value, f)
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)
    finally:
        os.unlink(tmp)


def read_tensor(path, provenance=None):
    value = torch.load(path, weights_only=True, map_location="cpu")
    if provenance is not None and value["provenance"] != provenance:
        raise ValueError(f"Stale checkpoint: {path}")
    return value


def sample_path(root, sample_id, suffix=".pt"):
    name = (
        base64.urlsafe_b64encode(sample_id.encode("utf-8")).decode("ascii").rstrip("=")
    )
    if not name or len(name + suffix) > 240:
        raise ValueError("Sample ID cannot be represented as a filename")
    return Path(root) / (name + suffix)


def verify_prediction(record, row, provenance):
    if (
        record["provenance"] != provenance
        or record["sample_id"] != row["sample_id"]
        or record["gold"] != row["label"]
        or record["task_group_id"] != row["task_group_id"]
    ):
        raise ValueError("Prediction checkpoint identity mismatch")
    return record


def evaluation_identity(provenance, split, config, artifact_id):
    return dict(run_id=provenance, split=split, config=config, artifact_id=artifact_id)
