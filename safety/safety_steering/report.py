"""Verify the complete Eval grid and export ACC / unsafe-F1."""

import csv
import io
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

from .data import read, write_new
from .metrics import metrics, parse_label
from .config import configs
from .storage import evaluation_identity, read_tensor


def verify_cell(folder, split, config, provenance, info, artifact_id):
    value = read(folder / split / config["config_id"] / "complete.json")
    for key in ("method", "layer", "alpha", "delta_norm", "config_id"):
        if value[key] != config[key]:
            raise ValueError("Evaluation configuration changed")
    identity = evaluation_identity(
        provenance,
        split,
        {k: config[k] for k in ("method", "layer", "alpha", "delta_norm", "config_id")},
        artifact_id,
    )
    if (value["provenance"], value["evaluation_identity"], value["split"]) != (
        provenance,
        identity,
        split,
    ):
        raise ValueError("Evaluation identity changed")
    predictions = value["predictions"]
    if len(predictions) != info["n"]:
        raise ValueError("Incomplete prediction set")
    if [r["sample_id"] for r in predictions] != info["sample_ids"] or [
        r["gold"] for r in predictions
    ] != info["gold"]:
        raise ValueError("Prediction sample order or gold labels changed")
    for record in predictions:
        if (
            record["provenance"] != identity
            or record["prediction"] != parse_label(record["raw"])
            or record["invalid"] != (record["prediction"] is None)
        ):
            raise ValueError("Prediction parser or provenance mismatch")
    if metrics(predictions) != value["metrics"]:
        raise ValueError("Saved metrics differ from predictions")
    return value


def publish_text(path, text):
    if path.exists():
        if path.read_text() != text:
            raise ValueError("Report output changed")
        return
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".partial-")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def report(folder):
    folder = Path(folder)
    protocol = read(folder / "protocol.json")
    provenance = protocol["run_id"]
    preflight = read(folder / "preflight.json")
    if preflight != protocol["preflight"]:
        raise ValueError("Preflight changed")
    rows = []
    for method in protocol["inputs"]["spec"]["methods"]:
        grid = read(folder / "grid" / f"{method}.json")
        if grid["provenance"] != provenance or grid["method"] != method:
            raise ValueError("Evaluation grid identity changed")
        artifact_path = folder / "artifacts" / f"{method}.pt"
        artifact = read_tensor(artifact_path, provenance) if artifact_path.exists() else None
        artifact_id = artifact["artifact_id"] if artifact else None
        if grid["artifact_id"] != artifact_id:
            raise ValueError("Artifact identity changed")
        expected = configs(
            SimpleNamespace(
                model=protocol["inputs"]["model_key"],
                spec=protocol["inputs"]["spec"],
            ),
            method,
            artifact,
        )
        if [cell["config_id"] for cell in grid["cells"]] != [
            cell["config_id"] for cell in expected
        ]:
            raise ValueError("Incomplete or reordered Eval grid")
        for cell in grid["cells"]:
            info = preflight["splits"]["eval"]
            verified = verify_cell(
                folder,
                "eval",
                cell,
                provenance,
                info,
                artifact_id,
            )
            if {k: v for k, v in verified.items() if k != "predictions"} != cell:
                raise ValueError("Grid summary differs from completed Eval cell")
            rows.append(
                dict(
                    model=protocol["inputs"]["model_key"],
                    method=method,
                    layer=verified["layer"],
                    alpha=verified["alpha"],
                    config_id=verified["config_id"],
                    **verified["metrics"],
                )
            )
    result = dict(provenance=provenance, rows=rows)
    output = folder / "summary.json"
    if output.exists():
        if read(output) != result:
            raise ValueError("Summary changed")
    else:
        write_new(output, result)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    publish_text(folder / "summary.csv", buffer.getvalue())
    lines = [
        "| Model | Method | Layer | Alpha | ACC | unsafe-F1 | Invalid |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['method']} | {row['layer']} | {row['alpha']:g} | {row['accuracy']:.4%} | {row['unsafe_f1']:.4%} | {row['invalid']} |"
        )
    publish_text(folder / "summary.md", "\n".join(lines) + "\n")
    print("\n".join(lines))
