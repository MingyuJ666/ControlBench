"""Verify frozen evaluations and export ACC / unsafe-F1 without model inference."""

import csv
import io
import os
import tempfile
from pathlib import Path

from .data import read, write_new
from .metrics import metrics, parse_label, selection_key
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
    frozen = read(folder / "frozen.json")
    if frozen["provenance"] != provenance or frozen["protocol"] != protocol:
        raise ValueError("Frozen protocol changed")
    preflight = read(folder / "preflight.json")
    if preflight != protocol["preflight"]:
        raise ValueError("Preflight changed")
    if (
        set(frozen["configs"]) != set(protocol["inputs"]["spec"]["methods"])
        or frozen["blocked"]
    ):
        raise ValueError("Required methods are incomplete")
    rows = []
    for method, record in frozen["configs"].items():
        cal_path = folder / "calibration" / f"{method}.json"
        cal = read(cal_path)
        if cal != record["calibration"]:
            raise ValueError("Calibration changed")
        artifact_id = record["artifact_id"]
        if (
            artifact_id
            and read_tensor(folder / "artifacts" / f"{method}.pt", provenance)[
                "artifact_id"
            ]
            != artifact_id
        ):
            raise ValueError("Artifact identity changed")
        if (
            cal["provenance"] != provenance
            or cal["chosen"] != record["chosen"]
            or cal["chosen"] != min(cal["cells"], key=selection_key)
        ):
            raise ValueError("Frozen choice differs from Dev selection")
        for cell in cal["cells"]:
            info = preflight["splits"]["dev"]
            verified = verify_cell(
                folder,
                "dev",
                cell,
                provenance,
                info,
                artifact_id,
            )
            if {k: v for k, v in verified.items() if k != "predictions"} != cell:
                raise ValueError("Calibration differs from completed Dev evaluation")
        info = preflight["splits"]["test"]
        test = verify_cell(
            folder,
            "test",
            record["chosen"],
            provenance,
            info,
            artifact_id,
        )
        rows.append(
            dict(
                model=protocol["inputs"]["model_key"],
                method=method,
                layer=test["layer"],
                alpha=test["alpha"],
                **test["metrics"],
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
