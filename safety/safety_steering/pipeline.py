"""Formal staged runner: capture → fit → Dev → freeze → Test. Resumable.

All dataset actions are inert classifier input. This program never executes them.
"""

from __future__ import annotations
import json
import os
import time
import torch
from .data import load, read, write_new
from .metrics import metrics, selection_key
from .runtime import (
    completion_ids,
    capture,
    predict_batch,
    prompt_ids,
    head_layout,
    prepare_prefix,
)
from .storage import save_tensor, read_tensor, sample_path, verify_prediction
from .learners import fit_residual, fit_iti_layer

from .config import configs
from .storage import evaluation_identity
from .provenance import model_for_run


def event(a, stage, **info):
    print(
        json.dumps(
            {
                "stage": stage,
                "model": a.model,
                "job_id": os.environ.get("SLURM_JOB_ID"),
                **info,
            }
        ),
        flush=True,
    )


def capture_train(a, provenance):
    rows, _ = load(a.root, a.model, "train")
    model, tok = model_for_run(a)
    start = time.monotonic()
    for i, row in enumerate(rows):
        path = sample_path(a.out / "cache", row["sample_id"])
        if path.exists():
            cached = read_tensor(path, provenance)
            if cached["sample_id"] != row["sample_id"]:
                raise ValueError("Capture identity")
            continue
        pair = {}
        for side, label in [("positive", row["label"]), ("negative", 1 - row["label"])]:
            ids, boundary = completion_ids(tok, row, a.model, label)
            pair[side] = capture(
                model, ids, list(range(len(model.model.layers))), heads=True
            )
            pair[side]["boundary"] = boundary
        save_tensor(
            path,
            {
                "provenance": provenance,
                "sample_id": row["sample_id"],
                "gold": row["label"],
                **pair,
            },
        )
        if i % 20 == 0:
            event(
                a,
                "capture",
                completed=i + 1,
                total=len(rows),
                elapsed=time.monotonic() - start,
            )
    done = a.out / "capture.complete.json"
    if not done.exists():
        write_new(
            done,
            {
                "provenance": provenance,
                "n": len(rows),
                "sample_ids": [r["sample_id"] for r in rows],
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "elapsed_seconds": time.monotonic() - start,
            },
        )


def read_cache(a, provenance):
    rows, _ = load(a.root, a.model, "train")
    done = read(a.out / "capture.complete.json")
    if done["provenance"] != provenance or done["n"] != len(rows):
        raise ValueError("Incomplete capture")
    if done["sample_ids"] != [r["sample_id"] for r in rows]:
        raise ValueError("Capture sample order changed")
    cached = []
    for row in rows:
        path = sample_path(a.out / "cache", row["sample_id"])
        value = read_tensor(path, provenance)
        if value["sample_id"] != row["sample_id"] or value["gold"] != row["label"]:
            raise ValueError("Capture identity changed")
        cached.append(value)
    return rows, cached


def fit_dense_and_iti(a, provenance):
    rows, cached = read_cache(a, provenance)
    labels = [r["label"] for r in rows]
    for method in [
        m for m in a.spec["methods"] if m in ("caa", "pca", "lat", "linear_probe")
    ]:
        destination = a.out / "artifacts" / f"{method}.pt"
        if destination.exists():
            read_tensor(destination, provenance)
            continue
        result = {}
        for layer in a.spec["layer_grid"][a.model]:
            checkpoint = a.out / "fit" / f"{method}-{layer}.pt"
            if checkpoint.exists():
                fitted = read_tensor(checkpoint, provenance)["fit"]
            else:
                pos = torch.stack([r["positive"]["residual"][layer] for r in cached])
                neg = torch.stack([r["negative"]["residual"][layer] for r in cached])
                fitted = fit_residual(method, pos, neg, labels, a.spec["seed"])
                save_tensor(checkpoint, {"provenance": provenance, "fit": fitted})
            result[layer] = fitted
            event(a, "fit", method=method, layer=layer)
        save_tensor(
            destination,
            {
                "provenance": provenance,
                "method": method,
                "layers": result,
                "train_sample_ids": [r["sample_id"] for r in rows],
            },
        )
    if "iti" not in a.spec["methods"]:
        return
    destination = a.out / "artifacts/iti.pt"
    if destination.exists():
        read_tensor(destination, provenance)
        return
    inner_fit, _ = load(a.root, a.model, "fit")
    inner_val, _ = load(a.root, a.model, "head_validation")
    index = {r["sample_id"]: i for i, r in enumerate(rows)}
    ii = [index[r["sample_id"]] for r in inner_fit]
    vi = [index[r["sample_id"]] for r in inner_val]
    ranking = []
    for layer in sorted(cached[0]["positive"]["heads"]):
        checkpoint = a.out / "fit" / f"iti-{layer}.pt"
        if checkpoint.exists():
            rank = read_tensor(checkpoint, provenance)["ranking"]
        else:
            pos = torch.stack([r["positive"]["heads"][layer] for r in cached])
            neg = torch.stack([r["negative"]["heads"][layer] for r in cached])
            rank = fit_iti_layer(
                pos[ii],
                neg[ii],
                [labels[i] for i in ii],
                pos[vi],
                neg[vi],
                [labels[i] for i in vi],
                layer,
            )
            save_tensor(checkpoint, {"provenance": provenance, "ranking": rank})
        ranking.extend(rank)
        event(a, "fit", method="iti", layer=layer)
    selected = sorted(ranking, key=lambda r: (-r["score"], r["layer"], r["head"]))[:48]
    if len(selected) != 48:
        raise ValueError("Insufficient valid heads")
    save_tensor(
        destination,
        {
            "provenance": provenance,
            "method": "iti",
            "selected": selected,
            "ranking": ranking,
            "fit_sample_ids": [r["sample_id"] for r in inner_fit],
            "validation_sample_ids": [r["sample_id"] for r in inner_val],
        },
    )


def deltas(model, config, artifact):
    if config["method"] in ("no_steering", "prompted"):
        return {}, {}
    if config["method"] != "iti":
        return {
            config["layer"]: config["alpha"]
            * artifact["layers"][config["layer"]]["vector"]
        }, {}
    result = {}
    layout = head_layout(model)
    for r in artifact["selected"]:
        layer = r["layer"]
        _, dim, width = layout[layer]
        if layer not in result:
            result[layer] = torch.zeros(width)
        result[layer][r["head"] * dim : (r["head"] + 1) * dim] = (
            config["alpha"] * r["sigma"] * r["direction"]
        )
    return {}, result


def evaluate(a, provenance, model, tok, split, config, artifact):
    rows, path = load(a.root, a.model, split)
    identity = evaluation_identity(
        provenance, split, config, artifact["artifact_id"] if artifact else None
    )
    folder = a.out / split / config["config_id"]
    records = []
    residual, heads = deltas(model, config, artifact)
    batches = []
    batch = []
    width = 0
    for row in rows:
        length = len(prompt_ids(tok, row, a.model, config["method"] == "prompted"))
        if batch and (len(batch) == 4 or max(width, length) * (len(batch) + 1) > 12288):
            batches.append(batch)
            batch = []
            width = 0
        batch.append(row)
        width = max(width, length)
    if batch:
        batches.append(batch)
    # Small immutable chunks avoid hundreds of thousands of NFS metadata writes.
    # A preemption can lose at most eight not-yet-published fixed batches.
    for start in range(0, len(batches), 8):
        chunk = batches[start : start + 8]
        chunk_rows = [r for batch in chunk for r in batch]
        checkpoint = folder / "prediction_chunks" / f"{start // 8:05d}.json"
        if checkpoint.exists():
            predictions = read(checkpoint)
            if len(predictions) != len(chunk_rows):
                raise ValueError("Incomplete prediction chunk")
            records.extend(
                verify_prediction(r, row, identity)
                for r, row in zip(predictions, chunk_rows)
            )
            continue
        predictions = []
        for batch in chunk:
            prepared = prepare_prefix(
                model, tok, batch, a.model, prompted=config["method"] == "prompted"
            )
            batch_predictions = predict_batch(
                model,
                tok,
                batch,
                a.model,
                prompted=config["method"] == "prompted",
                residual=residual,
                head_deltas=heads,
                prepared=prepared,
            )
            del prepared
            for record in batch_predictions:
                record["provenance"] = identity
            predictions.extend(batch_predictions)
        write_new(checkpoint, predictions)
        records.extend(predictions)
    result = {
        **config,
        "provenance": provenance,
        "evaluation_identity": identity,
        "split": split,
        "metrics": metrics(records),
        "predictions": records,
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }
    done = folder / "complete.json"
    if done.exists():
        old = read(done)
        if old["evaluation_identity"] != identity or old["predictions"] != records:
            raise ValueError("Completed evaluation changed")
        return old
    write_new(done, result)
    return result


def calibrate(a, provenance):
    model, tok = model_for_run(a)
    # All dense/ITI/SAE configs share the same unsteered causal prefix.
    rows, source = load(a.root, a.model, "dev")
    all_configs = []
    artifacts = {}
    identities = {}
    for method in a.spec["methods"]:
        path = a.out / "artifacts" / f"{method}.pt"
        artifact = None
        if method not in ("no_steering", "prompted"):
            if not path.exists():
                raise ValueError(f"Missing required artifact: {method}")
            artifact = read_tensor(path, provenance)
        artifacts[method] = artifact
        for config in configs(a, method, artifact):
            all_configs.append(config)
            identities[config["config_id"]] = evaluation_identity(
                provenance, "dev", config, artifact["artifact_id"] if artifact else None
            )
    for prompted in [False, True]:
        selected = [c for c in all_configs if (c["method"] == "prompted") == prompted]
        batches = []
        batch = []
        width = 0
        for row in rows:
            n = len(prompt_ids(tok, row, a.model, prompted))
            if batch and (len(batch) == 4 or max(width, n) * (len(batch) + 1) > 12288):
                batches.append(batch)
                batch = []
                width = 0
            batch.append(row)
            width = max(width, n)
        if batch:
            batches.append(batch)
        for start in range(0, len(batches), 8):
            if (start // 8) % a.shards != a.rank:
                continue
            chunk = batches[start : start + 8]
            chunk_rows = [r for b in chunk for r in b]
            pending = []
            for config in selected:
                checkpoint = (
                    a.out
                    / "dev"
                    / config["config_id"]
                    / "prediction_chunks"
                    / f"{start // 8:05d}.json"
                )
                if checkpoint.exists():
                    predictions = read(checkpoint)
                    if len(predictions) != len(chunk_rows):
                        raise ValueError("Wrong prediction chunk coverage")
                    for r, row in zip(predictions, chunk_rows):
                        verify_prediction(r, row, identities[config["config_id"]])
                else:
                    pending.append(config)
            collected = {c["config_id"]: [] for c in pending}
            for batch in chunk:
                if not pending:
                    break
                prepared = prepare_prefix(model, tok, batch, a.model, prompted)
                for config in pending:
                    residual, heads = deltas(model, config, artifacts[config["method"]])
                    predictions = predict_batch(
                        model,
                        tok,
                        batch,
                        a.model,
                        prompted=prompted,
                        residual=residual,
                        head_deltas=heads,
                        prepared=prepared,
                    )
                    for r in predictions:
                        r["provenance"] = identities[config["config_id"]]
                    collected[config["config_id"]].extend(predictions)
                del prepared
            for config in pending:
                write_new(
                    a.out
                    / "dev"
                    / config["config_id"]
                    / "prediction_chunks"
                    / f"{start // 8:05d}.json",
                    collected[config["config_id"]],
                )
            event(
                a,
                "dev_chunk",
                prompted=prompted,
                chunk=start // 8,
                completed_batches=min(start + 8, len(batches)),
                total_batches=len(batches),
                configs=len(selected),
            )
    if a.shards > 1:
        return
    for method, artifact in artifacts.items():
        cells = []
        for config in configs(a, method, artifact):
            cell = evaluate(a, provenance, model, tok, "dev", config, artifact)
            cells.append({k: v for k, v in cell.items() if k != "predictions"})
            event(
                a,
                "dev",
                method=method,
                config=config["config_id"],
                metrics=cell["metrics"],
            )
        path = a.out / "calibration" / f"{method}.json"
        result = {
            "provenance": provenance,
            "method": method,
            "cells": cells,
            "chosen": min(cells, key=selection_key),
        }
        if path.exists():
            if read(path) != result:
                raise ValueError("Calibration changed")
        else:
            write_new(path, result)


def freeze(a, provenance):
    result = {
        "provenance": provenance,
        "configs": {},
        "blocked": {},
        "protocol": read(a.out / "protocol.json"),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "frozen_unix_time": time.time(),
    }
    for method in a.spec["methods"]:
        calibration = a.out / "calibration" / f"{method}.json"
        if not calibration.exists():
            raise ValueError(f"Missing calibration: {method}")
        cal = read(calibration)
        if cal["provenance"] != provenance:
            raise ValueError("Wrong calibration provenance")
        artifact = a.out / "artifacts" / f"{method}.pt"
        obj = read_tensor(artifact, provenance) if artifact.exists() else None
        expected = {c["config_id"] for c in configs(a, method, obj)}
        if expected != {c["config_id"] for c in cal["cells"]} or len(expected) != len(
            cal["cells"]
        ):
            raise ValueError("Incomplete Dev grid")
        if cal["chosen"] != min(cal["cells"], key=selection_key):
            raise ValueError("Selection differs from Dev tie rule")
        from .report import verify_cell

        info = read(a.out / "preflight.json")["splits"]["dev"]
        artifact_id = obj["artifact_id"] if obj else None
        for cell in cal["cells"]:
            verified = verify_cell(a.out, "dev", cell, provenance, info, artifact_id)
            if {k: v for k, v in verified.items() if k != "predictions"} != cell:
                raise ValueError("Calibration differs from Dev predictions")
        result["configs"][method] = {
            "chosen": cal["chosen"],
            "calibration": cal,
            "artifact_id": artifact_id,
        }
    path = a.out / "frozen.json"
    if path.exists():
        old = read(path)
        if (
            old["provenance"] != provenance
            or old["configs"] != result["configs"]
            or old["blocked"] != result["blocked"]
        ):
            raise ValueError("Frozen configuration changed")
    else:
        write_new(path, result)


def test(a, provenance):
    frozen = read(a.out / "frozen.json")
    if frozen["provenance"] != provenance or frozen["protocol"] != read(
        a.out / "protocol.json"
    ):
        raise ValueError("Frozen protocol mismatch")
    if set(frozen["configs"]) != set(a.spec["methods"]) or frozen["blocked"]:
        raise ValueError("Incomplete frozen methods")
    model, tok = model_for_run(a)
    for method, saved in frozen["configs"].items():
        if read(a.out / "calibration" / f"{method}.json") != saved["calibration"]:
            raise ValueError("Calibration changed after freeze")
        artifact = None
        if saved["artifact_id"]:
            artifact = read_tensor(a.out / "artifacts" / f"{method}.pt", provenance)
            if artifact["artifact_id"] != saved["artifact_id"]:
                raise ValueError("Artifact identity changed after freeze")
        config = {
            k: saved["chosen"][k]
            for k in ["method", "layer", "alpha", "delta_norm", "config_id"]
        }
        result = evaluate(a, provenance, model, tok, "test", config, artifact)
        event(a, "test", method=method, metrics=result["metrics"])
