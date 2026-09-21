#!/usr/bin/env python3
"""Self-contained Safety CLI, usable from the ControlBench repository root."""

import argparse
import json
from pathlib import Path

from safety_steering.config import MODELS, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "status",
            "audit",
            "preflight",
            "validate",
            "capture",
            "fit",
            "sae",
            "eval",
            "report",
            "all",
        ],
    )
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parent / "configs/steering.json"
    )
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument(
        "--data", dest="root", type=Path, default=Path(__file__).parent / "data"
    )
    parser.add_argument(
        "--model-path", type=Path, help="Complete local Instruct/it checkpoint"
    )
    parser.add_argument(
        "--sae-dir", type=Path, default=Path(__file__).parent / "dictionaries"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--shards", type=int, default=1, help="Eval-only disjoint chunk workers"
    )
    parser.add_argument("--rank", type=int, default=0)
    a = parser.parse_args()
    a.spec = load_config(a.config, a.model)
    if (
        a.shards < 1
        or not 0 <= a.rank < a.shards
        or (a.shards != 1 and a.stage != "eval")
    ):
        parser.error("Sharding requires eval and 0 <= rank < shards")
    if a.stage == "status":
        stages = [
            "preflight.json",
            "validation.json",
            "protocol.json",
            "capture.complete.json",
            "grid",
            "summary.json",
        ]
        print(json.dumps({p: (a.out / p).exists() for p in stages}, indent=2))
        return
    from safety_steering.data import audit

    if a.stage == "audit":
        print(json.dumps(audit(a.root), indent=2))
        return
    if a.stage == "report":
        from safety_steering.report import report

        report(a.out)
        return
    if a.model_path is None:
        parser.error("--model-path is required for model and fitting stages")
    a.out.mkdir(parents=True, exist_ok=True)
    from safety_steering.provenance import invocation

    with invocation(a):
        execute(a)


def execute(a):
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(a.spec["seed"])
    from safety_steering import pipeline
    from safety_steering.provenance import preflight, protocol
    from safety_steering.validation import validate
    from safety_steering.sae import fit_sae
    from safety_steering.report import report

    if a.stage == "preflight":
        preflight(a)
        return
    if a.stage == "validate":
        validate(a)
        return
    if a.stage == "all":
        preflight(a)
        validate(a)
    provenance = protocol(a)
    stages = dict(
        capture=pipeline.capture_extraction,
        fit=pipeline.fit_dense_and_iti,
        sae=fit_sae,
        eval=pipeline.evaluate_grid,
    )
    selected = list(stages) if a.stage == "all" else [a.stage]
    for stage in selected:
        if stage == "sae" and "sae" not in a.spec["methods"]:
            continue
        if stage in ("capture", "fit") and set(a.spec["methods"]) <= {
            "no_steering",
            "prompted",
        }:
            continue
        pipeline.event(a, "stage_start", name=stage)
        stages[stage](a, provenance)
        pipeline.event(a, "stage_complete", name=stage)
    if a.stage == "all":
        report(a.out)


if __name__ == "__main__":
    main()
