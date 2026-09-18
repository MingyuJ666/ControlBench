"""Resumable baselines, zero-dose checks, and the complete positive-strength grid."""

from l2s.storage import read_json, write_json

from .artifacts import append_jsonl, read_jsonl
from .config import (
    LAYERS,
    RESIDUAL_METHODS,
    SAE_LAYERS,
    artifact_key,
    learned_cells,
    methods_for,
)
from .generation import generate_batch
from .progress import Progress
from .scoring import evaluate_output
from .settings import (
    batches,
    DEFAULT_BATCH_SIZE,
)


def row_key(row):
    return (
        row["eval_set"],
        row["problem_id"],
        row["method"],
        row["layer"],
        row["strength"],
    )


def intervention(root, name, method, layer, strength, model):
    import repsteer as rs

    artifact = rs.load_artifact(root / "artifacts" / name / artifact_key(method, layer))
    artifact.bind(model)
    if method == "ITI":
        return rs.recipes.iti(
            artifact,
            strength=strength,
            phase="decode",
            positions=rs.positions.GeneratedTokens(),
        )
    return rs.Intervention(
        artifact=artifact,
        operator=rs.operators.Add(),
        positions=rs.positions.GeneratedTokens(),
        strength=rs.schedules.Constant(strength),
        phase="decode",
    )


def zero_check(root, name, model, run_config, items):
    track = run_config["track"]
    path = root / "artifacts" / name / "ZERO_DOSE.json"
    if path.exists():
        result = read_json(path)
        if result.get("passed"):
            print(
                "Zero-dose checks already passed.",
                flush=True,
            )
            return
        raise RuntimeError("Previous zero-dose check failed")
    selected = [item for item in items if item["eval_set"] == "math500"][:10]
    if len(selected) != 10:
        raise RuntimeError("Zero-dose identity requires 10 MATH-500 problems")
    checks = []
    # One representative site per operator/artifact method, on ten problems.
    representatives = [(method, LAYERS[name][0]) for method in RESIDUAL_METHODS]
    representatives.append(("ITI", None))
    representatives += [("SAE", SAE_LAYERS[name][0])] if SAE_LAYERS[name] else []
    representatives = [
        (method, layer)
        for method, layer in representatives
        if method in methods_for(name, track)
    ]
    max_new_tokens = run_config["decoding"]["max_new_tokens"]
    groups = list(batches(selected, DEFAULT_BATCH_SIZE))
    baseline = {}
    for group in groups:
        outputs = generate_batch(
            model,
            name,
            [item["problem"] for item in group],
            max_new_tokens=max_new_tokens,
        )
        if len(outputs) != len(group):
            raise RuntimeError("Generation returned an unexpected batch size")
        baseline.update(
            (item["problem_id"], output) for item, output in zip(group, outputs)
        )
    for method, layer in representatives:
        control = intervention(root, name, method, layer, 0.0, model)
        for group in groups:
            outputs = generate_batch(
                model,
                name,
                [item["problem"] for item in group],
                intervention=control,
                max_new_tokens=max_new_tokens,
            )
            if len(outputs) != len(group):
                raise RuntimeError("Generation returned an unexpected batch size")
            for item, output in zip(group, outputs):
                reference = baseline[item["problem_id"]]
                passed = (
                    output["token_ids"] == reference["token_ids"]
                    and output["truncated"] == reference["truncated"]
                )
                checks.append(
                    dict(
                        method=method,
                        layer=layer,
                        problem_id=item["problem_id"],
                        passed=passed,
                    )
                )
                if not passed:
                    write_json(
                        path, dict(passed=False, checks=checks)
                    )
                    raise RuntimeError(
                        "Zero-dose token identity failed; grid is blocked"
                    )
        print(f"Zero dose: {method}, 10/10 identical.", flush=True)
    write_json(
        path,
        dict(
            passed=True,
            checks=checks,
            track=track,
            batch_size=DEFAULT_BATCH_SIZE,
            scope="one representative site per selected method; complete profile for ITI; ten MATH-500 items",
        ),
    )


def run_cells(root, name, model, run_config, items, cells, *, shard=None):
    max_new_tokens = run_config["decoding"]["max_new_tokens"]
    selected = run_config["eval_sets"]
    if any(item["eval_set"] not in selected for item in items):
        raise RuntimeError("Input items contain an unconfigured evaluation dataset")
    by_method = {}
    for method, layer, strength in cells:
        by_method.setdefault(method, []).append((layer, strength))
    prepared = []
    for method, settings in by_method.items():
        directory = root / "runs" / name / method
        directory.mkdir(parents=True, exist_ok=True)
        suffix = "grid.jsonl" if shard is None else f"grid-{shard}.jsonl"
        path = directory / suffix
        rows = read_jsonl(path, repair_tail=True)
        expected = {
            (item["eval_set"], item["problem_id"], method, layer, strength)
            for item in items
            for layer, strength in settings
        }
        seen = set()
        for row in rows:
            key = row_key(row)
            if (
                key in seen
                or key not in expected
            ):
                raise RuntimeError(
                    f"Duplicate, out-of-grid, or incompatible row in {path}"
                )
            seen.add(key)
        prepared.append((method, settings, directory, path, expected, seen))
    progress = Progress(
        sum(len(expected) for _, _, _, _, expected, _ in prepared),
        sum(len(seen) for _, _, _, _, _, seen in prepared),
    )
    for method, settings, directory, path, expected, seen in prepared:
        with path.open("a") as handle:
            for layer, strength in settings:
                control = (
                    None
                    if method in ("No-Steering", "System-Prompt")
                    else intervention(root, name, method, layer, strength, model)
                )
                completed = sum(key[2:] == (method, layer, strength) for key in seen)
                for group in batches(items, DEFAULT_BATCH_SIZE):
                    keys = [
                        (item["eval_set"], item["problem_id"], method, layer, strength)
                        for item in group
                    ]
                    if all(key in seen for key in keys):
                        continue
                    # Preserve the original batch members even when some rows
                    # were saved before an interruption: padding/shapes stay fixed.
                    outputs = generate_batch(
                        model,
                        name,
                        [item["problem"] for item in group],
                        [item["eval_set"] for item in group],
                        system=method == "System-Prompt",
                        intervention=control,
                        max_new_tokens=max_new_tokens,
                    )
                    if len(outputs) != len(group):
                        raise RuntimeError(
                            "Generation returned an unexpected batch size"
                        )
                    for item, key, output in zip(group, keys, outputs):
                        if key in seen:
                            continue
                        row = dict(
                            model=name,
                            method=method,
                            layer=layer,
                            strength=strength,
                            problem_id=item["problem_id"],
                            eval_set=item["eval_set"],
                            difficulty=item["difficulty"],
                            has_figure=item["has_figure"],
                            gold_answer=item["gold_answer"],
                            text=output["text"],
                            truncated=output["truncated"],
                            **evaluate_output(output, item, model.tokenizer),
                        )
                        append_jsonl(handle, row)
                        seen.add(key)
                        completed += 1
                        progress.advance(
                            f"{name} {method} layer={layer} strength={strength}",
                            detail=f"cell={completed}/{len(items)} ({100 * completed / len(items):.1f}%)",
                            force=completed == len(items),
                        )
        write_json(
            directory
            / ("COMPLETE.json" if shard is None else f"COMPLETE-{shard}.json"),
            dict(
                rows=len(seen),
                expected_rows=len(expected),
                complete=seen == expected,
            ),
        )


def baselines(root, name, model, run_config, items):
    run_cells(
        root,
        name,
        model,
        run_config,
        items,
        [("No-Steering", None, None), ("System-Prompt", None, None)],
    )
    rows = read_jsonl(root / "runs" / name / "No-Steering" / "grid.jsonl")
    rates = {}
    for dataset in run_config["eval_sets"]:
        subset = [row for row in rows if row["eval_set"] == dataset]
        if not subset:
            raise RuntimeError(f"No baseline rows for configured dataset: {dataset}")
        rates[dataset] = sum(row["truncated"] for row in subset) / len(subset)
    write_json(
        root / "runs" / name / "BASELINE_CHECK.json",
        dict(
            eval_sets=run_config["eval_sets"],
            truncation_rates=rates,
            max_new_tokens=run_config["decoding"]["max_new_tokens"],
            exceeds_five_percent={
                dataset: rate > 0.05 for dataset, rate in rates.items()
            },
        ),
    )
    for dataset, rate in rates.items():
        if rate > 0.05:
            print(
                f"{dataset} baseline truncation exceeds 5%; report collapse alongside accuracy and length. The token cap remains fixed.",
                flush=True,
            )


def grid(
    root,
    name,
    model,
    run_config,
    items,
    methods=None,
    layers=None,
    strengths=None,
    shard=None,
):
    complete = read_json(root / "artifacts" / name / "FIT_COMPLETE.json")
    if not complete.get("complete"):
        raise RuntimeError("Fit did not complete")
    if run_config.get("zero_check", False):
        zero_check(root, name, model, run_config, items)
    cells = [
        cell
        for cell in learned_cells(name, run_config["track"])
        if (not methods or cell[0] in methods)
        and (layers is None or cell[1] is None or cell[1] in layers)
        and (strengths is None or cell[2] in strengths)
    ]
    if not cells:
        raise RuntimeError("No configured grid cells match the requested shard")
    run_cells(root, name, model, run_config, items, cells, shard=shard)
