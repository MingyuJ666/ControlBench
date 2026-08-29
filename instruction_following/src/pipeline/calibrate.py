"""Choose a steering configuration on the public dev split.

Calibration is deliberately split around the semantic judge, and it is baseline
agnostic: the selected baseline declares a search plan (see ``baselines.shared.base``)
and this module executes it.

1. generate the plan's first stage;
2. judge those outputs and select its winner by full benchmark success;
3. generate any stage the plan declares after that, carrying the winning
   axes forward;
4. judge the new outputs and report the best dev configuration.

Every baseline currently declares one joint stage — three sites by three
strengths — so steps 3 and 4 usually do nothing. The runner reads the stages
from the plan rather than assuming two, because a plan that chooses one axis
before another has to be generated in that order.

The search runs on a fixed 150-prompt subset of dev (see
``kernel.splits.calibration_items``): its job is to rank nine configurations,
not to estimate an accuracy, and the published number comes from test. Grids
are a reference default, not a benchmark restriction: participants can supply
their own with ``--grid axis=v1,v2``, named for the axes the baseline
declares. Test evaluation consumes the resulting fixed configuration and must
not tune it again on test.
"""

# Run directly as a script as well as imported: the package root
# is this file's ancestor, not its own directory.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import baselines
from kernel.config import TARGET_MODEL, calibration_dir, slug
from kernel.judge import write_scored_records
from kernel.reference_conditions import load_reference_summary
from kernel.scoring import (build_records, merge_judgments, prepare_rule_passers,
                     summarize_merged, summarize_rules)
from baselines.shared.positions import DEFAULT as POSITION_MODE
from kernel.splits import (CALIBRATION_ITEMS, CALIBRATION_SEED,
                           calibration_items)
from kernel.tasks import TASKS
from pipeline.evaluate import code_version


N_JUDGE_BATCHES = 4


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _reference_summaries(model_slug: str, task: str) -> dict[str, dict]:
    """Return optional dev references without making calibration depend on them."""
    references = {}
    for condition in ("unsteered", "prompted"):
        summary = load_reference_summary(
            model_slug, "dev", task, condition, required=False
        )
        if summary is not None:
            references[condition] = summary
    return references


def parse_grid(entries: list[str] | None) -> dict:
    """Read ``axis=v1,v2,...`` entries into the grid a baseline understands.

    Axes are the baseline's own — layer and alpha for the residual-stream
    ones, heads_k and alpha for ITI — so a grid names the axis it varies
    instead of relying on a flag per axis, which silently did nothing for a
    baseline that had no such axis.
    """
    grid = {}
    for entry in entries or []:
        axis, _, values = entry.partition("=")
        axis, values = axis.strip(), values.strip()
        if not axis or not values:
            raise ValueError(f"a grid looks like axis=v1,v2; got {entry!r}")
        parsed = []
        for value in values.split(","):
            value = value.strip()
            parsed.append(
                float(value) if any(m in value for m in ".eE") else int(value)
            )
        if axis in grid:
            raise ValueError(f"axis {axis!r} given twice")
        grid[axis] = parsed
    return grid


def _search_overrides(args) -> dict:
    """The grids the caller asked for, if any."""
    return parse_grid(args.grid)


def run_config(generator, baseline, artifact, task: str, items: list[dict],
               config: dict, out_dir: Path) -> dict:
    """Generate and rule-score one configuration atomically."""
    name = baseline.config_name(config)
    outputs = generator.generate(
        [row["base_prompt"] for row in items],
        intervention=baseline.make_intervention(generator.model, artifact, config),
        progress=False,
    )
    records = build_records(
        task, "dev", items, outputs, name,
        {"method": baseline.KEY, "config": name, **config},
    )
    if not records:
        raise ValueError(f"No dev items were provided for task {task!r}")
    rules = summarize_rules(records)
    if not rules["n_rule_scored"]:
        raise RuntimeError(
            f"All outputs for {name} were excluded from rule scoring"
        )
    _write_jsonl(out_dir / f"{name}.jsonl", records)
    return {
        "config": name,
        **config,
        "values": dict(config),
        "n": len(records),
        "n_rule_scored": rules["n_rule_scored"],
        "rule_rate": rules["rule_pass_rate"],
        "truncation_rate": (
            sum(bool(row["truncated"]) for row in records) / len(records)
        ),
        # What this configuration actually added. alpha=2.0 is a different
        # physical quantity in each baseline — raw magnitude for caa, sigma
        # units for probe, a feature's peak activation for sae, and K times a
        # per-head sigma for iti — so a table of alphas is not a table of
        # doses, and the comparison needs this recorded beside the score.
        "dose": baseline.dose_metadata(artifact, config),
    }


def _load_config_records(out_dir: Path,
                         config_names: list[str]) -> list[dict]:
    records = []
    for name in config_names:
        path = out_dir / f"{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        rows = _read_jsonl(path)
        for row in rows:
            if row.get("config") != name:
                raise ValueError(
                    f"Record in {path} has config {row.get('config')!r}, "
                    f"expected {name!r}"
                )
        records.extend(rows)
    return records


def _prepare_judging(out_dir: Path, config_names: list[str], stage: str) -> int:
    """Prepare only the declared configs; stale JSONL files are never globbed."""
    records = _load_config_records(out_dir, config_names)
    n_passers = prepare_rule_passers(
        records, out_dir / "judging" / stage,
        n_batches=N_JUDGE_BATCHES, seed=11,
    )
    print(
        f"judge stage {stage}: {n_passers} rule-pass outputs "
        f"from {len(records)} total outputs"
    )
    return n_passers


def aggregate(records: list[dict], axes: list[str]) -> list[dict]:
    """Aggregate records after ``merge`` has attached full success decisions.

    ``axes`` names the baseline's configuration axes, so this stays
    baseline agnostic: it reports whatever the baseline varied, without knowing what a
    layer or a head count means.
    """
    grouped = {}
    for row in records:
        grouped.setdefault(row["config"], []).append(row)

    rows = []
    for name, config_records in grouped.items():
        summary = summarize_merged(config_records)
        first = config_records[0]
        values = {axis: first[axis] for axis in axes}
        rows.append({
            "config": name,
            **values,
            "values": values,
            "rule_rate": summary["rule_pass_rate"],
            "accuracy": summary["accuracy"],
            "incoherent_rate": summary["incoherent_rate"],
            "not_answering_rate": summary["not_answering_rate"],
            "degenerate_rate": summary["degenerate_rate"],
            "judge_suspect": summary["judge_suspect"],
            "n": summary["n_rule_scored"],
            "n_total": summary["n_total"],
            "n_truncated": summary["n_truncated"],
        })
    rows.sort(key=lambda row: tuple(row[axis] for axis in axes))
    return rows


def _tie_break_axes(record: dict) -> dict:
    """Name the axes the tie-break orders by, as the plan declared them."""
    metadata = record.get("plan", {}).get("metadata", {})
    axes = {}
    if "strength_axis" in metadata:
        axes["strength_axis"] = metadata["strength_axis"]
    if "structural_axis" in metadata:
        axes["structural_axis"] = metadata["structural_axis"]
    return axes


def choose_best(rows: list[dict], strength_axis: str = "alpha",
                structural_axis: str = "layer") -> dict:
    """Choose full accuracy; break ties by weaker strength, then lower site.

    The tie-break keeps the least invasive configuration among equals, which
    matters because strong settings often reach the same success by degrading
    the model in ways the rule cannot see.
    """
    if not rows:
        raise ValueError("No calibration configurations were scored")
    return max(
        rows,
        key=lambda row: (
            row["accuracy"],
            -abs(row.get(strength_axis, 0)),
            -row.get(structural_axis, 0),
        ),
    )


def _merge_complete(records: list[dict], judging_dirs: list[Path]) -> None:
    diagnostics = merge_judgments(records, judging_dirs)
    if diagnostics["n_unjudged"]:
        locations = ", ".join(str(path) for path in judging_dirs)
        raise SystemExit(
            f"{diagnostics['n_unjudged']} rule-pass outputs have no judge "
            f"verdict in {locations}; complete judging before continuing"
        )


def _owned_artifacts(out_dir: Path) -> list[Path]:
    names = ("search.json", "summary.json",
             "scored_outputs.jsonl", "judge_records.json")
    owned = [out_dir / name for name in names if (out_dir / name).exists()]
    # Per-configuration outputs are named by the baseline, so they are found by
    # exclusion rather than by a CAA-shaped glob like "L*_a*.jsonl".
    owned += [
        path for path in out_dir.glob("*.jsonl")
        if path.name not in names
    ]
    if (out_dir / "judging").exists():
        owned.append(out_dir / "judging")
    return owned


def _reset(out_dir: Path) -> None:
    for path in _owned_artifacts(out_dir):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def generate(args) -> None:
    """Generate stage 1 and prepare its judge inputs."""
    if args.batch < 1:
        raise ValueError(f"batch must be positive, got {args.batch}")
    if args.limit is not None and args.limit < 1:
        raise ValueError(f"limit must be positive, got {args.limit}")

    model_slug = slug(args.model)
    baseline = baselines.get(args.baseline)
    out_dir = calibration_dir(model_slug, baseline.KEY, args.task)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = _owned_artifacts(out_dir)
    if args.force:
        _reset(out_dir)
    elif existing:
        raise FileExistsError(
            f"Calibration artifacts already exist (for example {existing[0]}); "
            "pass --force to start this dev search again"
        )

    from kernel.generate import GenConfig, Generator

    items = calibration_items(args.task)
    if args.limit is not None:
        items = items[:args.limit]
    if not items:
        raise ValueError(f"No dev items found for task {args.task!r}")
    artifact = baseline.load_artifact(model_slug, args.task)
    generator = Generator(GenConfig(model=args.model, batch_size=args.batch))
    n_layers = generator.n_layers
    # What a baseline may need to build its plan: how deep the model is, how
    # many heads it has, which model and task it is planning for — SAE reads
    # its gate report to know which layers are admissible — and the artifact
    # itself, so a plan can refuse to search a site its own artifact left
    # empty rather than scoring an intervention that adds nothing.
    context = {
        "n_layers": n_layers,
        "n_heads_total": n_layers * int(getattr(
            getattr(generator.model, "config", None), "num_attention_heads", 0
        )),
        "model_slug": model_slug,
        "task": args.task,
        "artifact": artifact,
    }
    plan = baseline.search_plan(
        context, _search_overrides(args), args.budget
    )
    first_stage = plan.stages[0]

    print(
        f"{baseline.KEY} {first_stage.name}: {len(first_stage.configs)} configs, "
        f"items={len(items)}"
    )
    generation = []
    for config in first_stage.configs:
        result = run_config(
            generator, baseline, artifact, args.task, items, config, out_dir
        )
        generation.append(result)
        print(f"  {result['config']:12s} rule_pass={result['rule_rate']:.1%}")

    config_names = [row["config"] for row in generation]
    n_judged = _prepare_judging(out_dir, config_names, first_stage.name)
    _write_json(out_dir / "search.json", {
        "schema_version": 2,
        "model": args.model,
        "model_slug": model_slug,
        "task": args.task,
        "method": baseline.KEY,
        "method_version": getattr(baseline, "METHOD_VERSION", 1),
        "split": "dev",
        "intervention_positions": POSITION_MODE,
        "code_version": code_version(),
        # The generation budget belongs in the record for the same reason the
        # position rule does: two searches run under different budgets are not
        # comparable, and without this nothing on disk says which one this was.
        "generation": {
            "max_new_tokens": generator.cfg.max_new_tokens,
            "do_sample": False,
            "dtype": str(generator.cfg.dtype),
            "batch_size": generator.cfg.batch_size,
        },
        "search_strategy": "judge_guided_declared_plan",
        "plan": plan.as_record(),
        "axes": list(first_stage.configs[0]),
        "n_layers": n_layers,
        "grid_source": {
            axis: ("external" if axis in _search_overrides(args) else "default")
            for axis in first_stage.configs[0]
        },
        "planned_n_configs": plan.n_configs,
        "limit": args.limit,
        "n_items": len(items),
        "dev_subset": {"size": CALIBRATION_ITEMS, "seed": CALIBRATION_SEED},
        "config_names": config_names,
        "generation_results": generation,
        "n_rule_pass_sent_to_judge": n_judged,
        "references": _reference_summaries(model_slug, args.task),
    })
    print(f"next: judge_local.py {out_dir / 'judging' / first_stage.name}")
    print(f"then: calibrate.py {args.task} --baseline {baseline.KEY} --report")


def _print_references(references: dict[str, dict]) -> None:
    if not references:
        print("dev references: unavailable (calibration remains valid)")
        return
    values = []
    for condition in ("unsteered", "prompted"):
        summary = references.get(condition)
        if summary is not None and summary.get("rule_pass_rate") is not None:
            values.append(f"{condition} rule_pass={summary['rule_pass_rate']:.1%}")
    if values:
        print("dev references: " + ", ".join(values))


def report(args) -> None:
    model_slug = slug(args.model)
    baseline = baselines.get(args.baseline)
    out_dir = calibration_dir(model_slug, baseline.KEY, args.task)
    sweep = json.loads((out_dir / "search.json").read_text(encoding="utf-8"))
    axes = sweep.get("axes", ["layer", "alpha"])
    stage_names = [stage["name"] for stage in sweep["plan"]["stages"]]
    records = _load_config_records(out_dir, sweep["config_names"])
    _merge_complete(records, [
        out_dir / "judging" / name for name in stage_names
    ])
    scored_path = write_scored_records(
        records, out_dir / "scored_outputs.jsonl"
    )
    rows = aggregate(records, axes)
    # The dose was measured when each configuration was generated; carry it
    # onto the scored row so the chosen configuration is reported with what
    # it actually added, not only with its alpha.
    doses = {
        row["config"]: row.get("dose")
        for row in sweep.get("generation_results", [])
    }
    for row in rows:
        row["dose"] = doses.get(row["config"])
    best = choose_best(rows, **_tie_break_axes(sweep))

    print(
        f"{args.task}: n={sweep['n_items']}/config; "
        "success = rule_pass AND coherent AND answers_question "
        "AND NOT degenerate"
    )
    _print_references(sweep.get("references", {}))
    print(
        f"\n{'config':12s} {'rule':>7s} {'ACCURACY':>9s} "
        f"{'inflation':>10s} {'incoherent':>11s} {'not-answering':>13s} "
        f"{'degenerate':>11s}"
    )
    for row in rows:
        mark = "  <- best" if row["config"] == best["config"] else ""
        print(
            f"{row['config']:12s} {row['rule_rate']:>6.1%} "
            f"{row['accuracy']:>9.1%} "
            f"{row['rule_rate'] - row['accuracy']:>10.1%} "
            f"{row['incoherent_rate']:>11.1%} "
            f"{row['not_answering_rate']:>13.1%} "
            f"{row['degenerate_rate']:>11.1%}{mark}"
        )
    suspect = sum(row["judge_suspect"] for row in rows)
    if suspect:
        print(
            f"\n! judge called {suspect} repetition-loop outputs coherent; "
            "the degeneracy rule blocked them from counting as successes"
        )
    chosen = " ".join(f"{axis}={best[axis]}" for axis in axes)
    print(f"\nchosen: {chosen} accuracy={best['accuracy']:.1%}")

    summary = {
        key: sweep[key]
        for key in (
            "schema_version", "model", "task", "method", "method_version",
            "split", "search_strategy", "plan", "axes", "grid_source",
            "n_layers", "n_items", "n_configs", "references",
            "intervention_positions", "code_version", "generation",
        )
        if key in sweep
    }
    summary.update({
        "configs": rows,
        "chosen": best,
        "selection_rule": (
            "highest rule_pass AND coherent AND answers_question accuracy; "
            "ties use smaller absolute alpha, then lower layer"
        ),
        "scored_outputs": scored_path.name,
    })
    _write_json(out_dir / "summary.json", summary)
    print(f"-> {scored_path}")
    print(f"-> {out_dir / 'summary.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=list(TASKS))
    parser.add_argument("--model", default=TARGET_MODEL)
    parser.add_argument(
        "--baseline", default=baselines.DEFAULT_BASELINE, choices=baselines.keys()
    )
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--limit", type=int, help="pilot item cap")
    parser.add_argument(
        "--grid", action="append", metavar="AXIS=V1,V2",
        help="external grid for one of the baseline's axes, repeatable: "
             "layer=9,21,33 or heads_k=16,48,128 or alpha=1,2,4"
    )
    parser.add_argument(
        "--budget", type=int, default=None,
        help="maximum dev configurations this search may evaluate"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="merge both judge stages and save the chosen dev config"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="replace an existing search (valid with the layer stage)"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.report:
        report(arguments)
    else:
        generate(arguments)


if __name__ == "__main__":
    main()
