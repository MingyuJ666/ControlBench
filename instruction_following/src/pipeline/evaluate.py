"""Run one frozen CAA configuration on test and report benchmark accuracy.

Dev and test use the same scoring contract from ``scoring.py``.  Their policy
differs: calibration may compare many configs on public dev, while this module
runs exactly one config on the complete test split and never selects from test.
"""

# Run directly as a script as well as imported: the package root
# is this file's ancestor, not its own directory.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

from kernel.config import REPO, TARGET_MODEL, calibration_dir, slug, test_dir

import baselines
from baselines.shared.positions import DEFAULT as POSITION_MODE
from kernel.judge import write_scored_records
from kernel.reference_conditions import load_reference_summary
from kernel.scoring import (build_records, merge_judgments, prepare_rule_passers,
                     summarize_merged, summarize_rules)
from kernel.splits import eval_items
from kernel.tasks import TASKS


N_BOOT = 1000
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


def parse_config(text: str | None) -> dict | None:
    """Read ``axis=value,axis=value`` into the dict a baseline understands.

    Axes differ by baseline — layer and alpha for the residual-stream ones,
    heads_k and alpha for ITI — so the frozen configuration is named rather
    than positional. Values become numbers, which is what every axis so far
    is; a baseline needing something else would parse its own.
    """
    if text is None:
        return None
    config = {}
    for part in text.split(","):
        if "=" not in part:
            raise ValueError(
                f"config entries look like axis=value; got {part!r}"
            )
        axis, _, value = part.partition("=")
        axis, value = axis.strip(), value.strip()
        if not axis or not value:
            raise ValueError(f"config entry {part!r} is missing a side")
        # "48" is a count, "15.0" is a strength: the written form decides,
        # not the value, so a configuration keeps the name calibration gave it.
        config[axis] = (
            float(value) if any(mark in value for mark in ".eE") else int(value)
        )
    if not config:
        raise ValueError("an external config needs at least one axis")
    return config


def config_subdir(baseline, config: dict | None) -> str:
    """One run directory per frozen configuration, named by the baseline."""
    if config is None:
        return "calibrated"
    return f"manual_{baseline.config_name(config)}"


def resolve_config(task: str, model: str, config: dict | None,
                   baseline) -> tuple[dict, str, str]:
    """Resolve and validate the one config that is frozen before test."""
    subdir = config_subdir(baseline, config)
    if config is not None:
        return dict(config), "manual", subdir

    path = calibration_dir(slug(model), baseline.KEY, task) / "summary.json"
    if not path.exists():
        raise SystemExit(
            f"no calibration for {task} at {path}\n"
            "either run calibration first, or pass --config explicitly"
        )
    summary = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "model": model,
        "task": task,
        "method": baseline.KEY,
        "split": "dev",
    }
    mismatches = [
        f"{key}={summary.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if summary.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            f"Calibration identity mismatch in {path}: " + ", ".join(mismatches)
        )
    chosen = summary.get("chosen")
    if not isinstance(chosen, dict):
        raise ValueError(f"Calibration summary {path} has no chosen config")
    axes = summary["axes"]
    missing = [axis for axis in axes if axis not in chosen]
    if missing:
        raise ValueError(
            f"Calibration summary {path} chose a config without {missing}"
        )
    resolved = {axis: chosen[axis] for axis in axes}
    return resolved, "calibration", config_subdir(baseline, None)


def bootstrap(flags: list[bool], n: int = N_BOOT,
              seed: int = 0) -> tuple[float | None, float | None]:
    """Return a reproducible item-bootstrap 95% interval for a proportion."""
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    if not flags:
        return None, None
    rng = random.Random(seed)
    k = len(flags)
    means = sorted(sum(rng.choices(flags, k=k)) / k for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def code_version() -> str | None:
    """Return the repository commit these results were produced by.

    The test split answers once, but the kernel keeps evolving as methods are
    added. Without a code fingerprint there is no way to tell whether two
    methods' test numbers came from the same benchmark implementation.
    """
    def git(*arguments: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(REPO), *arguments],
                capture_output=True, text=True, check=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            return None

    head = git("rev-parse", "HEAD")
    if head is None:
        return None
    commit = head.strip()
    if not commit:
        return None
    # Uncommitted edits are not described by the commit hash, so say so
    # rather than let the result claim a version it was not produced by.
    dirty = git("status", "--porcelain")
    return f"{commit}-dirty" if dirty and dirty.strip() else commit


def _reference_summaries(model_slug: str, task: str) -> dict[str, dict]:
    references = {}
    for condition in ("unsteered", "prompted"):
        summary = load_reference_summary(
            model_slug, "test", task, condition, required=False
        )
        if summary is not None:
            references[condition] = summary
    return references


def _owned_artifacts(out_dir: Path) -> list[Path]:
    names = ("outputs.jsonl", "run.json", "summary.json", "scored_outputs.jsonl")
    return (
        [out_dir / name for name in names if (out_dir / name).exists()]
        + ([out_dir / "judging"] if (out_dir / "judging").exists() else [])
    )


def _reset(out_dir: Path) -> None:
    for path in _owned_artifacts(out_dir):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def generate(args) -> None:
    """Generate the complete test split and write ``run.json`` last."""
    if args.batch < 1:
        raise ValueError(f"batch must be positive, got {args.batch}")
    model_slug = slug(args.model)
    baseline = baselines.get(args.baseline)
    config, source, subdir = resolve_config(
        args.task, args.model, parse_config(args.config), baseline
    )
    out_dir = test_dir(model_slug, baseline.KEY, args.task, subdir)
    existing = _owned_artifacts(out_dir)
    if args.force:
        _reset(out_dir)
    elif existing:
        raise FileExistsError(
            f"Test artifacts already exist (for example {existing[0]}). "
            "Test is scored once; pass --force only to recover or replace "
            "that declared run."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    from kernel.generate import GenConfig, Generator

    items = eval_items(args.task, "test")
    if not items:
        raise ValueError(f"No test items found for task {args.task!r}")
    artifact = baseline.load_artifact(model_slug, args.task)
    generator = Generator(GenConfig(model=args.model, batch_size=args.batch))
    layer = config.get("layer")
    if layer is not None and not 1 <= int(layer) <= generator.n_layers:
        raise ValueError(
            f"layer must be between 1 and {generator.n_layers}, got {layer}"
        )
    described = " ".join(f"{axis}={value}" for axis, value in config.items())
    print(f"task={args.task} {described} ({source})")
    print(f"test items: {len(items)}")

    outputs = generator.generate(
        [row["base_prompt"] for row in items],
        intervention=baseline.make_intervention(
            generator.model, artifact, config
        ),
    )
    records = build_records(
        args.task, "test", items, outputs, subdir,
        {"method": baseline.KEY, "config": subdir, **config},
    )
    if not records:
        raise ValueError(f"No test records were generated for task {args.task!r}")
    rules = summarize_rules(records)
    if not rules["n_rule_scored"]:
        raise RuntimeError("All test outputs were excluded from rule scoring")
    _write_jsonl(out_dir / "outputs.jsonl", records)
    n_to_judge = prepare_rule_passers(
        records, out_dir / "judging", n_batches=N_JUDGE_BATCHES, seed=11
    )

    run = {
        "schema_version": 1,
        "model": args.model,
        "model_slug": model_slug,
        "model_revision": getattr(generator.model.config, "_commit_hash", None),
        "task": args.task,
        "method": baseline.KEY,
        "method_version": getattr(baseline, "METHOD_VERSION", 1),
        "code_version": code_version(),
        "split": "test",
        "intervention_positions": POSITION_MODE,
        "config_values": dict(config),
        "config_source": source,
        "config": subdir,
        # The published number needs the dose beside it: alpha means a
        # different physical quantity in each baseline, so a leaderboard of
        # alphas would invite a comparison the numbers do not support.
        "dose": baseline.dose_metadata(artifact, config),
        "n_items": len(records),
        "reason": args.reason,
        "generation": {
            "max_new_tokens": generator.cfg.max_new_tokens,
            "do_sample": False,
            "dtype": str(generator.cfg.dtype),
            "batch_size": generator.cfg.batch_size,
            "stripped_think_outputs": generator.stripped_think,
        },
        "rules": rules,
        "n_rule_pass_sent_to_judge": n_to_judge,
        "references": _reference_summaries(model_slug, args.task),
        "outputs": "outputs.jsonl",
        "judging": "judging",
    }
    _write_json(out_dir / "run.json", run)
    rate = rules["rule_pass_rate"]
    print(f"\nrule_pass {rate:.1%} (upper bound, not final accuracy)")
    print(f"to judge: {n_to_judge} rule-pass outputs")
    print(f"\nnext:\n  judge_local.py {out_dir / 'judging'}")
    print(
        f"  evaluate.py {args.task} --report"
        + ("" if source == "calibration"
           else " --config " + ",".join(
               f"{axis}={value}" for axis, value in config.items()))
    )


def _validate_run(run: dict, args, subdir: str) -> None:
    expected = {
        "model": args.model,
        "task": args.task,
        "method": args.baseline,
        "split": "test",
        "config": subdir,
    }
    mismatches = [
        f"{key}={run.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if run.get(key) != value
    ]
    requested = parse_config(args.config)
    if requested is not None and run.get("config_values") != requested:
        mismatches.append(
            f"stored config={run.get('config_values')} (expected {requested})"
        )
    if mismatches:
        raise ValueError("Test run identity mismatch: " + ", ".join(mismatches))


def report(args) -> None:
    model_slug = slug(args.model)
    subdir = config_subdir(baselines.get(args.baseline), parse_config(args.config))
    out_dir = test_dir(model_slug, args.baseline, args.task, subdir)
    run_path = out_dir / "run.json"
    if not run_path.exists():
        raise FileNotFoundError(
            f"{run_path} does not exist; complete test generation first"
        )
    run = json.loads(run_path.read_text(encoding="utf-8"))
    _validate_run(run, args, subdir)
    records = _read_jsonl(out_dir / "outputs.jsonl")
    if len(records) != run["n_items"]:
        raise ValueError(
            f"Test run records changed from {run['n_items']} to {len(records)}"
        )
    diagnostics = merge_judgments(records, [out_dir / "judging"])
    if diagnostics["n_unjudged"]:
        raise SystemExit(
            f"{diagnostics['n_unjudged']} rule-pass outputs have no judge "
            f"verdict in {out_dir / 'judging'}; complete judging before reporting"
        )
    metrics = summarize_merged(records)
    scored_path = write_scored_records(
        records, out_dir / "scored_outputs.jsonl"
    )
    flags = [row["success"] for row in records if row["success"] is not None]
    low, high = bootstrap(flags)

    print(
        f"\n{args.task} "
        + " ".join(f"{a}={v}" for a, v in run.get("config_values", {}).items())
        + " "
        f"({run['config_source']})"
    )
    print("=" * 62)
    print(
        f"ACCURACY {metrics['accuracy']:.1%} "
        f"95% CI [{low:.1%}, {high:.1%}] n={metrics['n_rule_scored']}"
    )
    print("=" * 62)
    print(f"  rule pass rate      {metrics['rule_pass_rate']:.1%} (upper bound)")
    print(f"  judged incoherent   {metrics['n_incoherent']}")
    print(f"  rule-pass outputs not answering  {metrics['n_not_answering']}")
    print(f"  truncated (counted as failures)  {metrics['n_truncated']}")
    if metrics["judge_suspect"]:
        print(
            f"  ! judge called {metrics['judge_suspect']} repetition loops coherent"
        )

    summary = {
        **run,
        "accuracy": metrics["accuracy"],
        "ci95": [low, high],
        "n_scored": metrics["n_rule_scored"],
        "n_rule_pass": metrics["n_rule_pass"],
        "rule_pass_rate": metrics["rule_pass_rate"],
        "incoherent": metrics["n_incoherent"],
        "not_answering": metrics["n_not_answering"],
        "n_truncated": metrics["n_truncated"],
        "n_judged": diagnostics["n_judged"],
        "n_unjudged": diagnostics["n_unjudged"],
        "judge_suspect": metrics["judge_suspect"],
        "n_bootstrap": N_BOOT,
        "scored_outputs": scored_path.name,
    }
    _write_json(out_dir / "summary.json", summary)
    print(f"\n-> {scored_path}")
    print(f"-> {out_dir / 'summary.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=list(TASKS))
    parser.add_argument("--model", default=TARGET_MODEL)
    parser.add_argument(
        "--baseline", default=baselines.DEFAULT_BASELINE, choices=baselines.keys()
    )
    parser.add_argument(
        "--config",
        help="externally selected configuration, e.g. layer=21,alpha=1.0 or "
             "heads_k=48,alpha=15 — the axes are the baseline's own",
    )
    parser.add_argument("--reason", default="", help="provenance for an external config")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument(
        "--force", action="store_true",
        help="replace a declared test run for explicit recovery"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="merge judge verdicts and write the final test score"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        parse_config(args.config)
    except ValueError as error:
        parser.error(str(error))
    (report if args.report else generate)(args)


if __name__ == "__main__":
    main()
