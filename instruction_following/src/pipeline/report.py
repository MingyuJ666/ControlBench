"""Collect every baseline's results for one model into one table.

Each run writes its own summary in its own directory, which is right for
producing them and useless for reading them: the question a reader asks is how
the baselines compare, and answering it means opening twenty-five files. This
writes the comparison instead, to ``results/<model>/``.

Three things it will not do quietly. A cell with no result is printed as a gap
rather than skipped, so a table that looks complete is complete. Results
carrying different code versions or different intervention-position rules are
reported as such, because a leaderboard implies its rows were produced the
same way. And the unsteered and prompted references are kept in their own
section, since they are rule-pass rates without a judge and are not the same
measurement as a baseline's accuracy.

  python src/pipeline/report.py                 # test results
  python src/pipeline/report.py --split dev     # the dev tables, before test
"""

# Run directly as a script as well as imported: the package root
# is this file's ancestor, not its own directory.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402

import baselines  # noqa: E402
from kernel.config import (RESULTS, TARGET_MODEL, calibration_dir,  # noqa: E402
                           slug, test_dir)
from kernel.reference_conditions import load_reference_summary  # noqa: E402
from kernel.residual_scale import load as load_residual_scale  # noqa: E402
from kernel.residual_scale import ratio as dose_ratio  # noqa: E402
from kernel.tasks import TASKS  # noqa: E402

GAP = "—"
CONDITIONS = ("unsteered", "prompted")


def _read(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def test_cell(model_slug: str, baseline: str, task: str) -> dict | None:
    """One frozen test result, with the provenance it was produced under."""
    directory = test_dir(model_slug, baseline, task, "calibrated")
    summary = _read(directory / "summary.json")
    if summary is None:
        return None
    return {
        "accuracy": summary.get("accuracy"),
        "ci95": summary.get("ci95"),
        "config": summary.get("config_values", {}),
        "n_scored": summary.get("n_scored"),
        "code_version": summary.get("code_version"),
        "positions": summary.get("intervention_positions"),
        "dose": summary.get("dose"),
    }


def dev_cell(model_slug: str, baseline: str, task: str) -> dict | None:
    """The dev configuration a baseline chose, and what it scored choosing it.

    A dev number is what selection maximized, so it is an optimistic estimate
    of the same quantity test measures. Reported to see progress before test,
    labelled so it is not mistaken for the result.
    """
    summary = _read(calibration_dir(model_slug, baseline, task) / "summary.json")
    if summary is None:
        return None
    chosen = summary.get("chosen") or {}
    axes = summary["axes"]
    return {
        "accuracy": chosen.get("accuracy"),
        "ci95": None,
        "config": {axis: chosen[axis] for axis in axes if axis in chosen},
        "n_scored": chosen.get("n"),
        "code_version": summary.get("code_version"),
        "positions": summary.get("intervention_positions"),
        "dose": chosen.get("dose"),
    }


def references(model_slug: str, split: str) -> dict[str, dict[str, float | None]]:
    """Rule-pass rates for the unsteered and prompted conditions."""
    rows = {}
    for condition in CONDITIONS:
        rows[condition] = {}
        for task in TASKS:
            summary = load_reference_summary(
                model_slug, split, task, condition, required=False
            )
            rows[condition][task] = (
                None if summary is None else summary.get("rule_pass_rate")
            )
    return rows


def collect(model: str, split: str, keys: list[str] | None = None) -> dict:
    """Gather every cell, plus what would make the cells incomparable."""
    model_slug = slug(model)
    keys = keys or baselines.keys()
    read_cell = test_cell if split == "test" else dev_cell

    table, provenance, gaps = {}, [], []
    for key in keys:
        table[key] = {}
        for task in TASKS:
            cell = read_cell(model_slug, key, task)
            table[key][task] = cell
            if cell is None:
                gaps.append((key, task))
            else:
                provenance.append({
                    "baseline": key, "task": task,
                    "code_version": cell["code_version"],
                    "positions": cell["positions"],
                })
    return {
        "model": model,
        "split": split,
        "tasks": list(TASKS),
        "baselines": keys,
        "table": table,
        "references": references(model_slug, split),
        "residual_scale": load_residual_scale(model_slug),
        "gaps": gaps,
        "provenance": provenance,
    }


def _distinct(provenance: list[dict], field: str) -> list:
    return sorted({str(row[field]) for row in provenance if row[field] is not None})


def _percent(value) -> str:
    return GAP if value is None else f"{value:.1%}"


def _cell_text(cell: dict | None) -> str:
    if cell is None or cell.get("accuracy") is None:
        return GAP
    text = _percent(cell["accuracy"])
    interval = cell.get("ci95")
    if interval and all(bound is not None for bound in interval):
        text += f" [{interval[0]:.1%}, {interval[1]:.1%}]"
    return text


def render_markdown(collected: dict) -> str:
    """The table a reader wants, with the caveats attached to it."""
    tasks = collected["tasks"]
    lines = [
        f"# {collected['split']} results — {collected['model']}",
        "",
        "| baseline | " + " | ".join(tasks) + " | mean |",
        "|---|" + "---|" * (len(tasks) + 1),
    ]
    for key in collected["baselines"]:
        cells = [collected["table"][key][task] for task in tasks]
        scores = [
            cell["accuracy"] for cell in cells
            if cell is not None and cell.get("accuracy") is not None
        ]
        mean = _percent(sum(scores) / len(scores)) if len(scores) == len(tasks) else GAP
        lines.append(
            f"| {key} | " + " | ".join(_cell_text(cell) for cell in cells)
            + f" | {mean} |"
        )

    lines += [
        "",
        "Chosen configuration per cell, with the dose it applied:",
        "",
        "An alpha is not a dose: it multiplies a raw magnitude for caa, a",
        "projection sigma for probe and lat_pca, a feature's peak activation",
        "for sae, and a per-head sigma once per selected head for iti. The",
        "ratio is what compares — the norm of what was added over the RMS of",
        "the residual stream at that site. It is blank where the site is not",
        "a single residual-stream layer, or where the scale was never",
        "measured (kernel/residual_scale.py).",
        "",
        "| baseline | " + " | ".join(tasks) + " |",
        "|---|" + "---|" * len(tasks),
    ]
    scale = collected.get("residual_scale")
    for key in collected["baselines"]:
        named = []
        for task in tasks:
            cell = collected["table"][key][task]
            if cell is None:
                named.append(GAP)
                continue
            text = " ".join(f"{a}={v}" for a, v in cell["config"].items()) or GAP
            measured = dose_ratio(cell.get("dose") or {}, scale)
            if measured is not None:
                text += f" ({measured:.2f}x)"
            named.append(text)
        lines.append(f"| {key} | " + " | ".join(named) + " |")

    lines += [
        "",
        "## Reference conditions",
        "",
        "Rule-pass rates, not benchmark accuracy: these conditions are not",
        "judged, so they bound the range rather than sharing its measurement.",
        "",
        "| condition | " + " | ".join(tasks) + " |",
        "|---|" + "---|" * len(tasks),
    ]
    for condition, row in collected["references"].items():
        lines.append(
            f"| {condition} | "
            + " | ".join(_percent(row[task]) for task in tasks) + " |"
        )

    lines += ["", "## Provenance", ""]
    versions = _distinct(collected["provenance"], "code_version")
    positions = _distinct(collected["provenance"], "positions")
    if len(versions) > 1:
        lines.append(
            f"- **cells come from {len(versions)} code versions**: "
            + ", ".join(versions)
        )
    elif versions:
        lines.append(f"- code version: {versions[0]}")
    if len(positions) > 1:
        lines.append(
            f"- **cells come from {len(positions)} intervention-position "
            f"rules**: " + ", ".join(positions)
        )
    elif positions:
        lines.append(f"- intervention positions: {positions[0]}")
    if collected["gaps"]:
        listed = ", ".join(f"{key}/{task}" for key, task in collected["gaps"])
        lines.append(f"- **{len(collected['gaps'])} cells not run**: {listed}")
    else:
        lines.append("- every cell present")
    return "\n".join(lines) + "\n"


def write(collected: dict, root: Path | None = None) -> list[Path]:
    """Write the table as markdown to read and csv to plot."""
    directory = (root or RESULTS) / slug(collected["model"])
    directory.mkdir(parents=True, exist_ok=True)
    written = []

    markdown = directory / f"{collected['split']}_leaderboard.md"
    markdown.write_text(render_markdown(collected), encoding="utf-8")
    written.append(markdown)

    table = directory / f"{collected['split']}_leaderboard.csv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "baseline", "task", "accuracy", "ci95_low", "ci95_high",
            "n_scored", "config", "code_version", "intervention_positions",
            "effective_delta_norm", "residual_rms_ratio",
        ])
        for key in collected["baselines"]:
            for task in collected["tasks"]:
                cell = collected["table"][key][task]
                if cell is None:
                    writer.writerow([key, task] + [""] * 9)
                    continue
                low, high = (cell["ci95"] or [None, None])[:2]
                dose = cell.get("dose") or {}
                writer.writerow([
                    key, task, cell["accuracy"], low, high, cell["n_scored"],
                    ";".join(f"{a}={v}" for a, v in cell["config"].items()),
                    cell["code_version"], cell["positions"],
                    dose.get("effective_delta_norm"),
                    dose_ratio(dose, collected.get("residual_scale")),
                ])
    written.append(table)

    raw = directory / f"{collected['split']}_leaderboard.json"
    raw.write_text(json.dumps(collected, indent=2), encoding="utf-8")
    written.append(raw)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=TARGET_MODEL)
    parser.add_argument("--split", default="test", choices=("dev", "test"))
    parser.add_argument(
        "--baselines", help="comma-separated subset; default is all registered"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    keys = args.baselines.split(",") if args.baselines else None
    collected = collect(args.model, args.split, keys)
    print(render_markdown(collected))
    for path in write(collected):
        print(f"-> {path}")


if __name__ == "__main__":
    main()
