"""Full surface reporting; never chooses an operating point."""

import csv
from collections import defaultdict

from l2s.storage import write_json

from .artifacts import read_jsonl
from .config import ITI_PROTOCOL, learned_cells
from .evaluate import row_key
from .judge import judgment_key, load_judgments
from .settings import (
    DEFAULT_BATCH_SIZE,
    EVAL_SIZES,
    MAX_NEW_TOKENS,
    eval_sets_for,
    token_limit,
)


def summarize(rows, baseline):
    length = sum(row["n_tokens"] for row in rows) / len(rows)
    baseline_length = (
        sum(row["n_tokens"] for row in baseline) / len(baseline) if baseline else None
    )
    unresolved = sum(row["correct"] is None for row in rows)
    return dict(
        n=len(rows),
        Length=length,
        Accuracy=None
        if unresolved
        else sum(bool(row["correct"]) for row in rows) / len(rows),
        Compression=None if not baseline_length else 1 - length / baseline_length,
        CollapseRate=sum(row["collapsed"] for row in rows) / len(rows),
        unresolved_answers=unresolved,
    )


def build_report(
    root,
    names,
    allow_partial=False,
    plots=True,
    *,
    track="all",
    eval_sets=None,
    max_new_tokens=None,
):
    judgments = load_judgments(root)
    selected = eval_sets_for(eval_sets)
    token_cap = token_limit(MAX_NEW_TOKENS if max_new_tokens is None else max_new_tokens)
    summary, status = [], {}
    for name in names:
        groups = defaultdict(list)
        seen = set()
        for path in sorted((root / "runs" / name).glob("*/grid*.jsonl")):
            for row in read_jsonl(path):
                key = row_key(row)
                if key in seen:
                    raise RuntimeError("Duplicate result rows")
                seen.add(key)
                if row["needs_judge"]:
                    row["correct"] = judgments.get(
                        judgment_key(row["answer_parsed"], row["gold_answer"])
                    )
                groups[
                    row["method"], row["layer"], row["strength"], row["eval_set"]
                ].append(row)
        expected = [
            (method, layer, strength, dataset)
            for method, layer, strength in [
                ("No-Steering", None, None),
                ("System-Prompt", None, None),
            ]
            + list(learned_cells(name, track))
            for dataset in selected
        ]
        if set(groups) - set(expected):
            raise RuntimeError("Results contain cells outside the configured full grid")
        missing = []
        for method, layer, strength, dataset in expected:
            key = method, layer, strength, dataset
            rows = groups.get(key, [])
            baseline = groups.get(("No-Steering", None, None, dataset), [])
            if len(rows) != EVAL_SIZES[dataset] or {r["problem_id"] for r in rows} != {
                r["problem_id"] for r in baseline
            }:
                missing.append(
                    dict(
                        method=method,
                        layer=layer,
                        strength=strength,
                        eval_set=dataset,
                        observed=len(rows),
                        expected=EVAL_SIZES[dataset],
                    )
                )
            if not rows:
                continue
            slices = [("all", rows, baseline)]
            for difficulty in sorted({row["difficulty"] for row in rows}):
                slices.append(
                    (
                        f"difficulty_{difficulty}",
                        [r for r in rows if r["difficulty"] == difficulty],
                        [r for r in baseline if r["difficulty"] == difficulty],
                    )
                )
            if dataset == "amc12":
                for has_figure in (False, True):
                    subset = [r for r in rows if r["has_figure"] == has_figure]
                    if subset:
                        slices.append(
                            (
                                "figure" if has_figure else "no_figure",
                                subset,
                                [r for r in baseline if r["has_figure"] == has_figure],
                            )
                        )
            for slice_name, subset, base in slices:
                # Compression uses the same observed problem IDs even in an explicitly partial report.
                subset_ids = {r["problem_id"] for r in subset}
                base = [r for r in base if r["problem_id"] in subset_ids]
                if {r["problem_id"] for r in base} != subset_ids:
                    base = []
                summary.append(
                    dict(
                        model=name,
                        method=method,
                        layer=layer,
                        strength=strength,
                        eval_set=dataset,
                        slice=slice_name,
                        **summarize(subset, base),
                    )
                )
        unresolved = sum(
            row["correct"] is None for rows in groups.values() for row in rows
        )
        status[name] = dict(
            available_grid_complete=not missing,
            missing_cells=missing,
            unresolved_answers=unresolved,
            blocked_methods={},
            track=track,
            eval_sets=selected,
            max_new_tokens=token_cap,
            batch_size=DEFAULT_BATCH_SIZE,
            iti_protocol=ITI_PROTOCOL if track != "non-iti" else None,
            track_complete=not missing and not unresolved,
            full_revised_plan_complete=track == "all"
            and not missing
            and not unresolved,
        )
    directory = root / "reports"
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "STATUS.json", status)
    incomplete = any(
        s["missing_cells"] or s["unresolved_answers"] for s in status.values()
    )
    if incomplete and not allow_partial:
        raise RuntimeError(
            "Report incomplete: see reports/STATUS.json. Finish generation/judging or use --allow-partial"
        )
    if not summary:
        raise RuntimeError("No generated results to report")
    stem = "partial_grid" if incomplete else "full_grid"
    with (directory / f"{stem}.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    write_json(directory / f"{stem}.json", summary)
    baseline = [
        row
        for row in summary
        if row["method"] == "No-Steering" and row["slice"] == "all"
    ]
    write_json(
        directory / "baselines.json",
        [
            dict(row, Acc_base=row["Accuracy"], Length_base=row["Length"])
            for row in baseline
        ],
    )
    if plots:
        heatmaps(directory, summary)
    write_json(directory / "iti_profiles.json", {})
    print(
        f"Report written to {directory}; execution track: {track}.",
        flush=True,
    )


def heatmaps(directory, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    grouped = defaultdict(list)
    for row in rows:
        if row["slice"] == "all" and (
            row["layer"] is not None or row["method"] == "ITI"
        ):
            grouped[row["model"], row["method"], row["eval_set"]].append(row)
    for (model, method, dataset), cells in grouped.items():
        layers = sorted({row["layer"] for row in cells})
        strengths = sorted({row["strength"] for row in cells})
        fig, axes = plt.subplots(1, 4, figsize=(18, 4), layout="constrained")
        for axis, metric in zip(
            axes, ("Length", "Accuracy", "Compression", "CollapseRate")
        ):
            matrix = np.full((len(layers), len(strengths)), np.nan)
            for row in cells:
                if row[metric] is not None:
                    matrix[
                        layers.index(row["layer"]), strengths.index(row["strength"])
                    ] = row[metric]
            image = axis.imshow(matrix, aspect="auto")
            axis.set(
                title=metric,
                xlabel="Strength",
                ylabel="Profile" if method == "ITI" else "Layer",
                xticks=range(len(strengths)),
                xticklabels=strengths,
                yticks=range(len(layers)),
                yticklabels=["all-layer Top-48"] if method == "ITI" else layers,
            )
            axis.tick_params(axis="x", rotation=90)
            fig.colorbar(image, ax=axis)
        fig.suptitle(f"{model} · {method} · {dataset}")
        fig.savefig(directory / f"{model}_{method}_{dataset}.png", dpi=160)
        plt.close(fig)
