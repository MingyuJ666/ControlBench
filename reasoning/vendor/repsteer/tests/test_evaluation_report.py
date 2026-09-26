import json

import pytest

from repsteer.evaluation import SweepRecord, SweepReport


def _record(strength: float, target: float, damage: float) -> SweepRecord:
    return SweepRecord(
        site={"stream": "language", "component": "resid_post", "layer": 1},
        strength=strength,
        positions="GeneratedTokens",
        phase="decode",
        metrics={"target": target, "damage": damage},
        samples=2,
        elapsed_seconds=0.01,
    )


def test_report_json_markdown_and_explicit_selection(tmp_path):
    report = SweepReport(
        records=(_record(0, 0.0, 0.0), _record(1, 0.8, 0.1), _record(2, 1.0, 0.5)),
        search_space={"strengths": [0, 1, 2]},
    )

    selected = report.select(maximize="target", subject_to={"damage": "<= 0.2"})
    assert selected.strength == 1

    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    assert json.loads(report.to_json(json_path))["schema_version"] == "1.0"
    assert "| site | strength |" in report.to_markdown(markdown_path)
    assert json_path.exists() and markdown_path.exists()


def test_report_requires_an_objective_and_computes_pareto_frontier():
    report = SweepReport(
        records=(_record(0, 0.0, 0.0), _record(1, 0.8, 0.1), _record(2, 1.0, 0.5)),
        search_space={},
    )

    with pytest.raises(ValueError, match="exactly one"):
        report.select()
    frontier = report.pareto_frontier(maximize=["target"], minimize=["damage"])
    assert len(frontier.records) == 3
