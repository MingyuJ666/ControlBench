import json
import math

import pytest

from repsteer.selection import (
    Candidate,
    CandidateEvaluation,
    HoldoutSplit,
    SecondaryMetric,
    SelectionReport,
    grid_search,
    make_holdout_split,
    select_best,
)


def _evaluations():
    return (
        CandidateEvaluation(
            Candidate("z", {"threshold": 0.5}), {"f1": 0.8, "accuracy": 0.9}
        ),
        CandidateEvaluation(
            Candidate("a", {"threshold": 0.2}), {"f1": 0.8, "accuracy": 0.9}
        ),
        CandidateEvaluation(
            Candidate("m", {"threshold": 0.1}), {"f1": 0.7, "accuracy": 1.0}
        ),
    )


def test_holdout_split_is_deterministic_disjoint_and_validates_source_ids():
    ids = [f"sample-{index}" for index in range(20)]
    first = make_holdout_split(ids, validation_ratio=0.25, seed=17)
    second = make_holdout_split(ids, validation_ratio=0.25, seed=17)
    changed_seed = make_holdout_split(ids, validation_ratio=0.25, seed=18)

    assert first == second
    assert first.validation_ids != changed_seed.validation_ids
    assert not set(first.train_ids) & set(first.validation_ids)
    assert first.source_fingerprint
    assert HoldoutSplit.from_dict(first.to_dict()) == first
    assert first.validate_against(ids) is first

    with pytest.raises(ValueError, match="duplicate"):
        make_holdout_split(["one", "one"], seed=1)
    with pytest.raises(ValueError, match="strictly between"):
        make_holdout_split(ids, validation_ratio=1)
    with pytest.raises(ValueError, match="unknown source IDs"):
        HoldoutSplit(("unknown",), ("sample-0",)).validate_against(ids)
    with pytest.raises(ValueError, match="non-empty string"):
        make_holdout_split(ids, source_fingerprint="")


def test_candidate_records_are_canonical_and_reject_executable_or_nonfinite_values():
    left = Candidate("candidate", {"layer": 2, "nested": {"b": 2, "a": 1}})
    right = Candidate("candidate", {"nested": {"a": 1, "b": 2}, "layer": 2})

    assert left.canonical == right.canonical
    assert left.fingerprint == right.fingerprint
    with pytest.raises(ValueError, match="finite"):
        Candidate("bad", {"threshold": math.nan})
    with pytest.raises(TypeError, match="JSON primitives"):
        Candidate("bad", {"callback": lambda: None})
    with pytest.raises(ValueError, match="finite"):
        CandidateEvaluation(Candidate("bad"), {"f1": math.inf})


def test_selection_supports_maximize_minimize_secondary_and_canonical_id_ties():
    values = _evaluations()

    assert select_best(values, objective="f1", maximize=True).candidate_id == "a"
    assert select_best(values, objective="accuracy", maximize=False).candidate_id == "a"
    assert (
        select_best(
            values,
            objective="f1",
            secondary_metrics=(SecondaryMetric("accuracy", maximize=True),),
        ).candidate_id
        == "a"
    )
    with pytest.raises(KeyError, match="missing required"):
        select_best(values, objective="missing")
    with pytest.raises(ValueError, match="unique"):
        select_best((values[0], values[0]), objective="f1")


def test_grid_search_round_trips_replays_and_rejects_report_tampering(tmp_path):
    split = make_holdout_split(
        ["a", "b", "c", "d"], seed=4, source_fingerprint="sha256:source"
    )
    candidates = (
        Candidate("higher", {"threshold": 0.8}),
        Candidate("lower", {"threshold": 0.2}),
    )
    report = grid_search(
        candidates,
        lambda candidate: {"f1": 0.75 if candidate.id == "lower" else 0.5},
        objective="f1",
        split=split,
        source_fingerprint="sha256:source",
        seed=4,
        metadata={"purpose": "unit"},
    )

    assert report.selected_id == "lower"
    assert report.selected.candidate == candidates[1]
    assert report.provenance["package"] == "repsteer"
    json_path = tmp_path / "selection.json"
    report.to_json(json_path)
    assert SelectionReport.from_json(json_path) == report
    assert SelectionReport.from_json(report.to_json()) == report

    tampered = report.to_dict()
    tampered["selected_id"] = "higher"
    with pytest.raises(ValueError, match="replayed selection|checksum"):
        SelectionReport.from_dict(tampered)
    tampered = json.loads(report.to_json())
    tampered["evaluations"][0]["metrics"]["f1"] = 1.0
    with pytest.raises(ValueError, match="replayed selection|checksum"):
        SelectionReport.from_dict(tampered)
    tampered = report.to_dict()
    tampered["schema_version"] = "99.0"
    with pytest.raises(ValueError, match="unsupported SelectionReport schema"):
        SelectionReport.from_dict(tampered)


def test_selection_report_is_canonical_and_rejects_incomplete_or_falsy_checksums():
    split = make_holdout_split(["a", "b", "c", "d"], seed=4)
    values = _evaluations()
    report = SelectionReport(
        objective="f1",
        maximize=True,
        evaluations=values,
        selected_id="a",
        split=split,
        selection_seed=4,
    )
    reversed_report = SelectionReport(
        objective="f1",
        maximize=True,
        evaluations=tuple(reversed(values)),
        selected_id="a",
        split=split,
        selection_seed=4,
    )

    assert reversed_report.to_dict() == report.to_dict()
    assert reversed_report.checksum == report.checksum

    for checksum in ("", None, 0, False):
        tampered = report.to_dict()
        tampered["checksum"] = checksum
        with pytest.raises(ValueError, match="checksum"):
            SelectionReport.from_dict(tampered)

    incomplete = report.to_dict()
    incomplete.pop("tie_break_rule")
    with pytest.raises(ValueError, match="missing required fields"):
        SelectionReport.from_dict(incomplete)

    with pytest.raises(ValueError, match="source_fingerprint does not match"):
        SelectionReport(
            objective="f1",
            maximize=True,
            evaluations=values,
            selected_id="a",
            split=split,
            source_fingerprint="sha256:not-the-split",
            selection_seed=4,
        )
    with pytest.raises(ValueError, match="provenance.package"):
        SelectionReport(
            objective="f1",
            maximize=True,
            evaluations=values,
            selected_id="a",
            split=split,
            selection_seed=4,
            provenance={"package": None, "version": ""},
        )


def test_grid_search_fails_closed_with_candidate_context_and_no_partial_report():
    candidate = Candidate("failing", {"x": 1})
    split = make_holdout_split(["a", "b"], seed=1)

    with pytest.raises(RuntimeError, match="candidate 'failing'"):
        grid_search(
            (candidate,),
            lambda _candidate: (_ for _ in ()).throw(RuntimeError("broken evaluator")),
            objective="f1",
            split=split,
        )
    with pytest.raises(ValueError, match="partial"):
        grid_search(
            (candidate,),
            lambda _candidate: {"f1": 1.0},
            objective="f1",
            split=split,
            allow_partial=True,
        )


def test_grid_search_validates_selection_configuration_before_calling_evaluator():
    calls = []

    def evaluator(_candidate):
        calls.append("called")
        return {"f1": 1.0}

    with pytest.raises(ValueError, match="objective cannot also"):
        grid_search(
            (Candidate("candidate", {"threshold": 0.1}),),
            evaluator,
            objective="f1",
            secondary_metrics=("f1",),
        )

    assert calls == []

    with pytest.raises(ValueError, match="requires split"):
        grid_search(
            (Candidate("candidate", {"threshold": 0.1}),),
            evaluator,
            objective="f1",
        )

    assert calls == []
