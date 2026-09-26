"""Internal synthetic condition/behavior vertical slice; not a method API."""

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from repsteer.artifacts import (
    ArtifactBundle,
    ArtifactBundleComponent,
    ArtifactMetadata,
    DirectionArtifact,
)
from repsteer.core import (
    ArtifactCompatibilityError,
    ArtifactFormatError,
    Site,
    SiteResolutionError,
    UnsafeArtifactError,
)
from repsteer.selection import Candidate, HoldoutSplit, SelectionReport, grid_search

_SCORES = {
    "train-0": (0.95, 1),
    "train-1": (0.05, 0),
    "validation-0": (0.90, 1),
    "validation-1": (0.80, 1),
    "validation-2": (0.20, 0),
    "validation-3": (0.10, 0),
}


def _f1(candidate, validation_ids):
    threshold = candidate.parameters["threshold"]
    comparator = candidate.parameters["comparator"]
    true_positive = false_positive = false_negative = 0
    for identifier in validation_ids:
        score, label = _SCORES[identifier]
        prediction = score > threshold if comparator == "greater" else score < threshold
        true_positive += int(prediction and label)
        false_positive += int(prediction and not label)
        false_negative += int(not prediction and label)
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


class _Target:
    model_id = "synthetic/model"
    revision = "r1"
    architecture = "SyntheticForCausalLM"
    hidden_size = 3

    def resolve_site(self, site):
        if site not in {
            Site("language", "resid_post", 0),
            Site("language", "resid_post", 1),
        }:
            raise SiteResolutionError(f"unsupported site {site}")
        return SimpleNamespace(site=site, hidden_dim=3)


def _artifact(layer, values):
    return DirectionArtifact(
        ArtifactMetadata(
            model_id="synthetic/model",
            model_revision="r1",
            architecture="SyntheticForCausalLM",
            site=Site("language", "resid_post", layer),
            method="synthetic_fixture",
            dataset_fingerprint="sha256:synthetic-data",
        ),
        torch.tensor(values),
    )


def _rewrite_root_checksum(path):
    manifest = path / "manifest.json"
    checksum = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (path / "checksums.json").write_text(
        json.dumps({"algorithm": "sha256", "files": {"manifest.json": checksum}}),
        encoding="utf-8",
    )


def test_synthetic_condition_behavior_selection_and_bundle_vertical_slice(tmp_path):
    split = HoldoutSplit(
        train_ids=("train-0", "train-1"),
        validation_ids=(
            "validation-0",
            "validation-1",
            "validation-2",
            "validation-3",
        ),
        seed=12,
        strategy="explicit",
        source_fingerprint="sha256:synthetic-data",
    )
    split.validate_against(_SCORES)
    candidates = (
        Candidate("greater_0.5", {"threshold": 0.5, "comparator": "greater"}),
        Candidate("greater_0.7", {"threshold": 0.7, "comparator": "greater"}),
        Candidate("less_0.5", {"threshold": 0.5, "comparator": "less"}),
    )

    report = grid_search(
        candidates,
        lambda candidate: {"f1": _f1(candidate, split.validation_ids)},
        objective="f1",
        split=split,
        source_fingerprint="sha256:synthetic-data",
        seed=12,
        metadata={"internal_fixture": True},
    )
    repeated = grid_search(
        candidates,
        lambda candidate: {"f1": _f1(candidate, split.validation_ids)},
        objective="f1",
        split=split,
        source_fingerprint="sha256:synthetic-data",
        seed=12,
        metadata={"internal_fixture": True},
    )

    assert report.selected_id == "greater_0.5"
    assert report.selected.candidate.parameters["comparator"] == "greater"
    assert report.selected.candidate.parameters["threshold"] == 0.5
    assert not set(split.train_ids) & set(split.validation_ids)
    assert SelectionReport.from_dict(report.to_dict()) == report
    assert report.to_dict() == repeated.to_dict()

    bundle = ArtifactBundle(
        components=(
            ArtifactBundleComponent(
                "condition/layer_0", "condition", _artifact(0, [1, 0, 0])
            ),
            ArtifactBundleComponent(
                "behavior/layer_1", "behavior", _artifact(1, [0, 1, 0])
            ),
        ),
        selection_report=report,
        provenance={"internal_fixture": "synthetic"},
    )
    bundle.save(tmp_path)
    first_manifest = (tmp_path / "manifest.json").read_bytes()
    bundle.save(tmp_path)
    assert (tmp_path / "manifest.json").read_bytes() == first_manifest
    restored = ArtifactBundle.load(tmp_path)

    assert restored.component("condition/layer_0").role == "condition"
    assert restored.component("behavior/layer_1").role == "behavior"
    assert (
        restored.component("condition/layer_0").artifact.fingerprint()
        != restored.component("behavior/layer_1").artifact.fingerprint()
    )
    assert restored.bind(_Target()) is restored

    wrong_target = _Target()
    wrong_target.model_id = "other/model"
    with pytest.raises(ArtifactCompatibilityError, match="condition/layer_0"):
        restored.bind(wrong_target)

    component_tamper_path = tmp_path / "component-tamper"
    bundle.save(component_tamper_path)
    condition_path = (
        component_tamper_path
        / bundle.component("condition/layer_0").relative_path
        / "tensors.safetensors"
    )
    behavior_path = (
        component_tamper_path
        / bundle.component("behavior/layer_1").relative_path
        / "tensors.safetensors"
    )
    condition_bytes = condition_path.read_bytes()
    behavior_bytes = behavior_path.read_bytes()
    condition_path.write_bytes(behavior_bytes)
    behavior_path.write_bytes(condition_bytes)
    with pytest.raises(UnsafeArtifactError, match="Checksum mismatch"):
        ArtifactBundle.load(component_tamper_path)

    report_tamper_path = tmp_path / "report-tamper"
    bundle.save(report_tamper_path)
    manifest_path = report_tamper_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selection_report"]["selected_id"] = "less_0.5"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _rewrite_root_checksum(report_tamper_path)
    with pytest.raises(ArtifactFormatError, match="selection_report is invalid"):
        ArtifactBundle.load(report_tamper_path)
