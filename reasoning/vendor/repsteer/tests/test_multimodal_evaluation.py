import json

import pytest

from repsteer.evaluation import (
    EvaluationMetricGroups,
    MultimodalEvaluationRecord,
    MultimodalEvaluationReport,
)


def test_multimodal_evaluation_schema_groups_metrics_and_provenance(tmp_path):
    record = MultimodalEvaluationRecord(
        sample_id="sample-1",
        input_fingerprints={"image": "sha256:image", "prompt": "sha256:prompt"},
        modality={"image_patch_grids": [[2, 3]], "image_token_counts": [6]},
        interventions=(
            {"site": "vision.vision_resid[layer=1]", "strength": 2.0},
            {"site": "language.resid_post[layer=1]", "strength": 1.0},
        ),
        metrics=EvaluationMetricGroups(
            representation={"feature_activation": 4.5},
            causal_behavior={"target_success": 1.0},
            capability_quality={"instruction_following": 0.9},
        ),
        output={"text": "A concise description."},
    )
    report = MultimodalEvaluationReport(
        model_id="tiny/vlm",
        model_revision="model-r1",
        processor_id="tiny/vlm",
        processor_revision="processor-r1",
        records=(record,),
        provenance={"seed": 7},
    )

    destination = tmp_path / "report.json"
    payload = json.loads(report.to_json(destination))

    assert payload["kind"] == "multimodal_steering_evaluation"
    assert payload["processor"]["revision"] == "processor-r1"
    assert payload["records"][0]["metrics"]["causal_behavior"] == {
        "target_success": 1.0
    }
    assert json.loads(destination.read_text()) == payload


def test_multimodal_evaluation_rejects_non_finite_metrics():
    with pytest.raises(ValueError, match="must be finite"):
        EvaluationMetricGroups(causal_behavior={"target_success": float("nan")})
