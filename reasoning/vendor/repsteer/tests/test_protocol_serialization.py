"""CPU-only protocol serialization contracts.

These tests deliberately exercise only configuration objects.  They do not
construct a Transformers model or serialize model-weight tensors.
"""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pytest
import torch

from repsteer.artifacts import ArtifactMetadata, DirectionArtifact
from repsteer.core import Intervention, Site, SteeringPlan
from repsteer.core.errors import PositionResolutionError
from repsteer.core.serialization import json_safe
from repsteer.gates import Always, AndGate, NotGate, OrGate, gate_from_dict
from repsteer.operators import (
    Ablate,
    Add,
    Clamp,
    ITIAdd,
    RemoveProjection,
    Replace,
    SAEAblate,
    SAEClamp,
    Subtract,
    operator_from_dict,
)
from repsteer.positions import (
    AllTokens,
    GeneratedTokens,
    ImagePatches,
    ImageTokens,
    LastNonPaddingToken,
    LastPromptToken,
    ObjectPatches,
    PromptTokens,
    SpecialToken,
    TextSpan,
    TokenIndices,
    position_selector_from_dict,
)
from repsteer.schedules import Constant, NormRelative, strength_schedule_from_dict


@pytest.mark.parametrize(
    "unit",
    [None, 3, slice(-4, None, 2), (1, 4)],
)
def test_site_round_trips_every_supported_unit_encoding(unit):
    site = Site(
        "language",
        "resid_post",
        layer=2,
        unit=unit,
        io="input",
        tensor_path=("hidden_states", 0),
    )

    document = site.to_dict()

    assert Site.from_dict(document) == site
    assert json.loads(json.dumps(document)) == document


def test_site_from_dict_accepts_the_legacy_list_unit_encoding():
    site = Site.from_dict(
        {
            "stream": "vision",
            "component": "vision_resid",
            "layer": 1,
            "unit": [0, 2],
            "io": "output",
            "tensor_path": ["hidden_states", 0],
        }
    )

    assert site.unit == (0, 2)
    assert site.tensor_path == ("hidden_states", 0)


def test_position_selector_round_trips_are_json_safe():
    selectors = (
        LastNonPaddingToken(),
        LastPromptToken(),
        PromptTokens(),
        GeneratedTokens(),
        AllTokens(),
        TokenIndices((-2, 0, 4), strict=True),
        TextSpan("HELLO", occurrence="all", case_sensitive=False),
        SpecialToken("<image>", token_id=42),
        ImageTokens(image_index=None),
        ImagePatches([[True, False], [False, True]], image_index=1),
        ObjectPatches(((0.0, 0.0, 0.5, 0.5),), image_index=None),
    )

    for selector in selectors:
        document = selector.to_dict()
        restored = position_selector_from_dict(document)

        assert restored.to_dict() == document
        assert json.loads(json.dumps(json_safe(document))) == document


def test_position_selector_deserialization_fails_closed_for_unknown_type():
    with pytest.raises(PositionResolutionError, match="Unknown position selector"):
        position_selector_from_dict({"type": "arbitrary_python"})


def test_structural_gate_round_trip_and_invalid_nested_gate_rejection():
    gate = AndGate(Always(), NotGate(OrGate(Always(), Always())))

    restored = gate_from_dict(gate.to_dict())

    assert restored.to_dict() == gate.to_dict()
    assert isinstance(gate_from_dict({"type": "ALWAYS"}), Always)
    with pytest.raises(ValueError, match="nested 'gate'"):
        gate_from_dict({"type": "not"})
    with pytest.raises(ValueError, match="Unknown gate type"):
        gate_from_dict({"type": "callable"})


@pytest.mark.parametrize(
    "operator",
    (
        Add(),
        Subtract(),
        RemoveProjection(eps=1e-6),
        Replace(),
        Clamp(0.25, feature_id=3),
        Ablate(feature_id=2),
        ITIAdd(layer=3),
        SAEClamp(-0.5, feature_id=1),
        SAEAblate(feature_id=0),
    ),
)
def test_builtin_operator_configs_round_trip_without_runtime_handles(operator):
    document = operator.to_dict()

    restored = operator_from_dict(document)

    assert restored.to_dict() == document


def test_operator_deserialization_rejects_runtime_sae_handles_and_unknown_types():
    with pytest.raises(ValueError, match="rebound explicitly"):
        operator_from_dict(
            {
                "type": "sae_clamp",
                "value": 1.0,
                "feature_id": 0,
                "sae_adapter": "example.RuntimeAdapter",
            }
        )
    with pytest.raises(ValueError, match="Unknown operator type"):
        operator_from_dict({"type": "arbitrary_python"})


@pytest.mark.parametrize(
    "schedule",
    (
        Constant(-0.75),
        NormRelative(ratio=0.25, p=1.5, eps=1e-4),
    ),
)
def test_builtin_strength_schedule_configs_round_trip(schedule):
    document = schedule.to_dict()

    restored = strength_schedule_from_dict(document)

    assert restored.to_dict() == document


def test_strength_schedule_deserialization_accepts_value_alias_and_rejects_unknown():
    assert strength_schedule_from_dict({"type": "constant", "value": 0.5}) == Constant(
        0.5
    )
    with pytest.raises(ValueError, match="Unknown strength schedule"):
        strength_schedule_from_dict({"type": "arbitrary_python"})


class _Mode(Enum):
    CPU = "cpu"


@dataclass(frozen=True)
class _NestedConfig:
    path: Path
    mode: _Mode
    values: tuple[int, int]


def test_json_safe_converts_core_configuration_primitives_without_pickle():
    document = json_safe(
        {
            "site": Site("language", "resid_post", 0, unit=slice(1, None, 2)),
            "dtype": torch.float32,
            "device": torch.device("cpu"),
            "tensor": torch.tensor([1.0, 2.0]),
            "set": {"z", "a"},
            "nested": _NestedConfig(Path("artifacts/direction"), _Mode.CPU, (1, 2)),
            7: "numeric mapping keys become strings",
        }
    )

    assert document["site"]["unit"] == {
        "kind": "slice",
        "start": 1,
        "stop": None,
        "step": 2,
    }
    assert document["dtype"] == "torch.float32"
    assert document["device"] == "cpu"
    assert document["tensor"] == [1.0, 2.0]
    assert document["set"] == ["a", "z"]
    assert document["nested"] == {
        "path": "artifacts/direction",
        "mode": "cpu",
        "values": [1, 2],
    }
    assert document["7"] == "numeric mapping keys become strings"
    json.dumps(document, allow_nan=False)


def test_json_safe_rejects_large_tensors_and_unsupported_values():
    with pytest.raises(TypeError, match="Large tensors"):
        json_safe(torch.zeros(1025))
    with pytest.raises(TypeError, match="not safely JSON serializable"):
        json_safe(object())


def test_plan_snapshot_composes_only_portable_protocol_descriptions():
    site = Site("language", "resid_post", 0)
    artifact = DirectionArtifact(
        ArtifactMetadata(
            model_id="fake/model",
            model_revision="revision-1",
            architecture="llama",
            site=site,
            hidden_size=2,
            method="unit",
        ),
        torch.tensor([1.0, -1.0]),
    )
    intervention = Intervention(
        artifact=artifact,
        operator=Add(),
        positions=TokenIndices((0, -1), strict=True),
        strength=NormRelative(0.5),
        gate=AndGate(Always(), NotGate(OrGate(Always(), Always()))),
        site=site,
        phase="prefill",
        priority=7,
    )

    document = SteeringPlan((intervention,)).to_dict()
    entry = document["interventions"][0]

    assert (
        position_selector_from_dict(entry["positions"]).to_dict() == entry["positions"]
    )
    assert operator_from_dict(entry["operator"]).to_dict() == entry["operator"]
    assert strength_schedule_from_dict(entry["strength"]).to_dict() == entry["strength"]
    assert gate_from_dict(entry["gate"]).to_dict() == entry["gate"]
    assert Site.from_dict(entry["site"]) == site
    json.dumps(json_safe(document), allow_nan=False)
