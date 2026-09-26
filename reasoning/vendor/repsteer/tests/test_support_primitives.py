from __future__ import annotations

import enum
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from repsteer.core import UnsupportedArchitectureError
from repsteer.data import canonical_json, stable_fingerprint
from repsteer.evaluation import (
    CallableMetric,
    ContainsText,
    EvaluationBatch,
    MeanOutputLength,
)
from repsteer.models.hf.adapters import AdapterRegistry
from repsteer.sae import (
    FunctionalSAEAdapter,
    available_providers,
    load,
    register_provider,
    unregister_provider,
)


class _Mode(enum.Enum):
    READY = "ready"


def test_fingerprints_are_canonical_for_common_runtime_values():
    left = {
        "tensor": torch.tensor([[1, 2]], dtype=torch.int64),
        "path": Path("artifacts/example"),
        "mode": _Mode.READY,
        "zero": -0.0,
        "nested": {"b": 2, "a": 1},
    }
    right = {
        "nested": {"a": 1, "b": 2},
        "zero": 0.0,
        "mode": _Mode.READY,
        "path": Path("artifacts/example"),
        "tensor": torch.tensor([[1, 2]], dtype=torch.int64),
    }

    assert canonical_json(left) == canonical_json(right)
    assert stable_fingerprint(left) == stable_fingerprint(right)


def test_incremental_metrics_reset_and_handle_text_or_token_ids():
    batch = EvaluationBatch(
        prompt="prompt",
        result=SimpleNamespace(
            text="Hello WORLD",
            token_ids=torch.tensor([[1, 2, 3, 4]]),
        ),
        configuration={},
    )
    text_only = EvaluationBatch(
        prompt="prompt",
        result=SimpleNamespace(text="one two"),
        configuration={},
    )

    contains = ContainsText("world")
    contains.update(batch)
    assert contains.compute() == {"contains": 1.0}
    contains.reset()
    assert math.isnan(contains.compute()["contains"])

    callable_metric = CallableMetric("score", lambda item: len(str(item.prompt)))
    callable_metric.update(batch)
    callable_metric.update(text_only)
    assert callable_metric.compute() == {"score": 6.0}

    length = MeanOutputLength()
    length.update(batch)
    length.update(text_only)
    assert length.compute() == {"output_length": 3.0}


def test_custom_sae_provider_registry_normalizes_names_and_validates_results():
    provider = "unit-support-provider"
    unregister_provider(provider)
    sae = FunctionalSAEAdapter(
        2,
        2,
        encode_fn=lambda value: value,
        decode_fn=lambda value: value,
        decoder_direction_fn=lambda feature_id: torch.eye(2)[feature_id],
    )
    try:
        register_provider("Unit-Support_Provider", lambda **_kwargs: sae)
        assert "unitsupportprovider" in available_providers()
        assert load(provider=provider) is sae
        with pytest.raises(ValueError, match="already registered"):
            register_provider(provider, lambda **_kwargs: sae)
        with pytest.raises(ValueError, match="Unknown SAE provider"):
            load(provider="missing-provider")
    finally:
        unregister_provider(provider)


class _RegistryModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(model_type="registry-test")


class _RegistryAdapter:
    def __init__(self, name: str, supported: bool) -> None:
        self.architecture_name = name
        self.supported = supported

    def supports(self, model: nn.Module) -> bool:
        del model
        return self.supported

    def resolve(self, model, site):
        del model, site
        raise NotImplementedError

    def hidden_size(self, model, site=None) -> int:
        del model, site
        return 1


def test_adapter_registry_honors_precedence_and_has_actionable_failures():
    fallback = _RegistryAdapter("fallback", True)
    preferred = _RegistryAdapter("preferred", True)
    registry = AdapterRegistry([fallback])
    registry.register(preferred, prepend=True)

    assert registry.resolve(_RegistryModel()) is preferred
    registry.unregister(preferred)
    assert registry.resolve(_RegistryModel()) is fallback

    with pytest.raises(UnsupportedArchitectureError, match="registry-test"):
        AdapterRegistry().resolve(_RegistryModel())
