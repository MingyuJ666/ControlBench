"""CPU-only tests for capture pooling, result coercion, and cache semantics."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from repsteer.capture import (
    ActivationBatch,
    CaptureRequest,
    CaptureRunner,
    MemoryStore,
    capture_activations,
    pool_activations,
    resolve_pooling,
)


def _request(**overrides: Any) -> CaptureRequest:
    values: dict[str, Any] = {
        "inputs": ("first", "second"),
        "site": "test.site",
        "positions": "backend-owned",
    }
    values.update(overrides)
    return CaptureRequest(**values)


def _sequence_activations() -> torch.Tensor:
    return torch.tensor(
        [
            [[1.0, 10.0], [2.0, 20.0], [100.0, 1000.0]],
            [[-4.0, 4.0], [-2.0, 2.0], [-100.0, -100.0]],
        ]
    )


def test_builtin_pooling_has_expected_mask_weight_and_span_semantics():
    activations = _sequence_activations()
    attention_mask = torch.tensor([[True, True, False], [False, True, True]])

    assert torch.equal(
        pool_activations(activations, "identity"),
        activations,
    )
    assert torch.equal(
        pool_activations(
            activations,
            "last_non_padding_token",
            attention_mask=attention_mask,
        ),
        torch.tensor([[2.0, 20.0], [-100.0, -100.0]]),
    )
    assert torch.allclose(
        pool_activations(activations, "mean", attention_mask=attention_mask),
        torch.tensor([[1.5, 15.0], [-51.0, -49.0]]),
    )
    assert torch.equal(
        pool_activations(activations, "max", attention_mask=attention_mask),
        torch.tensor([[2.0, 20.0], [-2.0, 2.0]]),
    )
    assert torch.allclose(
        pool_activations(
            activations,
            "weighted_mean",
            attention_mask=attention_mask,
            token_weights=torch.tensor([[1.0, 3.0, 100.0], [10.0, 2.0, 1.0]]),
        ),
        torch.tensor([[1.75, 17.5], [-34.666666, -32.0]]),
    )
    assert torch.allclose(
        pool_activations(
            activations,
            "span_mean",
            spans=torch.tensor([[-3, -1], [1, 3]]),
        ),
        torch.tensor([[1.5, 15.0], [-51.0, -49.0]]),
    )


def test_pooling_resolution_and_invalid_inputs_fail_closed():
    activations = _sequence_activations()

    assert type(resolve_pooling(None)).__name__ == "IdentityPooling"
    assert type(resolve_pooling("last token")).__name__ == "LastTokenPooling"
    assert type(resolve_pooling("weighted")).__name__ == "WeightedMeanPooling"
    assert type(resolve_pooling("span")).__name__ == "SpanMeanPooling"

    with pytest.raises(ValueError, match="unknown activation pooling"):
        resolve_pooling("median")
    with pytest.raises(TypeError, match="pooling must"):
        resolve_pooling(object())
    with pytest.raises(ValueError, match="requires token_weights"):
        pool_activations(activations, "weighted_mean")
    with pytest.raises(ValueError, match="non-zero sum"):
        pool_activations(
            activations,
            "weighted_mean",
            token_weights=torch.zeros(2, 3),
        )
    with pytest.raises(ValueError, match="invalid half-open token span"):
        pool_activations(activations, "span_mean", spans=[(0, 0), (1, 3)])
    with pytest.raises(ValueError, match="no selected tokens"):
        pool_activations(
            activations,
            "mean",
            attention_mask=torch.tensor([[False, False, False], [True, False, False]]),
        )


def test_activation_batch_pool_clone_to_and_detach_are_tensor_safe():
    request = _request(pooling="weighted_mean", sample_weights=[0.25, 0.75])
    activations = torch.tensor([[[1.0], [3.0]], [[5.0], [9.0]]], requires_grad=True)
    batch = ActivationBatch(
        activations,
        request=request,
        attention_mask=torch.tensor([[True, True], [True, False]]),
        token_ids=torch.tensor([[11, 12], [13, 14]]),
        sample_metadata=[{"name": "first"}, {"name": "second"}],
        metadata={"token_weights": [[1.0, 3.0], [1.0, 100.0]]},
    )

    pooled = batch.pool()
    clone = batch.clone()
    converted = batch.to(dtype=torch.float64)
    detached = batch.detach()

    assert pooled.pooled
    assert torch.allclose(pooled.activations, torch.tensor([[2.5], [5.0]]))
    assert pooled.weights is not None
    assert torch.equal(pooled.weights, torch.tensor([0.25, 0.75]))
    assert converted.dtype == torch.float64
    assert detached.activations.requires_grad is False
    assert batch.activations.requires_grad is True

    with torch.no_grad():
        clone.activations[0, 0, 0] = -99.0
    clone.attention_mask[0, 0] = False
    clone.token_ids[0, 0] = -1
    assert clone.weights is not None
    clone.weights[0] = 99.0

    assert batch.activations[0, 0, 0].item() == 1.0
    assert batch.attention_mask[0, 0].item() is True
    assert batch.token_ids[0, 0].item() == 11
    assert batch.weights is not None
    assert batch.weights[0].item() == pytest.approx(0.25)


def _batch(value: float) -> ActivationBatch:
    return ActivationBatch(torch.tensor([[value]]))


def test_memory_store_evicts_least_recently_used_entry_and_can_clone_reads():
    store = MemoryStore(max_entries=2)
    first, second, third = _batch(1.0), _batch(2.0), _batch(3.0)

    store.put("first", first)
    store.put("second", second)
    assert store.get("first") is first
    store.put("third", third)

    assert store.get("second") is None
    assert store.get("first") is first
    assert store.get("third") is third
    assert len(store) == 2

    cloning_store = MemoryStore(clone_on_read=True)
    source = ActivationBatch(
        torch.tensor([[[1.0]]]),
        attention_mask=torch.tensor([[True]]),
        token_ids=torch.tensor([[5]]),
        sample_weights=torch.tensor([1.0]),
    )
    cloning_store.put("activation", source)
    copy = cloning_store.get("activation")

    assert copy is not None
    assert copy is not source
    copy.activations[0, 0, 0] = -1.0
    copy.attention_mask[0, 0] = False
    copy.token_ids[0, 0] = -1
    assert copy.weights is not None
    copy.weights[0] = 2.0

    assert source.activations.item() == 1.0
    assert source.attention_mask.item() is True
    assert source.token_ids.item() == 5
    assert source.weights is not None
    assert source.weights.item() == 1.0


class _TensorCaptureModel:
    model_id = "test/tensor-capture"
    revision = "r1"

    def capture(self, request: CaptureRequest) -> torch.Tensor:
        assert tuple(request.inputs) == ("first", "second")
        return torch.tensor([[[1.0], [3.0]], [[5.0], [7.0]]], requires_grad=True)


class _MappingCaptureModel:
    model_id = "test/mapping-capture"
    revision = "r1"

    def capture(self, _request: CaptureRequest) -> dict[str, Any]:
        return {
            "activation": torch.tensor([[[1.0]], [[2.0]]]),
            "mask": torch.tensor([[True], [True]]),
            "input_ids": torch.tensor([[10], [20]]),
            "sample_metadata": [{"source": "a"}, {"source": "b"}],
            "metadata": {"format": "mapping"},
        }


class _BatchSequenceCaptureModel:
    model_id = "test/batch-sequence-capture"
    revision = "r1"

    def capture(self, _request: CaptureRequest) -> list[ActivationBatch]:
        return [
            ActivationBatch(
                torch.tensor([[[1.0], [2.0]]]),
                attention_mask=torch.tensor([[True, True]]),
            ),
            ActivationBatch(
                torch.tensor([[[3.0], [4.0], [5.0]]]),
                attention_mask=torch.tensor([[True, True, True]]),
            ),
        ]


class _AlternateCaptureModel:
    model_id = "test/alternate-capture"
    revision = "r1"

    def capture_activations(
        self, _request: CaptureRequest
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor([[[4.0]]]), torch.tensor([[True]])


class _InvalidCaptureModel:
    model_id = "test/invalid-capture"
    revision = "r1"

    def __init__(self, result: Any) -> None:
        self.result = result

    def capture(self, _request: CaptureRequest) -> Any:
        return self.result


def test_capture_runner_coerces_tensor_mapping_and_batch_sequence_results():
    tensor_result = CaptureRunner().capture(
        _TensorCaptureModel(), _request(pooling="mean", sample_weights=[1.0, 2.0])
    )
    mapping_result = CaptureRunner().capture(
        _MappingCaptureModel(), _request(sample_weights=[0.5, 1.5])
    )
    sequence_result = CaptureRunner().capture(_BatchSequenceCaptureModel(), _request())

    assert tensor_result.pooled
    assert tensor_result.activations.tolist() == [[2.0], [6.0]]
    assert tensor_result.activations.requires_grad is False
    assert tensor_result.weights is not None
    assert tensor_result.weights.tolist() == [1.0, 2.0]

    assert mapping_result.pooled
    assert mapping_result.metadata["format"] == "mapping"
    assert mapping_result.attention_mask.tolist() == [[True], [True]]
    assert mapping_result.token_ids.tolist() == [[10], [20]]
    assert mapping_result.sample_metadata == ({"source": "a"}, {"source": "b"})
    assert mapping_result.weights is not None
    assert mapping_result.weights.tolist() == [0.5, 1.5]

    assert sequence_result.pooled
    assert sequence_result.activations.shape == (2, 3, 1)
    assert sequence_result.activations[:, :, 0].tolist() == [
        [1.0, 2.0, 0.0],
        [3.0, 4.0, 5.0],
    ]
    assert sequence_result.attention_mask.tolist() == [
        [True, True, False],
        [True, True, True],
    ]


def test_capture_runner_uses_alternate_method_and_reports_invalid_results():
    alternate = capture_activations(
        _AlternateCaptureModel(),
        inputs=["only"],
        site="test.site",
        positions="backend-owned",
    )

    assert alternate.pooled
    assert alternate.activations.tolist() == [[[4.0]]]
    assert alternate.attention_mask.tolist() == [[True]]

    with pytest.raises(TypeError, match=r"capture\(request\) or capture_activations"):
        CaptureRunner().capture(object(), _request())
    with pytest.raises(ValueError, match="contain 'activations'"):
        CaptureRunner().capture(
            _InvalidCaptureModel({"mask": torch.ones(2, 1)}), _request()
        )
    with pytest.raises(TypeError, match="must return ActivationBatch"):
        CaptureRunner().capture(_InvalidCaptureModel(object()), _request())
