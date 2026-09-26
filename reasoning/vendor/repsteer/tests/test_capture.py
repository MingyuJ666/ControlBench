import torch

from repsteer.capture import (
    ActivationBatch,
    CaptureRequest,
    MemoryStore,
    capture_activations,
)


class _VariableLengthCaptureModel:
    model_id = "tiny/batched-capture"
    revision = "r1"

    def __init__(self):
        self.calls = []

    def capture(self, request):
        self.calls.append(tuple(request.inputs))
        lengths = [int(value) for value in request.inputs]
        width = max(lengths)
        activations = torch.zeros(len(lengths), width, 2)
        attention_mask = torch.zeros(len(lengths), width, dtype=torch.bool)
        for row, length in enumerate(lengths):
            activations[row, :length] = float(length)
            attention_mask[row, :length] = True
        return ActivationBatch(
            activations,
            request=request,
            attention_mask=attention_mask,
        )


def test_capture_request_batch_size_is_executed_and_ragged_chunks_are_padded():
    model = _VariableLengthCaptureModel()
    store = MemoryStore()
    request = CaptureRequest(
        inputs=[2, 4, 1, 3, 2],
        site="fake.site",
        positions="backend-owned",
        pooling="identity",
        batch_size=2,
    )

    result = capture_activations(model, request, store=store)
    repeated = capture_activations(model, request, store=store)

    assert model.calls == [(2, 4), (1, 3), (2,)]
    assert result.activations.shape == (5, 4, 2)
    assert result.attention_mask.tolist() == [
        [True, True, False, False],
        [True, True, True, True],
        [True, False, False, False],
        [True, True, True, False],
        [True, True, False, False],
    ]
    assert result.metadata["capture_batches"] == 3
    assert repeated is result


class _PoolingCaptureModel:
    model_id = "tiny/callable-pooling"
    revision = "r1"

    def __init__(self):
        self.calls = 0

    def capture(self, request):
        self.calls += 1
        return ActivationBatch(
            torch.tensor([[[1.0], [2.0], [3.0]]]),
            request=request,
        )


def test_callable_pooling_implementations_have_distinct_cache_keys():
    model = _PoolingCaptureModel()
    store = MemoryStore()
    base = {
        "inputs": ["sample"],
        "site": "fake.site",
        "positions": "backend-owned",
    }
    mean_request = CaptureRequest(
        **base,
        pooling=lambda activation: activation.mean(dim=1),
    )
    max_request = CaptureRequest(
        **base,
        pooling=lambda activation: activation.amax(dim=1),
    )

    mean = capture_activations(model, mean_request, store=store)
    maximum = capture_activations(model, max_request, store=store)

    assert mean_request.fingerprint != max_request.fingerprint
    assert mean.activations.item() == 2.0
    assert maximum.activations.item() == 3.0
    assert model.calls == 2
