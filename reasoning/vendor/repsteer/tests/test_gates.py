import torch

from repsteer.artifacts import ArtifactMetadata, ProbeArtifact
from repsteer.core import StepContext
from repsteer.gates import (
    AndGate,
    CallableGate,
    CosineGate,
    NotGate,
    OrGate,
    ProbeGate,
    SAEActivationGate,
)
from repsteer.positions import LastPromptToken
from repsteer.sae import FunctionalSAEAdapter, feature_artifact


def _context(activation, *, metadata_key="activation"):
    return StepContext(
        phase="prefill",
        prompt_lengths=torch.tensor([activation.shape[1]] * activation.shape[0]),
        attention_mask=torch.ones(activation.shape[:2]),
        metadata={metadata_key: activation},
    )


def test_probe_gate_thresholds_sequence_probabilities_and_positions():
    probe = ProbeArtifact(
        ArtifactMetadata(method="linear_probe"),
        weight=torch.tensor([1.0, 0.0]),
        bias=0.0,
    )
    activation = torch.tensor(
        [
            [[-10.0, 0.0], [2.0, 0.0]],
            [[10.0, 0.0], [-2.0, 0.0]],
        ]
    )
    gate = ProbeGate(
        probe,
        threshold=0.75,
        positions=LastPromptToken(),
        reduction="last",
    )

    assert torch.equal(
        gate.evaluate(_context(activation)),
        torch.tensor([True, False]),
    )


def test_sae_activation_gate_encodes_residuals_or_accepts_latents():
    encoder = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    decoder = torch.eye(2)
    sae = FunctionalSAEAdapter(
        2,
        2,
        encode_fn=lambda x: x @ encoder,
        decode_fn=lambda z: z @ decoder,
        decoder_direction_fn=lambda feature_id: decoder[feature_id],
    )
    feature = feature_artifact(sae, 0)
    residual = torch.tensor(
        [
            [[0.0, 1.0], [2.0, 0.0]],
            [[0.1, 0.0], [0.2, 0.0]],
        ]
    )

    encoded_gate = SAEActivationGate(feature, threshold=1.0, sae=sae)
    latent_gate = SAEActivationGate(feature, threshold=1.0)

    assert torch.equal(
        encoded_gate.evaluate(_context(residual)),
        torch.tensor([True, False]),
    )
    assert torch.equal(
        latent_gate.evaluate(_context(residual, metadata_key="sae_latents")),
        torch.tensor([True, False]),
    )


def test_cosine_and_logical_gates_preserve_boolean_or_soft_semantics():
    activation = torch.tensor(
        [
            [[1.0, 0.0]],
            [[-1.0, 0.0]],
        ]
    )
    context = _context(activation)
    cosine = CosineGate(torch.tensor([1.0, 0.0]), threshold=0.5)
    truth = CallableGate(lambda _: torch.tensor([True, True]), name="truth")
    soft_a = CallableGate(lambda _: torch.tensor([0.2, 0.8]))
    soft_b = CallableGate(lambda _: torch.tensor([0.6, 0.4]))

    assert torch.equal(cosine.evaluate(context), torch.tensor([True, False]))
    assert torch.equal(
        AndGate(cosine, truth).evaluate(context),
        torch.tensor([True, False]),
    )
    assert torch.equal(
        OrGate(cosine, NotGate(truth)).evaluate(context),
        torch.tensor([True, False]),
    )
    assert torch.equal(
        AndGate(soft_a, soft_b).evaluate(context),
        torch.tensor([0.2, 0.4]),
    )
    assert torch.equal(
        OrGate(soft_a, soft_b).evaluate(context),
        torch.tensor([0.6, 0.8]),
    )
