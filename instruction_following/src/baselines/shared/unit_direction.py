"""Shared machinery for methods that steer along a unit direction.

Several methods differ only in how they estimate a direction — a probe's
separating normal, a principal component, an SAE decoder column — and agree on
everything after that: the artifact holds one unit vector per hidden-state
index plus a per-index scale, the intervention adds ``alpha * scale * unit``
at one layer, and the dev search is the reference two-stage one over layer and
alpha. That agreement lives here so a new estimator only writes its estimator.

Scale is per-index because activation magnitudes differ by layer; expressing
alpha in scale units is what makes alpha mean the same thing at layer 9 and
layer 33.
"""

from pathlib import Path

import torch

from kernel.config import BASELINES, slug
from baselines.shared.base import SearchPlan, joint_plan
from baselines.shared.residual_add import add_direction, check_layer

# Three normalized depths and three strengths: a reference default, not a
# benchmark restriction. Three of each fills the budget as a joint grid, which
# is what lets the search avoid assuming that the best site at one strength is
# the best site at every strength.
DEFAULT_LAYER_FRACTIONS = [1 / 4, 7 / 12, 11 / 12]
DEFAULT_ALPHAS = [1.0, 2.0, 4.0]
DEFAULT_BUDGET = 9


def artifact_path(model_slug: str, method_key: str, task: str) -> Path:
    return BASELINES / model_slug / method_key / f"{task}.pt"


def default_layers(n_layers: int) -> list[int]:
    return list(dict.fromkeys(
        max(1, min(n_layers, round(n_layers * fraction)))
        for fraction in DEFAULT_LAYER_FRACTIONS
    ))


def config_name(config: dict) -> str:
    return f"L{config['layer']}_a{config['alpha']}"


def layer_alpha_plan(context: dict, overrides: dict | None,
                     budget: int | None, notes: str) -> SearchPlan:
    """The reference joint search over (layer, alpha)."""
    overrides = overrides or {}
    n_layers = context["n_layers"]
    layers = overrides.get("layer") or default_layers(n_layers)
    alphas = overrides.get("alpha") or list(DEFAULT_ALPHAS)

    if len(set(layers)) != len(layers):
        raise ValueError(f"layers contain duplicates: {layers}")
    invalid = [layer for layer in layers if not 1 <= layer <= n_layers]
    if invalid:
        raise ValueError(
            f"layers must be between 1 and {n_layers}, got {invalid}"
        )
    if len(set(alphas)) != len(alphas):
        raise ValueError(f"alphas contain duplicates: {alphas}")

    return joint_plan(
        "layer", list(layers), "alpha", list(alphas),
        budget or DEFAULT_BUDGET, notes=notes,
    )


EXTRACTION_POSITION = "last_prompt_token"


def validate_artifact(record, path: Path, method_name: str, task: str,
                      model_slug: str, scale_key: str = "sigma") -> dict:
    """Check the identity and shape contract shared by unit-direction methods.

    Indices the producer recorded as contrast-free must hold zero directions,
    and every other index must hold a unit vector: a zero direction that was
    not deliberate would steer by nothing while looking like it worked.
    """
    if not isinstance(record, dict):
        raise ValueError(f"Artifact must be a dictionary: {path}")
    if record.get("method") != method_name:
        raise ValueError(f"Artifact at {path} is not {method_name}")
    if record.get("task") != task:
        raise ValueError(
            f"Artifact task {record.get('task')!r} does not match {task!r}"
        )
    artifact_model = record.get("model")
    if not isinstance(artifact_model, str) or slug(artifact_model) != model_slug:
        raise ValueError(
            f"Artifact model {artifact_model!r} does not match {model_slug!r}"
        )
    # Where the activations were read. Recorded by every producer and, until
    # this check, read by nobody — so an artifact captured somewhere else
    # loaded clean and steered with a direction learned at a different site.
    if record.get("extraction_position") != EXTRACTION_POSITION:
        raise ValueError(
            f"Artifact was captured at "
            f"{record.get('extraction_position')!r}, not "
            f"{EXTRACTION_POSITION!r}; produce it again: {path}"
        )
    direction = record.get("direction")
    scale = record.get(scale_key)
    if not isinstance(direction, torch.Tensor) or direction.ndim != 2:
        raise ValueError(
            f"Direction must have shape [layers+1, hidden]: {path}"
        )
    if not isinstance(scale, torch.Tensor) or scale.shape != direction.shape[:1]:
        raise ValueError(
            f"{scale_key} must have one value per hidden-state index: {path}"
        )

    # Two reasons an index can hold no direction: the activations there carry
    # no contrast to learn from, or the baseline was not allowed to use it —
    # SAE may only steer layers its reconstruction gate admitted.
    norms = torch.linalg.vector_norm(direction.float(), dim=-1)
    unusable = set(record.get("indices_without_contrast", [])) | set(
        record.get("indices_without_direction", [])
    )
    steerable = [
        index for index in range(norms.shape[0]) if index not in unusable
    ]
    if not torch.allclose(
            norms[steerable], torch.ones(len(steerable)), atol=1e-3):
        raise ValueError(f"Directions must be unit vectors: {path}")
    if unusable and not torch.allclose(
            norms[sorted(unusable)], torch.zeros(len(unusable))):
        raise ValueError(
            f"Indices recorded as unusable must hold zero directions: {path}"
        )
    return record


def delta(artifact: dict, config: dict, scale_key: str = "sigma"):
    """The vector this configuration adds: ``alpha * scale * unit``."""
    layer = config["layer"]
    return (
        config["alpha"]
        * float(artifact[scale_key][layer])
        * artifact["direction"][layer]
    )


def make_intervention(model, artifact: dict, config: dict,
                      scale_key: str = "sigma"):
    """Return a zero-argument callable producing the intervention context."""
    n_layers = check_layer(model, config["layer"])
    expected = (n_layers + 1, int(model.config.hidden_size))
    shape = tuple(artifact["direction"].shape)
    if shape != expected:
        raise ValueError(f"direction must have shape {expected}; got {shape}")
    return lambda: add_direction(
        model, delta(artifact, config, scale_key), config["layer"]
    )


def dose_metadata(artifact: dict, config: dict, method_key: str,
                  scale_key: str = "sigma") -> dict:
    """Strength facts that do not need a forward pass."""
    layer = config["layer"]
    scale = float(artifact[scale_key][layer])
    # Measured, not assumed: an index the producer recorded as unusable holds
    # a zero direction, and reporting 1.0 there would describe a dose that was
    # never applied.
    norm = float(torch.linalg.vector_norm(artifact["direction"][layer].float()))
    return {
        "method": method_key,
        "raw_strength": float(config["alpha"]),
        "projection_sigma": scale,
        "direction_norm": norm,
        "effective_delta_norm": abs(float(config["alpha"])) * scale * norm,
        "site": {"component": "resid_post", "layer": int(layer)},
    }
