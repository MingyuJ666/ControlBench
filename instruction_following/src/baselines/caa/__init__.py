"""CAA: steer with the mean difference between contrastive activations.

The artifact stores one direction per hidden-state index, ``[n_layers+1,
hidden]``. Strength multiplies the raw mean-difference vector: this method
deliberately keeps the magnitude the extraction produced, following Rimsky et
al. and Tan et al., rather than normalizing to unit length. Its useful alpha is
therefore an empirical property selected on dev, not a semantic unit — which is
why cross-method figures use measured dose metadata instead of alpha.
"""

from pathlib import Path

import torch

from kernel.config import BASELINES, slug
from baselines.shared import unit_direction
from baselines.shared.base import SearchPlan
from baselines.shared.residual_add import add_direction, check_layer

KEY = "caa"
METHOD_VERSION = 1
ARTIFACT_METHOD = "caa_diff_in_means"
# Where the activations were read, as opposed to where a direction is
# injected — the two are different questions and used to share a name.
EXTRACTION_POSITION = "last_prompt_token"

STRUCTURAL_AXIS = "layer"
STRENGTH_AXIS = "alpha"
# Five normalized depths: a reference default, not a benchmark restriction.



def check_available() -> str | None:
    """Return None when the method can run, else why it cannot."""
    return None


def artifact_path(model_slug: str, task: str) -> Path:
    return unit_direction.artifact_path(model_slug, KEY, task)


def load_artifact(model_slug: str, task: str) -> dict:
    """Load a CAA artifact after checking its minimal identity contract."""
    path = artifact_path(model_slug, task)
    record = torch.load(path, weights_only=True)
    if not isinstance(record, dict):
        raise ValueError(f"CAA artifact must be a dictionary: {path}")
    if record.get("method") != ARTIFACT_METHOD:
        raise ValueError(f"Artifact at {path} is not a CAA mean-difference")
    if record.get("task") != task:
        raise ValueError(
            f"Artifact task {record.get('task')!r} does not match {task!r}"
        )
    artifact_model = record.get("model")
    if not isinstance(artifact_model, str) or slug(artifact_model) != model_slug:
        raise ValueError(
            f"Artifact model {artifact_model!r} does not match {model_slug!r}"
        )
    vector = record.get("vector")
    if not isinstance(vector, torch.Tensor) or vector.ndim != 2:
        raise ValueError(
            f"CAA vector must have shape [layers+1, hidden]: {path}"
        )
    # The other four baselines check this through validate_artifact. Without
    # it a vector containing NaN loads without complaint and only shows up as
    # empty generations hours later, with nothing pointing at the artifact.
    if not torch.isfinite(vector).all():
        raise ValueError(f"CAA vector contains NaN or Inf: {path}")
    if record.get("extraction_position") != EXTRACTION_POSITION:
        raise ValueError(
            f"Artifact was captured at "
            f"{record.get('extraction_position')!r}, not "
            f"{EXTRACTION_POSITION!r}; produce it again: {path}"
        )
    return record


def config_name(config: dict) -> str:
    return f"L{config['layer']}_a{config['alpha']}"



def search_plan(context: dict, overrides: dict | None = None,
                budget: int | None = None) -> SearchPlan:
    """The reference two-stage search, which this baseline does not vary.

    The grid and its validation live in ``baselines.shared.unit_direction``
    with the other residual-stream baselines. A second copy here held the
    same numbers under different names, so changing the reference grid in one
    place would have left this baseline quietly searching the old one.
    """
    return unit_direction.layer_alpha_plan(
        context, overrides, budget,
        notes="every candidate layer paired with every strength",
    )


def delta(artifact: dict, config: dict) -> torch.Tensor:
    """Return the vector this configuration adds to the residual stream."""
    return config["alpha"] * artifact["vector"][config["layer"]]


def make_intervention(model, artifact: dict, config: dict):
    """Return a zero-argument callable producing the intervention context."""
    vector = artifact["vector"]
    # Validate the layer before indexing the vector, so an out-of-range layer
    # reports itself instead of raising IndexError from the tensor.
    n_layers = check_layer(model, config["layer"])
    expected = (n_layers + 1, int(model.config.hidden_size))
    if tuple(vector.shape) != expected:
        raise ValueError(
            f"vector must have shape {expected}; got {tuple(vector.shape)}"
        )
    return lambda: add_direction(
        model, delta(artifact, config), config["layer"]
    )


def dose_metadata(artifact: dict, config: dict) -> dict:
    """Strength facts that do not need a forward pass.

    ``residual_rms_ratio`` needs activations and is filled in by the analysis
    utility that measures them; it is absent here rather than guessed.
    """
    direction = artifact["vector"][config["layer"]].float()
    return {
        "method": KEY,
        "raw_strength": float(config["alpha"]),
        "direction_norm": float(torch.linalg.vector_norm(direction)),
        "effective_delta_norm": float(
            abs(config["alpha"]) * torch.linalg.vector_norm(direction)
        ),
        "site": {"component": "resid_post", "layer": int(config["layer"])},
    }
