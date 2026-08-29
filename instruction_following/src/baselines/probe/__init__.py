"""Probe: steer along the direction a linear classifier separates by.

Where CAA takes the mean difference between the two activation sets, a probe
asks which direction best *discriminates* them. The two answers differ
whenever the difference has a large component along directions that also vary
within each class; the probe divides that variance out and CAA does not, so
comparing them says whether the steering signal needs to be the discriminative
direction or merely the average one.

Strength is expressed in standard deviations of the activation projection
along the unit probe direction, so alpha=1 shifts a prompt by one standard
deviation of what that site already varies by. This is the ITI convention and
it is not CAA's: the artifact records which one it is (Section 7 of the
restructure specification), and cross-method figures use measured dose rather
than alpha.
"""

from pathlib import Path

import torch

from baselines.shared import unit_direction

KEY = "probe"
METHOD_VERSION = 1
ARTIFACT_METHOD = "probe_logistic_regression"



def check_available() -> str | None:
    """Report the missing dependency instead of failing at import time."""
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return "missing scikit-learn"
    return None


def artifact_path(model_slug: str, task: str) -> Path:
    return unit_direction.artifact_path(model_slug, KEY, task)


def load_artifact(model_slug: str, task: str) -> dict:
    """Load a probe artifact after checking its minimal identity contract."""
    path = artifact_path(model_slug, task)
    return unit_direction.validate_artifact(
        torch.load(path, weights_only=True), path, ARTIFACT_METHOD, task,
        model_slug,
    )


def config_name(config: dict) -> str:
    return unit_direction.config_name(config)


def search_plan(context: dict, overrides: dict | None = None,
                budget: int | None = None):
    """The same joint (layer, alpha) grid the other residual-stream
    baselines search."""
    return unit_direction.layer_alpha_plan(
        context, overrides, budget,
        notes="alpha is in standard deviations of the projection at the layer",
    )


def delta(artifact: dict, config: dict) -> torch.Tensor:
    """Return the vector this configuration adds to the residual stream."""
    return unit_direction.delta(artifact, config)


def make_intervention(model, artifact: dict, config: dict):
    """Return a zero-argument callable producing the intervention context."""
    return unit_direction.make_intervention(model, artifact, config)


def dose_metadata(artifact: dict, config: dict) -> dict:
    """Strength facts that do not need a forward pass."""
    return unit_direction.dose_metadata(artifact, config, KEY)
