"""SAE with discriminative feature selection.

The same dictionary, the same decoder direction, the same dose, the same
admitted layers as ``sae``. One thing differs: which of the 65,536 features
is chosen.

``sae`` ranks by how much a feature responds to the instruction being present
at all, which is what its source method does. On this benchmark that ranking
returned the same feature at layer 33 for four of the five tasks — a feature
that reports "there is a formatting instruction here" rather than which one.
Steering along it cannot produce five different behaviours.

This variant ranks by how much more a feature responds to *this* instruction
than to the other four, so a general instruction-detector scores near zero.
It exists beside ``sae`` rather than replacing it because the difference
between the two rankings is the measurement: whether sparse-feature steering
underperforms because the features are not there, or because the usual way of
looking for them finds the wrong one.
"""

from pathlib import Path

import torch

from baselines import sae
from baselines.shared import unit_direction
from baselines.shared.base import SearchPlan

KEY = "sae_disc"
METHOD_VERSION = 1
ARTIFACT_METHOD = "sae_discriminative_feature"
SCALE_KEY = sae.SCALE_KEY

ALPHAS = sae.ALPHAS
MAX_SEARCHED_LAYERS = sae.MAX_SEARCHED_LAYERS
DEFAULT_BUDGET = sae.DEFAULT_BUDGET


def check_available() -> str | None:
    return sae.check_available()


def artifact_path(model_slug: str, task: str) -> Path:
    return unit_direction.artifact_path(model_slug, KEY, task)


def load_artifact(model_slug: str, task: str) -> dict:
    path = artifact_path(model_slug, task)
    return unit_direction.validate_artifact(
        torch.load(path, weights_only=True), path, ARTIFACT_METHOD, task,
        model_slug, scale_key=SCALE_KEY,
    )


def config_name(config: dict) -> str:
    return unit_direction.config_name(config)


def search_plan(context: dict, overrides: dict | None = None,
                budget: int | None = None) -> SearchPlan:
    """The same grid ``sae`` searches, over the same gate-admitted layers.

    The gate is a statement about the dictionary and the model, not about how
    a feature was picked, so both variants inherit it — and searching the same
    sites is what makes their scores comparable.
    """
    return sae.search_plan(context, overrides, budget)


def delta(artifact: dict, config: dict) -> torch.Tensor:
    return unit_direction.delta(artifact, config, scale_key=SCALE_KEY)


def make_intervention(model, artifact: dict, config: dict):
    return unit_direction.make_intervention(
        model, artifact, config, scale_key=SCALE_KEY
    )


def dose_metadata(artifact: dict, config: dict) -> dict:
    dose = unit_direction.dose_metadata(
        artifact, config, KEY, scale_key=SCALE_KEY
    )
    layer = config["layer"]
    dose["projection_sigma"] = None          # the scale here is not a sigma
    dose["max_act"] = float(artifact[SCALE_KEY][layer])
    dose["feature"] = int(artifact["feature"][layer])
    return dose
