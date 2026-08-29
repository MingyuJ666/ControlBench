"""LAT-PCA: steer along the dominant axis of the contrastive differences.

CAA averages the positive-minus-negative differences; this method asks which
single axis those differences vary along most, taking the first principal
component of the centred differences (the reading-vector construction from
representation engineering). The two coincide only when the mean dominates
the spread: where carriers disagree about *where* the instruction moves the
activation, the average points somewhere no individual difference does, while
the principal axis follows the shared direction of movement.

With probe, that makes three estimators over one activation set — mean,
discriminative normal, dominant axis — differing only in what they take from
the same numbers.

Strength is in units of the projection's standard deviation, as for probe, so
the two are directly comparable; CAA's raw magnitude is not, which is what the
recorded dose metadata exists to convert.
"""

from pathlib import Path

import torch

from baselines.shared import unit_direction

KEY = "lat_pca"
METHOD_VERSION = 1
ARTIFACT_METHOD = "lat_pca_first_component"



def check_available() -> str | None:
    return None


def artifact_path(model_slug: str, task: str) -> Path:
    return unit_direction.artifact_path(model_slug, KEY, task)


def load_artifact(model_slug: str, task: str) -> dict:
    path = artifact_path(model_slug, task)
    record = unit_direction.validate_artifact(
        torch.load(path, weights_only=True), path, ARTIFACT_METHOD, task,
        model_slug,
    )
    # An artifact made before the signs alternated holds a different direction
    # under the same method name, and nothing in the tensor shows it. Refuse
    # it rather than steer along the axis of the residual variation.
    from baselines.lat_pca.producer import CONSTRUCTION

    built = record.get("producer_config", {}).get("construction")
    if built != CONSTRUCTION:
        raise ValueError(
            f"{path} was built as {built!r}, not {CONSTRUCTION!r}; "
            "produce it again with the current recipe"
        )
    return record


def config_name(config: dict) -> str:
    return unit_direction.config_name(config)


def search_plan(context: dict, overrides: dict | None = None,
                budget: int | None = None):
    return unit_direction.layer_alpha_plan(
        context, overrides, budget,
        notes="alpha is in standard deviations of the projection at the layer",
    )


def delta(artifact: dict, config: dict) -> torch.Tensor:
    return unit_direction.delta(artifact, config)


def make_intervention(model, artifact: dict, config: dict):
    return unit_direction.make_intervention(model, artifact, config)


def dose_metadata(artifact: dict, config: dict) -> dict:
    return unit_direction.dose_metadata(artifact, config, KEY)
