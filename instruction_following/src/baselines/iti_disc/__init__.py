"""ITI with discriminative head selection.

The same heads, the same site before ``o_proj``, the same mass-mean direction
and the same per-head sigma as ``iti``. What differs is the ranking that
decides which heads are edited.

``iti`` ranks heads by a probe's held-out accuracy at telling the instructed
prompt from the bare one, which is what its source method does. On this
benchmark that accuracy reached 1.000 on 279 of 1152 heads, because the two
prompts differ by a whole added sentence and almost any head that represents
the prompt can see it. Inside a tie that large the top-K is decided by index
order, and K=16 landed entirely in one layer for that reason rather than on
merit.

This variant ranks by an effect size against the *other* tasks' instructions.
That fixes both halves of the problem: a head that responds to any
instruction scores near zero, and an effect size keeps ordering heads that a
proportion-correct score has already saturated on.
"""

from pathlib import Path

import torch

from baselines import iti
from baselines.iti.heads import add_to_heads, head_layout
from baselines.shared.base import SearchPlan
from kernel.config import BASELINES, slug

KEY = "iti_disc"
METHOD_VERSION = 1
ARTIFACT_METHOD = "iti_discriminative_heads"
EXTRACTION_POSITION = iti.EXTRACTION_POSITION

HEAD_COUNTS = iti.HEAD_COUNTS
ALPHAS = iti.ALPHAS
DEFAULT_BUDGET = iti.DEFAULT_BUDGET


def check_available() -> str | None:
    return iti.check_available()


def artifact_path(model_slug: str, task: str) -> Path:
    return BASELINES / model_slug / KEY / f"{task}.pt"


def load_artifact(model_slug: str, task: str) -> dict:
    """Same contract as ``iti``'s, under this variant's method name."""
    path = artifact_path(model_slug, task)
    record = torch.load(path, weights_only=True)
    if not isinstance(record, dict):
        raise ValueError(f"artifact must be a dictionary: {path}")
    if record.get("method") != ARTIFACT_METHOD:
        raise ValueError(f"Artifact at {path} is not {ARTIFACT_METHOD}")
    if record.get("task") != task:
        raise ValueError(
            f"Artifact task {record.get('task')!r} does not match {task!r}"
        )
    artifact_model = record.get("model")
    if not isinstance(artifact_model, str) or slug(artifact_model) != model_slug:
        raise ValueError(
            f"Artifact model {artifact_model!r} does not match {model_slug!r}"
        )
    if record.get("extraction_position") != EXTRACTION_POSITION:
        raise ValueError(
            f"Artifact was captured at "
            f"{record.get('extraction_position')!r}, not "
            f"{EXTRACTION_POSITION!r}; produce it again: {path}"
        )
    direction = record.get("direction")
    sigma = record.get("sigma")
    if not isinstance(direction, torch.Tensor) or direction.ndim != 3:
        raise ValueError(
            f"Directions must have shape [layers, heads, head_dim]: {path}"
        )
    if not isinstance(sigma, torch.Tensor) or sigma.shape != direction.shape[:2]:
        raise ValueError(f"sigma must have one value per head: {path}")
    ranking = record.get("ranking")
    if not isinstance(ranking, list) or (
            len(ranking) != direction.shape[0] * direction.shape[1]):
        raise ValueError(f"ranking must order every head exactly once: {path}")
    blank = {(row[0], row[1]) for row in record.get("heads_without_direction", [])}
    norms = torch.linalg.vector_norm(direction.float(), dim=-1)
    for layer in range(norms.shape[0]):
        for head in range(norms.shape[1]):
            norm = float(norms[layer, head])
            if (layer, head) in blank:
                if norm != 0.0:
                    raise ValueError(
                        f"Head ({layer}, {head}) was recorded as carrying no "
                        f"direction but holds one: {path}"
                    )
            elif abs(norm - 1.0) > 1e-3:
                raise ValueError(
                    f"Head ({layer}, {head}) direction is not a unit "
                    f"vector: {path}"
                )
    return record


def config_name(config: dict) -> str:
    return iti.config_name(config)


def search_plan(context: dict, overrides: dict | None = None,
                budget: int | None = None) -> SearchPlan:
    """The same joint (heads_k, alpha) grid ``iti`` searches."""
    return iti.search_plan(context, overrides, budget)


def selected_heads(artifact: dict, heads_k: int) -> list[tuple[int, int]]:
    return iti.selected_heads(artifact, heads_k)


def additions(artifact: dict, config: dict) -> dict[tuple[int, int], torch.Tensor]:
    return iti.additions(artifact, config)


def make_intervention(model, artifact: dict, config: dict):
    n_layers, n_heads, head_dim = head_layout(model)
    expected = (n_layers, n_heads, head_dim)
    shape = tuple(artifact["direction"].shape)
    if shape != expected:
        raise ValueError(f"directions must have shape {expected}; got {shape}")
    return lambda: add_to_heads(model, additions(artifact, config))


def dose_metadata(artifact: dict, config: dict) -> dict:
    dose = iti.dose_metadata(artifact, config)
    dose["method"] = KEY
    return dose
