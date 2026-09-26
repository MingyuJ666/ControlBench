"""Small, artifact-first builders for common representation interventions.

Recipes construct ordinary :class:`~repsteer.core.SteeringPlan` values.  They
do not install hooks, compile a plan, or run generation.
"""

from __future__ import annotations

from typing import TypeAlias

import torch

from repsteer.artifacts import (
    DirectionArtifact,
    ITIArtifact,
    SAEFeatureArtifact,
    SubspaceArtifact,
)
from repsteer.core import Intervention, InterventionPhase, Site, SteeringPlan
from repsteer.gates import Gate
from repsteer.operators import Add, ITIAdd, RemoveProjection
from repsteer.positions import GeneratedTokens, PositionSelector
from repsteer.schedules import Constant, StrengthSchedule
from repsteer.sites import head_result

DirectionalArtifact: TypeAlias = DirectionArtifact | SubspaceArtifact
DirectionAddArtifact: TypeAlias = (
    DirectionArtifact | SAEFeatureArtifact | SubspaceArtifact
)


def _require_finite(value: torch.Tensor, *, name: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(
            f"{name} must contain only finite values; relearn or sanitize the "
            "artifact before building this recipe"
        )


def _require_nonzero_direction(value: torch.Tensor, *, name: str) -> None:
    if value.numel() == 0:
        raise ValueError(
            f"{name} must not be empty; provide a direction with a positive hidden "
            "dimension"
        )
    _require_finite(value, name=name)
    if bool(torch.linalg.vector_norm(value).eq(0).item()):
        raise ValueError(
            f"{name} must have non-zero norm; a zero direction would be a silent no-op"
        )


def _validate_ablation_artifact(artifact: DirectionalArtifact) -> None:
    if isinstance(artifact, DirectionArtifact):
        _require_nonzero_direction(artifact.direction, name="direction artifact")
        return
    if not isinstance(artifact, SubspaceArtifact):  # pragma: no cover - type guard
        raise TypeError(
            "directional_ablation() requires a DirectionArtifact or "
            f"SubspaceArtifact, got {type(artifact).__name__}"
        )
    basis = artifact.basis
    if basis.ndim != 2 or basis.shape[0] == 0 or basis.shape[1] == 0:
        raise ValueError(
            "subspace artifact must contain at least one non-empty basis vector"
        )
    _require_finite(basis, name="subspace basis")
    if bool(torch.linalg.vector_norm(basis).eq(0).item()):
        raise ValueError(
            "subspace artifact has zero span; provide at least one non-zero basis "
            "vector"
        )
    if artifact.mean is not None:
        _require_finite(artifact.mean, name="subspace mean")


def _validate_direction_add_artifact(artifact: DirectionAddArtifact) -> None:
    if isinstance(artifact, DirectionArtifact):
        _require_nonzero_direction(artifact.direction, name="steering direction")
        return
    if isinstance(artifact, SAEFeatureArtifact):
        _require_nonzero_direction(
            artifact.decoder_direction,
            name="SAE feature decoder direction",
        )
        return
    if isinstance(artifact, SubspaceArtifact):
        if artifact.rank != 1:
            raise ValueError(
                "gated_direction() requires a DirectionArtifact, SAEFeatureArtifact, "
                "or rank-1 SubspaceArtifact; select one subspace component before "
                "using an Add direction"
            )
        _require_nonzero_direction(artifact.basis[0], name="subspace direction")
        return
    raise TypeError(
        "gated_direction() requires an artifact-backed direction "
        "(DirectionArtifact, SAEFeatureArtifact, or rank-1 SubspaceArtifact), "
        f"got {type(artifact).__name__}; wrap tensors in an existing artifact first"
    )


def directional_ablation(
    artifact: DirectionalArtifact,
    *,
    positions: PositionSelector,
    site: Site | None = None,
    phase: InterventionPhase = "both",
    priority: int = 0,
) -> SteeringPlan:
    """Build a full direction or subspace projection-removal plan.

    The returned plan uses :class:`~repsteer.operators.RemoveProjection` with
    unit strength, preserving activation components orthogonal to ``artifact``.
    Model, site, and hidden-dimension compatibility remain the compiler's
    responsibility.

    Example:
        ``plan = rs.recipes.directional_ablation(artifact, positions=positions)``
    """

    _validate_ablation_artifact(artifact)
    return SteeringPlan(
        (
            Intervention(
                artifact=artifact,
                operator=RemoveProjection(),
                positions=positions,
                strength=Constant(1.0),
                site=site,
                phase=phase,
                priority=priority,
            ),
        )
    )


def gated_direction(
    steering_artifact: DirectionAddArtifact,
    *,
    gate: Gate,
    positions: PositionSelector,
    strength: float | StrengthSchedule = 1.0,
    site: Site | None = None,
    phase: InterventionPhase = "both",
    priority: int = 0,
) -> SteeringPlan:
    """Build an artifact-backed conditional direction-addition plan.

    ``gate`` is an existing :class:`~repsteer.gates.Gate`; this recipe only
    composes it with the ordinary :class:`~repsteer.operators.Add` operator.
    A gate with ``evaluate_at`` retains the runtime's prefill sequence-decision
    semantics.  This is a general conditional-steering API, not a complete
    CAST implementation.

    Example:
        ``plan = rs.recipes.gated_direction(artifact, gate=gate, positions=positions)``
    """

    _validate_direction_add_artifact(steering_artifact)
    intervention = Intervention(
        artifact=steering_artifact,
        operator=Add(),
        positions=positions,
        strength=Constant(1.0),
        gate=gate,
        site=site,
        phase=phase,
        priority=priority,
    ).with_strength(strength)
    return SteeringPlan((intervention,))


def iti(
    profile: ITIArtifact,
    *,
    positions: PositionSelector | None = None,
    strength: float | StrengthSchedule = 1.0,
    phase: InterventionPhase = "decode",
    priority: int = 0,
) -> SteeringPlan:
    """Build a faithful pre-``o_proj`` ITI plan from a learned profile.

    ``strength`` is the ITI paper's :math:`\\alpha`; each profile row already
    contains the calibrated :math:`\\sigma_{l,h}\\theta_{l,h}` term. The default
    targets generated tokens only, matching the official implementation's
    decode-only intervention rather than editing the input prompt.
    """

    if not isinstance(profile, ITIArtifact):
        raise TypeError("iti() requires an ITIArtifact")
    if phase not in {"prefill", "decode", "both"}:
        raise ValueError("iti() phase must be 'prefill', 'decode', or 'both'")
    selected_positions = GeneratedTokens() if positions is None else positions
    schedule = (
        strength
        if isinstance(strength, StrengthSchedule)
        or callable(getattr(strength, "value", None))
        else Constant(strength)
    )
    return SteeringPlan(
        Intervention(
            artifact=profile,
            operator=ITIAdd(layer=layer),
            positions=selected_positions,
            strength=schedule,
            site=head_result(layer),
            phase=phase,
            priority=priority,
        )
        for layer in profile.layers
    )


__all__ = ["directional_ablation", "gated_direction", "iti"]
