"""Public exception hierarchy for :mod:`repsteer`.

The core library deliberately uses a small, actionable exception hierarchy so
applications do not need to match backend-specific error messages.
"""

from __future__ import annotations


class RepSteerError(Exception):
    """Base class for all errors raised by repsteer."""


# Kept as an alias for the spelling used in an early RFC draft.
repsteerError = RepSteerError


class UnsupportedArchitectureError(RepSteerError):
    """The requested model architecture has no compatible adapter."""


class SiteResolutionError(RepSteerError):
    """A semantic site cannot be resolved on the target model."""


class PositionResolutionError(RepSteerError):
    """A position selector cannot be resolved for the current inputs."""


class ArtifactCompatibilityError(RepSteerError):
    """An artifact is not compatible at the requested trust level."""


class ArtifactFormatError(RepSteerError):
    """An artifact bundle is malformed or uses an unsupported schema."""


class PlanCompilationError(RepSteerError):
    """A steering plan cannot be compiled without changing its semantics."""


class HookLifecycleError(RepSteerError):
    """A runtime hook could not be safely registered or removed."""


class GenerationPhaseError(RepSteerError):
    """An operation is invalid for the current generation phase."""


class MissingOptionalDependencyError(RepSteerError, ImportError):
    """A requested integration needs an optional dependency."""


class ProcessorCompatibilityError(RepSteerError):
    """A multimodal processor does not match the wrapped model revision."""


class UnsafeArtifactError(RepSteerError):
    """An artifact failed an integrity or trust-policy check."""


__all__ = [
    "RepSteerError",
    "repsteerError",
    "UnsupportedArchitectureError",
    "SiteResolutionError",
    "PositionResolutionError",
    "ArtifactCompatibilityError",
    "ArtifactFormatError",
    "PlanCompilationError",
    "ProcessorCompatibilityError",
    "HookLifecycleError",
    "GenerationPhaseError",
    "MissingOptionalDependencyError",
    "UnsafeArtifactError",
]
