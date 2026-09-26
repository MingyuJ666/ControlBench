"""Explicit runtime context passed to selectors, schedules, gates and operators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, TypeAlias

import torch
from torch import Tensor

from .errors import GenerationPhaseError, PositionResolutionError

GenerationPhase: TypeAlias = Literal["prefill", "decode", "forward"]


@dataclass(frozen=True, slots=True)
class StepContext:
    """Information that makes generation-time intervention semantics explicit.

    ``batch_size`` and ``sequence_length`` may be omitted by callers that have
    the activation available; selectors infer them from that activation.  The
    complete fields are nevertheless retained so runtimes can validate their
    bookkeeping before invoking a hook.
    """

    phase: GenerationPhase
    prompt_lengths: Tensor
    batch_size: int | None = None
    sequence_length: int | None = None
    generation_step: int | None = None
    attention_mask: Tensor | None = None
    token_ids: Tensor | None = None
    modality_map: Any | None = None
    device: torch.device | str | None = None
    dtype: torch.dtype | None = None
    texts: Sequence[str] | None = None
    token_offsets: Any | None = None
    token_strings: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.phase not in ("prefill", "decode", "forward"):
            raise GenerationPhaseError(
                f"Unsupported generation phase {self.phase!r}; expected "
                "'prefill', 'decode', or 'forward'"
            )
        prompt_lengths = torch.as_tensor(self.prompt_lengths, dtype=torch.long)
        if prompt_lengths.ndim == 0:
            prompt_lengths = prompt_lengths.reshape(1)
        if prompt_lengths.ndim != 1:
            raise ValueError(
                "StepContext.prompt_lengths must be a scalar or 1-D tensor"
            )
        if torch.any(prompt_lengths < 0):
            raise ValueError("StepContext.prompt_lengths cannot be negative")
        object.__setattr__(self, "prompt_lengths", prompt_lengths)
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("StepContext.batch_size must be positive")
        if self.sequence_length is not None and self.sequence_length < 0:
            raise ValueError("StepContext.sequence_length cannot be negative")
        if self.generation_step is not None and self.generation_step < 0:
            raise ValueError("StepContext.generation_step cannot be negative")
        if self.device is not None:
            object.__setattr__(self, "device", torch.device(self.device))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def resolved_batch_size(self, activation: Tensor | None = None) -> int:
        candidates: list[tuple[str, int]] = []
        if self.batch_size is not None:
            candidates.append(("batch_size", self.batch_size))
        if self.prompt_lengths.numel() > 1:
            candidates.append(("prompt_lengths", int(self.prompt_lengths.numel())))
        if self.attention_mask is not None and self.attention_mask.ndim >= 2:
            candidates.append(("attention_mask", int(self.attention_mask.shape[0])))
        if self.token_ids is not None and self.token_ids.ndim >= 2:
            candidates.append(("token_ids", int(self.token_ids.shape[0])))
        if activation is not None:
            if activation.ndim >= 3:
                candidates.append(("activation", int(activation.shape[0])))
            elif activation.ndim == 2:
                candidates.append(("activation", 1))
            else:
                raise PositionResolutionError(
                    "Position selectors require an activation with at least sequence "
                    "and hidden dimensions"
                )
        if not candidates:
            return int(self.prompt_lengths.numel())
        expected = candidates[0][1]
        mismatches = [(name, value) for name, value in candidates if value != expected]
        if mismatches:
            details = ", ".join(f"{name}={value}" for name, value in candidates)
            raise PositionResolutionError(f"Inconsistent batch dimensions: {details}")
        return expected

    def resolved_sequence_length(self, activation: Tensor | None = None) -> int:
        inferred: int | None = None
        if activation is not None:
            if activation.ndim < 2:
                raise PositionResolutionError(
                    "Position selectors require an activation with at least "
                    "2 dimensions"
                )
            inferred = int(activation.shape[-2])
        if (
            self.sequence_length is not None
            and inferred is not None
            and inferred != self.sequence_length
            and self.phase != "decode"
        ):
            # During cached decoding the context may describe the full sequence
            # while the current activation contains only its suffix.  That is a
            # valid and common state, so only reject it outside decode.
            raise PositionResolutionError(
                "Context sequence_length does not match activation: "
                f"{self.sequence_length} != {inferred}"
            )
        if self.sequence_length is not None:
            return inferred if inferred is not None else self.sequence_length
        if inferred is not None:
            return inferred
        if self.token_ids is not None:
            return int(self.token_ids.shape[-1])
        if self.attention_mask is not None:
            return int(self.attention_mask.shape[-1])
        raise PositionResolutionError("Cannot infer the current sequence length")

    def prompt_lengths_for(self, batch_size: int, device: torch.device) -> Tensor:
        lengths = self.prompt_lengths.to(device=device, dtype=torch.long)
        if lengths.numel() == 1 and batch_size != 1:
            lengths = lengths.expand(batch_size)
        if lengths.numel() != batch_size:
            raise PositionResolutionError(
                "prompt_lengths has "
                f"{lengths.numel()} entries for a batch of size {batch_size}"
            )
        return lengths

    def current_attention_mask(self, activation: Tensor) -> Tensor:
        """Return a bool mask aligned to the activation's current sequence axis."""

        batch = self.resolved_batch_size(activation)
        sequence = self.resolved_sequence_length(activation)
        device = activation.device
        if self.attention_mask is None:
            return torch.ones((batch, sequence), dtype=torch.bool, device=device)
        mask = torch.as_tensor(self.attention_mask, device=device)
        if mask.ndim == 1 and batch == 1:
            mask = mask.unsqueeze(0)
        while mask.ndim > 2 and mask.shape[1] == 1:
            mask = mask.squeeze(1)
        if mask.ndim != 2:
            raise PositionResolutionError(
                "attention_mask must resolve to [batch, sequence], got "
                f"shape {tuple(mask.shape)}"
            )
        if mask.shape[0] != batch:
            raise PositionResolutionError(
                f"attention_mask batch {mask.shape[0]} does not match "
                f"activation batch {batch}"
            )
        if mask.shape[1] < sequence:
            raise PositionResolutionError(
                f"attention_mask length {mask.shape[1]} is shorter than the current "
                f"activation length {sequence}"
            )
        # Cached decode commonly supplies a full-history attention mask and a
        # one-token activation.  The current activation corresponds to the suffix.
        if mask.shape[1] > sequence:
            mask = mask[:, -sequence:] if sequence else mask[:, :0]
        return mask.to(dtype=torch.bool)

    @property
    def runtime_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        for value in (self.token_ids, self.attention_mask, self.prompt_lengths):
            if isinstance(value, Tensor):
                return value.device
        return torch.device("cpu")

    def with_updates(self, **changes: Any) -> StepContext:
        return replace(self, **changes)


# MVP gates use StepContext directly; this alias keeps the public vocabulary
# available without introducing a second context object with divergent fields.
GateContext = StepContext


__all__ = ["GateContext", "GenerationPhase", "StepContext"]
