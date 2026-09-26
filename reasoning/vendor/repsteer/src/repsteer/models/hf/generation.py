from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

from repsteer.core.context import StepContext
from repsteer.core.errors import GenerationPhaseError

from .adapters.base import _cache_has_content


def _first_tensor(*values: Any) -> Tensor | None:
    for value in values:
        if isinstance(value, Tensor):
            return value
    return None


def _model_inputs(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    input_ids = kwargs.get("input_ids")
    if not isinstance(input_ids, Tensor) and args and isinstance(args[0], Tensor):
        input_ids = args[0]
    inputs_embeds = kwargs.get("inputs_embeds")
    attention_mask = kwargs.get("attention_mask")
    return (
        input_ids if isinstance(input_ids, Tensor) else None,
        inputs_embeds if isinstance(inputs_embeds, Tensor) else None,
        attention_mask if isinstance(attention_mask, Tensor) else None,
    )


def _lengths(
    input_ids: Tensor | None,
    inputs_embeds: Tensor | None,
    attention_mask: Tensor | None,
) -> Tensor:
    source = _first_tensor(input_ids, inputs_embeds, attention_mask)
    if source is None:
        return torch.ones(1, dtype=torch.long)
    batch = int(source.shape[0]) if source.ndim >= 2 else 1
    if attention_mask is not None and attention_mask.ndim >= 2:
        return attention_mask.to(dtype=torch.long).sum(dim=-1)
    sequence = int(source.shape[-2] if inputs_embeds is source else source.shape[-1])
    return torch.full((batch,), sequence, dtype=torch.long, device=source.device)


def _expand_batch(value: Tensor | None, batch: int) -> Tensor | None:
    if value is None or value.ndim == 0 or value.shape[0] == batch:
        return value
    if batch % int(value.shape[0]) != 0:
        return value
    factor = batch // int(value.shape[0])
    return value.repeat_interleave(factor, dim=0)


@dataclass
class GenerationTracker:
    """Tracks top-level model forwards during one explicit generate call.

    Input length alone is ambiguous (a one-token prompt and cached decoding both
    have length one), so phase decisions combine an explicit generation scope,
    forward ordinal, and cache state.
    """

    current: StepContext | None = None
    _active: bool = False
    _forward_index: int = 0
    _prompt_lengths: Tensor | None = None
    _initial_input_ids: Tensor | None = None
    _initial_attention_mask: Tensor | None = None
    _modality_map: Any | None = None

    @property
    def generation_active(self) -> bool:
        return self._active

    @property
    def modality_map(self) -> Any | None:
        return self._modality_map

    def begin(
        self,
        *,
        input_ids: Tensor | None,
        inputs_embeds: Tensor | None,
        attention_mask: Tensor | None,
        modality_map: Any | None = None,
    ) -> None:
        if self._active:
            raise GenerationPhaseError(
                "nested generate() calls on one model are unsupported"
            )
        self._active = True
        self._forward_index = 0
        self._prompt_lengths = _lengths(
            input_ids, inputs_embeds, attention_mask
        ).detach()
        self._initial_input_ids = input_ids
        self._initial_attention_mask = attention_mask
        self._modality_map = modality_map
        self.current = None

    def end(self) -> None:
        self._active = False
        self._forward_index = 0
        self._prompt_lengths = None
        self._initial_input_ids = None
        self._initial_attention_mask = None
        self._modality_map = None
        self.current = None

    @contextmanager
    def generation(
        self,
        *,
        input_ids: Tensor | None,
        inputs_embeds: Tensor | None,
        attention_mask: Tensor | None,
        modality_map: Any | None = None,
    ) -> Iterator[None]:
        self.begin(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            modality_map=modality_map,
        )
        try:
            yield
        finally:
            self.end()

    def update(
        self,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        *,
        modality_map: Any | None = None,
    ) -> StepContext:
        input_ids, inputs_embeds, attention_mask = _model_inputs(args, kwargs)
        source = _first_tensor(input_ids, inputs_embeds, attention_mask)
        batch = int(source.shape[0]) if source is not None and source.ndim >= 2 else 1
        if input_ids is not None:
            sequence = int(input_ids.shape[-1])
        elif inputs_embeds is not None:
            sequence = int(inputs_embeds.shape[-2])
        elif attention_mask is not None:
            sequence = int(attention_mask.shape[-1])
        else:
            sequence = 1

        phase: Literal["prefill", "decode", "forward"]
        prompt_lengths: Tensor
        if not self._active:
            prompt_lengths = _lengths(input_ids, inputs_embeds, attention_mask)
            phase = "forward"
            generation_step = None
        else:
            cache = kwargs.get("past_key_values", kwargs.get("past_key_value"))
            cached_decode = _cache_has_content(cache)
            # A pre-populated cache makes even the first tracked call decode.
            # Otherwise only the first model call is prefill; when use_cache is
            # false later forwards carry a growing sequence but remain decode.
            phase = "decode" if cached_decode or self._forward_index > 0 else "prefill"
            generation_step = (
                (
                    self._forward_index
                    if cached_decode and self._forward_index == 0
                    else self._forward_index - 1
                )
                if phase == "decode"
                else None
            )
            tracked_lengths = self._prompt_lengths
            if tracked_lengths is None:
                raise GenerationPhaseError("generation tracker lost prompt lengths")
            expanded_lengths = _expand_batch(tracked_lengths, batch)
            if expanded_lengths is None:
                raise GenerationPhaseError(
                    "generation tracker could not expand prompt lengths"
                )
            prompt_lengths = expanded_lengths
            input_ids = _expand_batch(input_ids, batch)
            attention_mask = _expand_batch(attention_mask, batch)
            self._forward_index += 1

        device = source.device if source is not None else torch.device("cpu")
        if modality_map is not None:
            if self._active and self._modality_map is not None:
                # Processor-derived maps are authoritative for a generation.
                # A model forward may expose only a subset of those inputs.
                modality_map = self._modality_map
            elif self._active:
                self._modality_map = modality_map
        elif self._active:
            modality_map = self._modality_map

        self.current = StepContext(
            phase=phase,
            batch_size=batch,
            sequence_length=sequence,
            prompt_lengths=prompt_lengths,
            generation_step=generation_step,
            attention_mask=attention_mask,
            token_ids=input_ids,
            modality_map=modality_map,
            device=device,
            dtype=None,
            metadata={
                "cache_present": bool(
                    _cache_has_content(
                        kwargs.get("past_key_values", kwargs.get("past_key_value"))
                    )
                ),
                "tracked_forward_index": max(0, self._forward_index - 1),
            },
        )
        return self.current

    def for_activation(
        self,
        activation: Tensor,
        *,
        stream: str = "language",
        component: str | None = None,
    ) -> StepContext:
        if activation.ndim < 2:
            raise GenerationPhaseError(
                "steering activation must have at least 2 dimensions, "
                f"got {activation.ndim}"
            )
        batch = int(activation.shape[0]) if activation.ndim >= 3 else 1
        sequence = int(activation.shape[-2])
        context = self.current
        if stream != "language":
            language_batch = (
                context.resolved_batch_size() if context is not None else None
            )
            metadata = dict(context.metadata) if context is not None else {}
            metadata["language_batch_size"] = language_batch
            metadata["site_stream"] = stream
            metadata["site_component"] = component
            return StepContext(
                phase=context.phase if context is not None else "forward",
                batch_size=batch,
                sequence_length=sequence,
                prompt_lengths=torch.full(
                    (batch,),
                    sequence,
                    dtype=torch.long,
                    device=activation.device,
                ),
                generation_step=(
                    context.generation_step if context is not None else None
                ),
                attention_mask=None,
                token_ids=None,
                modality_map=(
                    context.modality_map if context is not None else self._modality_map
                ),
                device=activation.device,
                dtype=activation.dtype,
                metadata=metadata,
            )
        if context is None:
            context = StepContext(
                phase="forward",
                batch_size=batch,
                sequence_length=sequence,
                prompt_lengths=torch.full(
                    (batch,), sequence, dtype=torch.long, device=activation.device
                ),
                attention_mask=None,
                token_ids=None,
                modality_map=self._modality_map,
                device=activation.device,
                dtype=activation.dtype,
            )
        prompt_lengths = context.prompt_lengths_for(batch, activation.device)
        attention_mask = _expand_batch(context.attention_mask, batch)
        token_ids = _expand_batch(context.token_ids, batch)
        metadata = dict(context.metadata)
        metadata["site_stream"] = stream
        metadata["site_component"] = component
        return context.with_updates(
            batch_size=batch,
            sequence_length=sequence,
            prompt_lengths=prompt_lengths,
            attention_mask=attention_mask,
            token_ids=token_ids,
            device=activation.device,
            dtype=activation.dtype,
            metadata=metadata,
        )


__all__ = ["GenerationTracker"]
