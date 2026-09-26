from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from repsteer.core.errors import (
    GenerationPhaseError,
    HookLifecycleError,
    PositionResolutionError,
)
from repsteer.positions.base import apply_mask, validate_mask

from .compiled_plan import CompiledIntervention, CompiledPlan

_MISSING = object()


def _owner_key() -> tuple[int, int | None]:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return (threading.get_ident(), id(task) if task is not None else None)


def _as_tensor(
    value: Any, activation: Tensor, *, dtype: torch.dtype | None = None
) -> Tensor:
    return torch.as_tensor(
        value,
        device=activation.device,
        dtype=dtype if dtype is not None else activation.dtype,
    )


def _gate_weights(
    gate: Any,
    mask: Tensor,
    activation: Tensor,
    *,
    cached_sequence_gate: bool = False,
) -> tuple[Tensor, bool]:
    """Return [B,S] gate weights and whether they are strictly boolean."""

    # Preserve boolean gates so selected/unselected merging uses torch.where
    # exactly (and cannot propagate NaN from an unselected candidate).
    value = torch.as_tensor(gate, device=activation.device)
    batch, sequence = mask.shape
    if value.ndim == 0:
        if cached_sequence_gate and batch != 1:
            raise PositionResolutionError(
                "cached sequence gate has one decision for a target batch of "
                f"{batch}; cached decisions must remain one per original sample"
            )
        value = value.expand(batch, sequence)
    elif value.ndim == 1:
        if value.numel() == batch:
            value = value[:, None].expand(batch, sequence)
        elif cached_sequence_gate:
            raise PositionResolutionError(
                "cached sequence gate has "
                f"{value.numel()} decisions for a target batch of {batch}; cached "
                "decisions must remain one per original sample"
            )
        elif value.numel() > 0 and batch % value.numel() == 0:
            value = value.repeat_interleave(batch // value.numel())
            value = value[:, None].expand(batch, sequence)
        elif batch == 1 and value.numel() == sequence:
            value = value[None, :]
        else:
            raise PositionResolutionError(
                f"gate returned shape {tuple(value.shape)} for selector mask "
                f"{tuple(mask.shape)}"
            )
    else:
        while value.ndim > 2 and value.shape[-1] == 1:
            value = value.squeeze(-1)
        if tuple(value.shape) != (batch, sequence):
            try:
                value = torch.broadcast_to(value, (batch, sequence))
            except RuntimeError as exc:
                raise PositionResolutionError(
                    f"gate returned non-broadcastable shape {tuple(value.shape)} for "
                    f"selector mask {tuple(mask.shape)}"
                ) from exc
    if value.dtype == torch.bool:
        return value & mask, True
    if not bool(torch.isfinite(value).all()):
        raise PositionResolutionError("gate returned non-finite weights")
    if bool(((value < 0) | (value > 1)).any()):
        raise PositionResolutionError("gate weights must lie in [0, 1]")
    return value * mask.to(dtype=value.dtype), False


def _cacheable_sequence_gate(value: Any, activation: Tensor, context: Any) -> Tensor:
    """Validate that a prefill gate can be reused as one value per sequence."""

    tensor = torch.as_tensor(value, device=activation.device)
    while tensor.ndim > 1 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    batch = context.resolved_batch_size()
    if batch > 1 and (tensor.ndim != 1 or tensor.numel() != batch):
        raise GenerationPhaseError(
            "a cached sequence gate must return exactly one value per batch item; "
            f"got shape {tuple(tensor.shape)} for batch {batch}. Scalar or "
            "single-decision broadcasting is not supported"
        )
    if batch == 1 and (tensor.ndim > 1 or (tensor.ndim == 1 and tensor.numel() != 1)):
        raise GenerationPhaseError(
            "a cached sequence gate must return a scalar or one value per "
            f"batch item; got shape {tuple(tensor.shape)} for batch {batch}"
        )
    if tensor.dtype != torch.bool:
        if not bool(torch.isfinite(tensor).all()):
            raise GenerationPhaseError(
                "a cached sequence gate returned non-finite values"
            )
        if bool(((tensor < 0) | (tensor > 1)).any()):
            raise GenerationPhaseError(
                "cached sequence gate weights must lie in [0, 1]"
            )
    return tensor


def _blend(
    original: Tensor, candidate: Tensor, weights: Tensor, boolean: bool
) -> Tensor:
    if original.shape != candidate.shape:
        raise PositionResolutionError(
            f"operator changed activation shape {tuple(original.shape)} -> "
            f"{tuple(candidate.shape)}"
        )
    if boolean:
        return apply_mask(original, candidate, weights.to(dtype=torch.bool))
    if original.ndim == 2:
        if tuple(weights.shape) != (1, original.shape[0]):
            raise PositionResolutionError(
                "gate weight shape does not match unbatched activation"
            )
        broadcast = weights[0, :, None]
    else:
        broadcast = weights
        while broadcast.ndim < original.ndim:
            broadcast = broadcast.unsqueeze(-1)
    broadcast = broadcast.to(device=original.device, dtype=original.dtype)
    return original + broadcast * (candidate - original)


def apply_compiled_intervention(
    item: CompiledIntervention,
    activation: Tensor,
    context: Any,
    *,
    gate_value: Any = _MISSING,
) -> Tensor:
    """Execute one numeric intervention and merge only selected positions."""

    intervention = item.intervention
    # A zero schedule is a strict baseline invariant.  Resolve it before the
    # selector, gate, or operator so custom components cannot perturb RNG/state
    # in an intervention that is mathematically disabled.
    strength_value = intervention.strength.value(activation, context)
    strength = _as_tensor(strength_value, activation)
    if strength.numel() == 0 or bool(torch.all(strength == 0).item()):
        return activation

    # Non-zero controls invoke every remaining protocol component.  Position
    # selection and gate blending stay runtime-owned, so operators cannot write
    # outside the selected token mask.
    cached_sequence_gate = gate_value is not _MISSING and item.gate_site is not None
    raw_mask = intervention.positions.select(activation, context)
    mask = validate_mask(torch.as_tensor(raw_mask), activation, context)
    if gate_value is _MISSING:
        if item.gate_site is not None:
            raise GenerationPhaseError(
                "sequence gate has no cached prefill value. Its evaluate_at site "
                "must execute before the controlled site in prefill, and generation "
                "cannot start from a pre-populated KV cache"
            )
        metadata = dict(context.metadata)
        metadata.update(
            {
                "activation": activation,
                "gate_activation": activation,
            }
        )
        gate_context = context.with_updates(metadata=metadata)
        gate_value = intervention.gate.evaluate(gate_context)
    weights, boolean_gate = _gate_weights(
        gate_value,
        mask,
        activation,
        cached_sequence_gate=cached_sequence_gate,
    )

    # Pass a clone so a third-party in-place operator cannot alter positions
    # outside the runtime-owned mask before the merge.
    candidate = intervention.operator.apply(
        activation.clone(), intervention.artifact, strength, context
    )
    if not isinstance(candidate, Tensor):
        raise TypeError(
            f"operator {type(intervention.operator).__name__}.apply returned "
            f"{type(candidate).__name__}, expected Tensor"
        )
    candidate = candidate.to(device=activation.device, dtype=activation.dtype)

    return _blend(activation, candidate, weights, boolean_gate)


@dataclass
class _Frame:
    compiled: CompiledPlan
    handles: list[Any]
    conflict_keys: frozenset[tuple[int, str, str]]
    gate_values: dict[int, Tensor]
    idempotent: bool = False


class HookManager:
    """Owns hook registration for one model wrapper.

    Nested entry of the same compiled object is idempotent.  Distinct nested
    plans are allowed only when their concrete tensor targets are disjoint;
    overlapping mutable targets fail instead of inheriting ambiguous cross-plan
    ordering.  Concurrent thread/async-task access is rejected at the root model
    pre-hook, because PyTorch module hooks are process-global mutable state.
    """

    def __init__(self, wrapper: Any) -> None:
        self.wrapper = wrapper
        self.model = wrapper.model
        self.tracker = wrapper.generation_tracker
        self._lock = threading.RLock()
        self._owner: tuple[int, int | None] | None = None
        self._frames: list[_Frame] = []
        self._tracking_handle: Any | None = None

    @property
    def active(self) -> bool:
        with self._lock:
            return bool(self._frames)

    @property
    def depth(self) -> int:
        with self._lock:
            return len(self._frames)

    @property
    def hook_count(self) -> int:
        with self._lock:
            return sum(len(frame.handles) for frame in self._frames) + (
                1 if self._tracking_handle is not None else 0
            )

    def assert_owner(self) -> None:
        with self._lock:
            self._assert_owner_locked()

    def _assert_owner_locked(self) -> None:
        if self._owner is not None and self._owner != _owner_key():
            raise HookLifecycleError(
                "model has an active mutable steering session owned by another "
                "thread or asyncio task"
            )

    def _tracking_pre_with_kwargs(
        self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        del module
        self.assert_owner()
        modality_map = None
        if not (
            self.tracker.generation_active and self.tracker.modality_map is not None
        ):
            modality_inputs = kwargs
            if "input_ids" not in kwargs and args and isinstance(args[0], Tensor):
                modality_inputs = {**kwargs, "input_ids": args[0]}
            modality_map = self.wrapper.adapter.build_modality_map(
                modality_inputs, model=self.wrapper.model
            )
        context = self.tracker.update(args, kwargs, modality_map=modality_map)
        self._reset_gate_values(context)
        return None

    def _tracking_pre_without_kwargs(self, module: Any, args: tuple[Any, ...]) -> None:
        del module
        self.assert_owner()
        context = self.tracker.update(args, {})
        self._reset_gate_values(context)
        return None

    def _reset_gate_values(self, context: Any) -> None:
        first_tracked_forward = context.metadata.get("tracked_forward_index", 0) == 0
        if context.phase == "decode" and not first_tracked_forward:
            return
        with self._lock:
            self._clear_gate_values_locked()

    def _clear_gate_values_locked(self) -> None:
        for frame in self._frames:
            frame.gate_values.clear()

    @contextlib.contextmanager
    def generation_scope(self) -> Iterator[None]:
        """Isolate cached sequence-gate decisions to one wrapper generation.

        The tracker owns phase information; this manager owns hook-local state.
        Clearing at both boundaries makes a failed generation indistinguishable
        from a completed one to a reusable steering session.
        """

        self.assert_owner()
        with self._lock:
            self._clear_gate_values_locked()
        try:
            yield
        finally:
            with self._lock:
                self._clear_gate_values_locked()

    def _install_tracking_hook(self) -> Any:
        try:
            return self.model.register_forward_pre_hook(
                self._tracking_pre_with_kwargs, with_kwargs=True, prepend=True
            )
        except TypeError:
            # PyTorch <2 supports neither with_kwargs nor prepend.  Phase
            # tracking remains valid for positional input_ids, but cache-aware
            # decode detection may be less informative.
            return self.model.register_forward_pre_hook(
                self._tracking_pre_without_kwargs
            )

    def _callback(
        self,
        group: tuple[CompiledIntervention, ...],
        gate_values: dict[int, Tensor],
    ) -> Callable[..., Any]:
        resolved = group[0].resolved_site

        def callback(module: Any, args: tuple[Any, ...], output: Any = None) -> Any:
            del module
            container = args if resolved.hook_kind == "forward_pre" else output
            for item in group:
                intervention = item.intervention
                activation = item.resolved_site.read(container)
                context = self.tracker.for_activation(
                    activation,
                    stream=getattr(item.site, "stream", "language"),
                    component=getattr(item.site, "component", None),
                )
                if intervention.phase != "both" and intervention.phase != context.phase:
                    continue
                cached_gate = gate_values.get(item.execution_index, _MISSING)
                changed = apply_compiled_intervention(
                    item,
                    activation,
                    context,
                    gate_value=cached_gate,
                )
                container = item.resolved_site.rebuild(container, changed)
            return container

        return callback

    def _gate_callback(
        self,
        group: tuple[CompiledIntervention, ...],
        gate_values: dict[int, Tensor],
    ) -> Callable[..., Any]:
        resolved = group[0].gate_site
        if resolved is None:  # pragma: no cover - gate_groups guarantees this
            raise RuntimeError("compiled gate group has no resolved dependency")

        def callback(module: Any, args: tuple[Any, ...], output: Any = None) -> None:
            del module
            container = args if resolved.hook_kind == "forward_pre" else output
            activation = resolved.read(container)
            context = self.tracker.for_activation(
                activation,
                stream=getattr(resolved.site, "stream", "language"),
                component=getattr(resolved.site, "component", None),
            )
            if context.phase == "decode":
                return None
            metadata = dict(context.metadata)
            metadata.update(
                {
                    "activation": activation,
                    "gate_activation": activation,
                }
            )
            gate_context = context.with_updates(metadata=metadata)
            for item in group:
                value = item.intervention.gate.evaluate(gate_context)
                tensor = _cacheable_sequence_gate(value, activation, gate_context)
                gate_values[item.execution_index] = tensor.detach().clone()
            return None

        return callback

    @staticmethod
    def _register(resolved: Any, callback: Callable[..., Any]) -> Any:
        if resolved.hook_kind == "forward_pre":
            return resolved.module.register_forward_pre_hook(callback)
        return resolved.module.register_forward_hook(callback)

    def _install_plan(
        self,
        compiled: CompiledPlan,
        gate_values: dict[int, Tensor],
    ) -> list[Any]:
        handles: list[Any] = []
        try:
            # Read dependencies are installed first. If a gate reads the same
            # tensor that an intervention writes, it observes the unmodified
            # activation and caches one sequence value per batch item.
            for group in compiled.gate_groups():
                resolved = group[0].gate_site
                if resolved is None:  # pragma: no cover - structural guard
                    continue
                handles.append(
                    self._register(
                        resolved,
                        self._gate_callback(group, gate_values),
                    )
                )
            for group in compiled.groups():
                resolved = group[0].resolved_site
                handles.append(
                    self._register(
                        resolved,
                        self._callback(group, gate_values),
                    )
                )
            return handles
        except BaseException:
            for handle in reversed(handles):
                with contextlib.suppress(Exception):
                    handle.remove()
            raise

    def enter(self, compiled: CompiledPlan) -> None:
        with self._lock:
            key = _owner_key()
            if self._owner is not None and self._owner != key:
                raise HookLifecycleError(
                    "cannot enter steering session: this model is already active in "
                    "another thread or asyncio task"
                )
            same_as_outer = bool(self._frames) and (
                self._frames[-1].compiled is compiled
                or _same_declared_interventions(self._frames[-1].compiled, compiled)
            )
            if same_as_outer:
                self._frames.append(
                    _Frame(
                        compiled,
                        [],
                        frozenset(),
                        {},
                        idempotent=True,
                    )
                )
                return

            active_keys = frozenset(
                key for frame in self._frames for key in frame.conflict_keys
            )
            overlap = active_keys & compiled.conflict_keys
            if overlap:
                paths = sorted(
                    {
                        item.resolved_site.module_path
                        for item in compiled
                        if item.conflict_key in overlap
                    }
                )
                raise HookLifecycleError(
                    "nested steering plans target overlapping mutable tensors at "
                    f"{', '.join(paths)}; compile one SteeringPlan to define order"
                )

            root_frame = not self._frames
            installed_tracking = False
            try:
                if root_frame:
                    self._owner = key
                if self._tracking_handle is None and (
                    compiled.groups() or compiled.gate_groups()
                ):
                    self._tracking_handle = self._install_tracking_hook()
                    installed_tracking = True
                gate_values: dict[int, Tensor] = {}
                handles = self._install_plan(compiled, gate_values)
                self._frames.append(
                    _Frame(
                        compiled,
                        handles,
                        compiled.conflict_keys,
                        gate_values,
                    )
                )
            except BaseException as exc:
                if installed_tracking and self._tracking_handle is not None:
                    try:
                        self._tracking_handle.remove()
                    finally:
                        self._tracking_handle = None
                if root_frame:
                    self._owner = None
                if isinstance(exc, HookLifecycleError):
                    raise
                raise HookLifecycleError(
                    f"failed to register steering hooks: {exc}"
                ) from exc

    def exit(self, compiled: CompiledPlan) -> None:
        with self._lock:
            self._assert_owner_locked()
            if not self._frames:
                raise HookLifecycleError("steering session exit without matching entry")
            frame = self._frames[-1]
            if frame.compiled is not compiled:
                raise HookLifecycleError(
                    "steering sessions must exit in last-in, first-out order"
                )
            self._frames.pop()
            frame.gate_values.clear()
            failures: list[str] = []
            for handle in reversed(frame.handles):
                try:
                    handle.remove()
                except Exception as exc:
                    failures.append(str(exc))
            if (
                self._frames
                and self._tracking_handle is not None
                and not any(active.handles for active in self._frames)
            ):
                try:
                    self._tracking_handle.remove()
                except Exception as exc:
                    failures.append(str(exc))
                self._tracking_handle = None
                if not self.tracker.generation_active:
                    self.tracker.current = None
            if not self._frames:
                if self._tracking_handle is not None:
                    try:
                        self._tracking_handle.remove()
                    except Exception as exc:
                        failures.append(str(exc))
                self._tracking_handle = None
                self._owner = None
                if not self.tracker.generation_active:
                    self.tracker.current = None
            if failures:
                raise HookLifecycleError(
                    "one or more steering hooks could not be removed: "
                    + "; ".join(failures)
                )

    def clear(self) -> None:
        """Remove all owned hooks; only the owning execution context may call it."""

        with self._lock:
            self._assert_owner_locked()
            failures: list[str] = []
            for frame in reversed(self._frames):
                for handle in reversed(frame.handles):
                    try:
                        handle.remove()
                    except Exception as exc:
                        failures.append(str(exc))
                frame.gate_values.clear()
            self._frames.clear()
            if self._tracking_handle is not None:
                try:
                    self._tracking_handle.remove()
                except Exception as exc:
                    failures.append(str(exc))
            self._tracking_handle = None
            self._owner = None
            self.tracker.end()
            if failures:
                raise HookLifecycleError(
                    "one or more steering hooks could not be removed: "
                    + "; ".join(failures)
                )


__all__ = ["HookManager", "apply_compiled_intervention"]


def _same_declared_interventions(left: CompiledPlan, right: CompiledPlan) -> bool:
    """Identity comparison avoids tensor-valued dataclass equality surprises."""

    if len(left) != len(right):
        return False
    return all(
        first.intervention is second.intervention
        and first.conflict_key == second.conflict_key
        for first, second in zip(left, right, strict=True)
    )
