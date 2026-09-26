from __future__ import annotations

import weakref
from typing import Any

from repsteer.core.errors import PlanCompilationError
from repsteer.core.intervention import as_plan

from .compiled_plan import CompiledIntervention, CompiledPlan


def _validate_protocol(value: Any, method: str, role: str, index: int) -> None:
    if not callable(getattr(value, method, None)):
        raise PlanCompilationError(
            f"intervention {index} {role} ({type(value).__name__}) must implement "
            f"{method}()"
        )


def _activation_gate_sites(gate: Any) -> tuple[Any, ...]:
    """Collect explicit activation dependencies from logical gate trees."""

    sites: list[Any] = []
    evaluate_at = getattr(gate, "evaluate_at", None)
    if evaluate_at is not None:
        sites.append(evaluate_at)
    nested = getattr(gate, "gates", ())
    if isinstance(nested, tuple | list):
        for value in nested:
            sites.extend(_activation_gate_sites(value))
    single = getattr(gate, "gate", None)
    if single is not None:
        sites.extend(_activation_gate_sites(single))
    unique: list[Any] = []
    for site in sites:
        if site not in unique:
            unique.append(site)
    return tuple(unique)


def _gate_artifacts(gate: Any) -> tuple[Any, ...]:
    values: list[Any] = []
    for name in ("probe", "direction", "feature"):
        artifact = getattr(gate, name, None)
        if (
            name == "feature"
            and hasattr(gate, "sae")
            and getattr(gate, "sae", None) is None
        ):
            # A portable SAE feature's primary tensor is a decoder direction
            # with residual width. A direct latent gate instead reads
            # num_features-wide values and validates that width from metadata
            # when it evaluates.
            continue
        if getattr(artifact, "metadata", None) is not None:
            values.append(artifact)
    nested = getattr(gate, "gates", ())
    if isinstance(nested, tuple | list):
        for value in nested:
            values.extend(_gate_artifacts(value))
    single = getattr(gate, "gate", None)
    if single is not None:
        values.extend(_gate_artifacts(single))
    return tuple(values)


def _gate_reductions(gate: Any) -> tuple[Any, ...]:
    """Collect reduction modes from every activation gate in a logical tree."""

    values: list[Any] = []
    reduction = getattr(gate, "reduction", None)
    if reduction is not None:
        values.append(reduction)
    nested = getattr(gate, "gates", ())
    if isinstance(nested, tuple | list):
        for value in nested:
            values.extend(_gate_reductions(value))
    single = getattr(gate, "gate", None)
    if single is not None:
        values.extend(_gate_reductions(single))
    return tuple(values)


def _is_statically_disabled(schedule: Any) -> bool:
    marker = getattr(schedule, "is_always_zero", False)
    return bool(marker() if callable(marker) else marker)


def _assert_compatibility(
    artifact: Any,
    model: Any,
    site: Any,
    hidden_size: int,
    compatibility: str,
) -> str:
    try:
        from repsteer.artifacts.compatibility import assert_compatible

        result = assert_compatible(
            artifact,
            model,
            compatibility=compatibility,
            site=site,
            hidden_size=hidden_size,
        )
        return result.level.value
    except ImportError:
        # Defensive fallback for minimal installations that use protocol-only
        # artifacts.  Hidden dimension and exact identity remain safe-fail.
        metadata = getattr(artifact, "metadata", None)
        artifact_hidden = getattr(metadata, "hidden_size", None)
        if artifact_hidden != hidden_size:
            raise PlanCompilationError(
                f"artifact hidden size {artifact_hidden} does not match resolved "
                f"site hidden size {hidden_size}"
            ) from None
        if compatibility == "exact":
            artifact_id = getattr(metadata, "model_id", None)
            artifact_revision = getattr(metadata, "model_revision", None)
            if artifact_id != model.model_id or artifact_revision != model.revision:
                raise PlanCompilationError(
                    "exact artifact identity mismatch: "
                    f"{artifact_id}@{artifact_revision} != "
                    f"{model.model_id}@{model.revision}"
                ) from None
        return compatibility


def compile_plan(
    model: Any,
    plan: Any,
    *,
    compatibility: str = "exact",
) -> CompiledPlan:
    """Resolve and validate a plan without registering any PyTorch hooks."""

    if isinstance(plan, CompiledPlan):
        source = plan.source_model
        if source is None:
            raise PlanCompilationError(
                "compiled plan has lost its source HFSteerableModel; compile the "
                "original semantic SteeringPlan again for the target wrapper"
            )
        if source is not model:
            raise PlanCompilationError(
                "compiled plan belongs to a different HFSteerableModel instance"
            )
        if plan.model_id != model.model_id or plan.revision != model.revision:
            raise PlanCompilationError(
                "compiled plan model identity no longer matches the target wrapper"
            )
        return plan

    try:
        normalized = as_plan(plan)
    except (TypeError, ValueError) as exc:
        raise PlanCompilationError(f"invalid SteeringPlan: {exc}") from exc

    ordered_pairs = sorted(
        enumerate(normalized.interventions),
        key=lambda pair: (pair[1].priority, pair[0]),
    )
    compiled: list[CompiledIntervention] = []
    warnings: list[str] = []
    for execution_index, (declaration_index, intervention) in enumerate(ordered_pairs):
        try:
            site = intervention.resolved_site
        except (AttributeError, TypeError, ValueError) as exc:
            raise PlanCompilationError(
                f"intervention {declaration_index} has no resolvable site: {exc}"
            ) from exc
        try:
            resolved = model.resolve_site(site)
        except Exception as exc:
            if isinstance(exc, PlanCompilationError):
                raise
            raise PlanCompilationError(
                f"failed to resolve intervention {declaration_index} at {site}: {exc}"
            ) from exc
        if getattr(site, "component", None) == "head_result":
            capabilities = getattr(
                getattr(model, "adapter", None), "capabilities", None
            )
            require = getattr(capabilities, "require", None)
            if not callable(require):
                raise PlanCompilationError(
                    "head_result interventions require an adapter with an explicit "
                    "head-result capability declaration"
                )
            try:
                require("head_result")
            except (TypeError, ValueError) as exc:
                raise PlanCompilationError(str(exc)) from exc

        _validate_protocol(
            intervention.positions, "select", "positions", declaration_index
        )
        _validate_protocol(
            intervention.strength, "value", "strength", declaration_index
        )
        _validate_protocol(intervention.gate, "evaluate", "gate", declaration_index)
        _validate_protocol(
            intervention.operator, "apply", "operator", declaration_index
        )
        if intervention.phase not in ("prefill", "decode", "both"):
            raise PlanCompilationError(
                f"intervention {declaration_index} has invalid phase "
                f"{intervention.phase!r}"
            )

        metadata = getattr(intervention.artifact, "metadata", None)
        source_site = getattr(metadata, "site", None)
        explicit_cross_site = intervention.site is not None and source_site != site
        # Explicitly targeting another site is a supported, visible research
        # decision.  Exact still binds model id + revision + hidden dimension;
        # it does not pretend the target is the artifact's capture site.
        compatibility_site = source_site if explicit_cross_site else site
        actual_compatibility = _assert_compatibility(
            intervention.artifact,
            model,
            compatibility_site,
            resolved.hidden_dim,
            compatibility,
        )
        item_warnings: list[str] = []
        if explicit_cross_site:
            item_warnings.append(
                "explicit cross-site application: "
                f"learned at {source_site}, applied at {site}"
            )
        artifact_dtype = getattr(metadata, "dtype", None)
        model_dtype = getattr(model, "dtype", None)
        if (
            artifact_dtype
            and model_dtype
            and str(model_dtype).removeprefix("torch.") != artifact_dtype
        ):
            item_warnings.append(
                f"artifact dtype {artifact_dtype} will be converted at "
                f"runtime to {model_dtype}"
            )

        gate_sites = _activation_gate_sites(intervention.gate)
        if len(gate_sites) > 1:
            rendered = ", ".join(str(value) for value in gate_sites)
            raise PlanCompilationError(
                f"intervention {declaration_index} combines activation gates at "
                f"multiple sites ({rendered}); sequence gates require one "
                "shared prefill evaluation site"
            )
        gate_site = None
        if gate_sites:
            gate_semantic_site = gate_sites[0]
            if getattr(gate_semantic_site, "stream", None) != "language":
                raise PlanCompilationError(
                    f"intervention {declaration_index} evaluates a cached sequence "
                    f"gate at {gate_semantic_site}; evaluate_at is supported only "
                    "on the language stream because vision/projector batches do not "
                    "map one-to-one to language batch items"
                )
            if getattr(site, "stream", None) != "language":
                raise PlanCompilationError(
                    f"intervention {declaration_index} applies a cached sequence "
                    f"gate to {site}; cached language decisions currently support "
                    "only language-stream targets because no target-row to sample "
                    "mapping contract is available"
                )
            try:
                gate_site = model.resolve_site(gate_semantic_site)
            except Exception as exc:
                raise PlanCompilationError(
                    f"failed to resolve gate dependency for intervention "
                    f"{declaration_index} at {gate_semantic_site}: {exc}"
                ) from exc
            reductions = _gate_reductions(intervention.gate)
            if "none" in reductions:
                raise PlanCompilationError(
                    "cached activation gates must reduce to one sequence-level "
                    "value per batch item; reduction='none' is dynamic token gating"
                )
            for gate_artifact in _gate_artifacts(intervention.gate):
                gate_metadata = getattr(gate_artifact, "metadata", None)
                gate_source_site = getattr(gate_metadata, "site", None)
                compatibility_site = (
                    gate_source_site
                    if gate_source_site is not None
                    and gate_source_site != gate_semantic_site
                    else gate_semantic_site
                )
                _assert_compatibility(
                    gate_artifact,
                    model,
                    compatibility_site,
                    gate_site.hidden_dim,
                    compatibility,
                )
            item_warnings.append(
                "sequence gate evaluated during prefill at "
                f"{gate_semantic_site} and cached for decode"
            )
        else:
            # Activation gates without evaluate_at consume the controlled
            # tensor directly. They are dynamic (not cached), but their
            # portable probe/direction still needs normal model compatibility.
            for gate_artifact in _gate_artifacts(intervention.gate):
                gate_metadata = getattr(gate_artifact, "metadata", None)
                gate_source_site = getattr(gate_metadata, "site", None)
                compatibility_site = (
                    gate_source_site
                    if gate_source_site is not None and gate_source_site != site
                    else site
                )
                _assert_compatibility(
                    gate_artifact,
                    model,
                    compatibility_site,
                    resolved.hidden_dim,
                    compatibility,
                )
        compiled.append(
            CompiledIntervention(
                declaration_index=declaration_index,
                execution_index=execution_index,
                intervention=intervention,
                site=site,
                resolved_site=resolved,
                gate_site=gate_site,
                compatibility=actual_compatibility,
                statically_disabled=_is_statically_disabled(intervention.strength),
                warnings=tuple(item_warnings),
            )
        )

    return CompiledPlan(
        interventions=tuple(compiled),
        model_id=model.model_id,
        revision=model.revision,
        architecture=model.architecture,
        compatibility=compatibility,
        warnings=tuple(warnings),
        _model_ref=weakref.ref(model),
    )


__all__ = ["compile_plan"]
