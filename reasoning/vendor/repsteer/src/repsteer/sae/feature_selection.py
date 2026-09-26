"""Supervised and causal SAE feature selection.

The causal API deliberately receives an evaluator callback.  repsteer owns
candidate construction and provenance; the callback owns the task-specific
model run and metric, avoiding a hidden dependency on any one benchmark.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from repsteer.artifacts import ArtifactMetadata, SAEFeatureArtifact
from repsteer.core import Site
from repsteer.data import ContrastivePairs
from repsteer.learners.base import ContrastiveLearner, artifact_metadata
from repsteer.positions import LastNonPaddingToken

from .adapter import SAEAdapter, validate_adapter, validate_feature_id

CausalEvaluator = Callable[..., float | Tensor | Mapping[str, Any] | Sequence[float]]


def _encoded_features(
    sae: SAEAdapter,
    activations: Any,
    *,
    encoded: bool,
) -> Tensor:
    values = torch.as_tensor(activations)
    if values.ndim < 2:
        raise ValueError("SAE selection activations must have a feature dimension")
    expected = sae.num_features if encoded else sae.input_dim
    if values.shape[-1] != expected:
        kind = "encoded features" if encoded else "SAE inputs"
        raise ValueError(f"{kind} have width {values.shape[-1]}, expected {expected}")
    latents = values if encoded else sae.encode(values)
    if not isinstance(latents, Tensor):
        raise TypeError("SAEAdapter.encode must return a torch.Tensor")
    if latents.shape[:-1] != values.shape[:-1]:
        raise ValueError("SAEAdapter.encode must preserve all leading dimensions")
    if latents.shape[-1] != sae.num_features:
        raise ValueError(
            f"SAEAdapter.encode returned {latents.shape[-1]} features, "
            f"expected {sae.num_features}"
        )
    if not latents.is_floating_point():
        latents = latents.float()
    return latents.detach()


def _expand_per_example(
    value: Any,
    leading_shape: torch.Size,
    *,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    leading_count = math.prod(leading_shape)
    if tensor.numel() == leading_count:
        return tensor.reshape(-1)
    if leading_shape and tensor.numel() == leading_shape[0]:
        view_shape = (leading_shape[0],) + (1,) * (len(leading_shape) - 1)
        return tensor.reshape(view_shape).expand(leading_shape).reshape(-1)
    raise ValueError(
        f"{name} must have one value per activation or per first-axis example; "
        f"got {tensor.numel()} values for leading shape {tuple(leading_shape)}"
    )


def supervised_feature_scores(
    sae: SAEAdapter,
    activations: Any,
    labels: Any,
    *,
    sample_weight: Any | None = None,
    encoded: bool = False,
    scoring: str = "mean_difference",
    eps: float = 1e-6,
) -> Tensor:
    """Score every feature using labeled positive/negative activations.

    ``labels`` may contain one binary label per leading activation entry, or
    one label per first-axis example (which is expanded across token/patch
    axes). ``standardized_mean_difference`` divides the mean gap by pooled
    standard deviation; ``mean_difference`` preserves the SAE activation
    scale.
    """

    sae = validate_adapter(sae)
    if eps <= 0:
        raise ValueError("eps must be positive")
    latents = _encoded_features(sae, activations, encoded=encoded)
    matrix = latents.reshape(-1, latents.shape[-1]).to(dtype=torch.float32)
    target = _expand_per_example(
        labels,
        latents.shape[:-1],
        name="labels",
        dtype=torch.float32,
        device=matrix.device,
    )
    if not bool(((target == 0) | (target == 1)).all()):
        raise ValueError("supervised SAE selection requires binary labels 0/1")
    positive = target == 1
    negative = target == 0
    if not bool(positive.any()) or not bool(negative.any()):
        raise ValueError("supervised SAE selection requires both label classes")
    if sample_weight is None:
        weights = torch.ones_like(target)
    else:
        weights = _expand_per_example(
            sample_weight,
            latents.shape[:-1],
            name="sample_weight",
            dtype=torch.float32,
            device=matrix.device,
        )
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError("sample_weight values must be finite and non-negative")
    positive_weight = weights[positive].sum()
    negative_weight = weights[negative].sum()
    if float(positive_weight) <= 0 or float(negative_weight) <= 0:
        raise ValueError("each label class must have positive total sample weight")
    positive_mean = (matrix[positive] * weights[positive, None]).sum(
        dim=0
    ) / positive_weight
    negative_mean = (matrix[negative] * weights[negative, None]).sum(
        dim=0
    ) / negative_weight
    difference = positive_mean - negative_mean
    normalized_scoring = scoring.lower().replace("-", "_")
    if normalized_scoring in {"mean_difference", "mean_diff", "difference"}:
        return difference
    if normalized_scoring in {
        "standardized_mean_difference",
        "standardized",
        "effect_size",
    }:
        centered = matrix - ((matrix * weights[:, None]).sum(dim=0) / weights.sum())
        variance = (centered.square() * weights[:, None]).sum(dim=0) / weights.sum()
        return difference / variance.sqrt().clamp_min(eps)
    raise ValueError(
        "scoring must be 'mean_difference' or 'standardized_mean_difference'"
    )


def _feature_ids(
    sae: SAEAdapter,
    candidates: Sequence[int] | Tensor | None,
) -> tuple[int, ...]:
    if candidates is None:
        return tuple(range(sae.num_features))
    if isinstance(candidates, Tensor):
        candidates = candidates.detach().cpu().flatten().tolist()
    result: list[int] = []
    seen: set[int] = set()
    for value in candidates:
        feature_id = validate_feature_id(sae, int(value))
        if feature_id not in seen:
            seen.add(feature_id)
            result.append(feature_id)
    if not result:
        raise ValueError("feature candidates cannot be empty")
    return tuple(result)


def _rank(
    scores: Tensor,
    feature_ids: Sequence[int],
    *,
    top_k: int,
    absolute: bool,
) -> tuple[tuple[int, float], ...]:
    if isinstance(top_k, bool) or int(top_k) <= 0:
        raise ValueError("top_k must be positive")
    top_k = min(int(top_k), len(feature_ids))
    values = scores.detach().to(device="cpu", dtype=torch.float64).flatten()
    if values.numel() != len(feature_ids):
        raise ValueError("scores must contain one value per feature candidate")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("feature scores must be finite")
    ranked = sorted(
        zip(feature_ids, (float(item) for item in values), strict=False),
        key=lambda item: (
            -(abs(item[1]) if absolute else item[1]),
            item[0],
        ),
    )
    return tuple(ranked[:top_k])


def feature_artifact(
    sae: SAEAdapter,
    feature_id: int,
    *,
    metadata: ArtifactMetadata | None = None,
    score: float | None = None,
    criterion: str | None = None,
) -> SAEFeatureArtifact:
    """Create the portable artifact used by add, latent operators and gates."""

    sae = validate_adapter(sae)
    feature_id = validate_feature_id(sae, feature_id)
    direction = sae.decoder_direction(feature_id)
    if not isinstance(direction, Tensor):
        direction = torch.as_tensor(direction)
    if direction.ndim != 1 or direction.numel() != sae.input_dim:
        raise ValueError(
            "SAEAdapter.decoder_direction returned an incompatible vector: "
            f"{tuple(direction.shape)}"
        )
    if not direction.is_floating_point():
        direction = direction.float()
    base = metadata or ArtifactMetadata()
    config = dict(base.config)
    config.update(
        {
            "feature_id": feature_id,
            "sae_adapter": f"{type(sae).__module__}.{type(sae).__qualname__}",
            "sae_input_dim": int(sae.input_dim),
            "sae_num_features": int(sae.num_features),
        }
    )
    for name in ("release", "sae_id"):
        value = getattr(sae, name, None)
        if value is not None:
            config[name] = str(value)
    if criterion is not None:
        config["selection_criterion"] = str(criterion)
    if score is not None:
        config["selection_score"] = float(score)
    method = base.method or ("sae_feature" if criterion is None else f"sae_{criterion}")
    base = base.with_updates(
        artifact_type="sae_feature",
        method=method,
        config=config,
    )
    return SAEFeatureArtifact(
        metadata=base,
        feature_id=feature_id,
        decoder_direction=direction.detach().cpu().contiguous(),
        score=score,
    )


def select_supervised_features(
    sae: SAEAdapter,
    activations: Any,
    labels: Any,
    *,
    top_k: int = 1,
    candidates: Sequence[int] | Tensor | None = None,
    sample_weight: Any | None = None,
    encoded: bool = False,
    scoring: str = "mean_difference",
    absolute: bool = False,
    metadata: ArtifactMetadata | None = None,
) -> tuple[SAEFeatureArtifact, ...]:
    """Return top supervised features in deterministic score order."""

    sae = validate_adapter(sae)
    ids = _feature_ids(sae, candidates)
    all_scores = supervised_feature_scores(
        sae,
        activations,
        labels,
        sample_weight=sample_weight,
        encoded=encoded,
        scoring=scoring,
    )
    selected_scores = all_scores[
        torch.tensor(ids, device=all_scores.device, dtype=torch.long)
    ]
    ranked = _rank(
        selected_scores,
        ids,
        top_k=top_k,
        absolute=absolute,
    )
    return tuple(
        feature_artifact(
            sae,
            feature_id,
            metadata=metadata,
            score=score,
            criterion="supervised",
        )
        for feature_id, score in ranked
    )


def _invoke_evaluator(
    evaluator: CausalEvaluator | Mapping[int, Any],
    artifact: SAEFeatureArtifact,
) -> Any:
    if isinstance(evaluator, Mapping):
        try:
            return evaluator[artifact.feature_id]
        except KeyError as exc:
            raise KeyError(
                f"causal evaluator has no value for feature {artifact.feature_id}"
            ) from exc
    if not callable(evaluator):
        raise TypeError(
            "causal evaluator must be callable or a feature->effect mapping"
        )
    try:
        signature = inspect.signature(evaluator)
    except (TypeError, ValueError):
        return evaluator(artifact)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(
        parameter.kind == parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    if has_varargs or len(positional) >= 2:
        return evaluator(artifact.feature_id, artifact)
    return evaluator(artifact)


def _causal_effect(value: Any) -> float:
    if isinstance(value, Mapping):
        if "effect" in value:
            value = value["effect"]
        elif "intervention" in value and "baseline" in value:
            value = value["intervention"] - value["baseline"]
        else:
            raise ValueError(
                "causal evaluator mappings need 'effect' or "
                "'intervention' and 'baseline'"
            )
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        if len(value) != 2:
            raise ValueError(
                "causal evaluator sequences must be (intervention, baseline)"
            )
        value = value[0] - value[1]
    tensor = torch.as_tensor(value)
    if tensor.numel() != 1:
        raise ValueError("causal evaluator must return one scalar effect per feature")
    effect = float(tensor.detach().cpu())
    if not math.isfinite(effect):
        raise ValueError("causal evaluator returned a non-finite effect")
    return effect


def select_causal_features(
    sae: SAEAdapter,
    evaluator: CausalEvaluator | Mapping[int, Any],
    *,
    candidates: Sequence[int] | Tensor | None = None,
    top_k: int = 1,
    absolute: bool = False,
    metadata: ArtifactMetadata | None = None,
) -> tuple[SAEFeatureArtifact, ...]:
    """Rank candidates by measured marginal causal effect."""

    sae = validate_adapter(sae)
    ids = _feature_ids(sae, candidates)
    scored: list[tuple[SAEFeatureArtifact, float]] = []
    for feature_id in ids:
        candidate = feature_artifact(
            sae,
            feature_id,
            metadata=metadata,
            criterion="causal_effect",
        )
        effect = _causal_effect(_invoke_evaluator(evaluator, candidate))
        scored.append((candidate, effect))
    score_tensor = torch.tensor([effect for _, effect in scored], dtype=torch.float64)
    ranked = _rank(score_tensor, ids, top_k=top_k, absolute=absolute)
    return tuple(
        feature_artifact(
            sae,
            feature_id,
            metadata=metadata,
            score=effect,
            criterion="causal_effect",
        )
        for feature_id, effect in ranked
    )


def _from_data(data: Any, name: str) -> Any | None:
    if data is None:
        return None
    if isinstance(data, Mapping):
        return data.get(name)
    return getattr(data, name, None)


def _metadata_with_fallbacks(
    explicit: ArtifactMetadata | None,
    inferred: ArtifactMetadata,
) -> ArtifactMetadata:
    """Preserve explicit provenance while filling its unset identity fields."""

    if explicit is None:
        return inferred
    return explicit.with_updates(
        model_id=explicit.model_id or inferred.model_id,
        model_revision=(
            explicit.model_revision
            if explicit.model_revision is not None
            else inferred.model_revision
        ),
        site=explicit.site if explicit.site is not None else inferred.site,
        hidden_size=explicit.hidden_size or inferred.hidden_size,
        method=explicit.method or inferred.method,
        architecture=explicit.architecture or inferred.architecture,
        tokenizer=explicit.tokenizer or inferred.tokenizer,
        processor=explicit.processor or inferred.processor,
        modality=explicit.modality or inferred.modality,
        dataset_fingerprint=(
            explicit.dataset_fingerprint or inferred.dataset_fingerprint
        ),
        dtype=explicit.dtype or inferred.dtype,
        normalization=(
            explicit.normalization
            if explicit.normalization is not None
            else inferred.normalization
        ),
        seed=explicit.seed if explicit.seed is not None else inferred.seed,
        config={**dict(inferred.config), **dict(explicit.config)},
        provenance={**dict(inferred.provenance), **dict(explicit.provenance)},
    )


def _contrastive_capture_setup(
    *,
    sae: SAEAdapter,
    model: Any,
    data: ContrastivePairs,
    site: Any,
    positions: Any,
    batch_size: int | None,
    seed: int,
    metadata: ArtifactMetadata | None,
) -> tuple[ContrastiveLearner, ArtifactMetadata]:
    if site is None:
        raise ValueError(
            "automatic SAE capture needs a semantic site. Pass site=... to "
            "select_feature(), SAELensAdapter(...), or FunctionalSAEAdapter(...)."
        )
    resolve_site = getattr(model, "resolve_site", None)
    if callable(resolve_site):
        resolved = resolve_site(site)
        hidden_dim = getattr(resolved, "hidden_dim", None)
        if hidden_dim is not None and int(hidden_dim) != int(sae.input_dim):
            raise ValueError(
                f"SAE input_dim {sae.input_dim} does not match the resolved "
                f"site hidden dimension {hidden_dim}"
            )
    learner = ContrastiveLearner(
        site=site,
        positions=positions,
        pooling="identity",
        normalize=None,
        seed=seed,
        batch_size=batch_size,
    )
    inferred = artifact_metadata(
        learner=learner,
        model=model,
        data=data,
        artifact_type="sae_feature",
        hidden_size=int(sae.input_dim),
        extra_config={"selection_input": "contrastive_capture"},
    ).with_updates(
        # The decoder direction, rather than the float32 selection matrix,
        # determines the portable artifact dtype. feature_artifact fills it.
        dtype=None,
        method="",
    )
    return learner, _metadata_with_fallbacks(metadata, inferred)


def _capture_contrastive_activations(
    learner: ContrastiveLearner,
    model: Any,
    data: ContrastivePairs,
) -> tuple[Tensor, Tensor, Tensor]:
    positive, negative = learner._capture_pair(model, data, store=None)
    positive_weights = (
        torch.ones(
            len(positive),
            dtype=torch.float32,
            device=positive.activations.device,
        )
        if positive.sample_weights is None
        else torch.as_tensor(
            positive.sample_weights,
            dtype=torch.float32,
            device=positive.activations.device,
        ).flatten()
    )
    negative_weights = (
        torch.ones(
            len(negative),
            dtype=torch.float32,
            device=negative.activations.device,
        )
        if negative.sample_weights is None
        else torch.as_tensor(
            negative.sample_weights,
            dtype=torch.float32,
            device=negative.activations.device,
        ).flatten()
    )
    if positive.activations.device != negative.activations.device:
        raise ValueError(
            "positive and negative SAE captures must be on the same device"
        )
    return (
        positive.activations,
        negative.activations,
        torch.cat((positive_weights, negative_weights)),
    )


def select_feature(
    *,
    sae: SAEAdapter,
    model: Any | None = None,
    data: Any | None = None,
    criterion: str = "supervised",
    top_k: int = 20,
    activations: Any | None = None,
    labels: Any | None = None,
    positive_activations: Any | None = None,
    negative_activations: Any | None = None,
    sample_weight: Any | None = None,
    encoded: bool = False,
    candidates: Sequence[int] | Tensor | None = None,
    evaluator: CausalEvaluator | Mapping[int, Any] | None = None,
    causal_evaluator: CausalEvaluator | Mapping[int, Any] | None = None,
    scoring: str = "mean_difference",
    absolute: bool = False,
    metadata: ArtifactMetadata | None = None,
    site: Any | None = None,
    positions: Any | None = None,
    batch_size: int | None = None,
    seed: int = 42,
) -> SAEFeatureArtifact:
    """Select one feature using the high-level RFC API.

    When ``model`` and :class:`ContrastivePairs` are supplied and no activation
    tensors are provided, the function captures both sides at ``site`` (or the
    adapter's inferred ``site``), selecting ``LastNonPaddingToken`` by default.
    The resulting artifact receives model, site, dataset and capture
    provenance. Direct activation tensors remain the lower-level fast path.

    For ``criterion="causal_effect"``, ``top_k`` is the supervised shortlist
    size, not the number of returned artifacts.  Provide an ``evaluator``
    callback (or a model exposing ``evaluate_sae_feature``) to measure each
    shortlisted artifact.
    """

    sae = validate_adapter(sae)
    capture_site = site if site is not None else getattr(sae, "site", None)
    if capture_site is not None and not isinstance(capture_site, Site):
        raise TypeError("select_feature site must be a Site or None")
    capture_positions = LastNonPaddingToken() if positions is None else positions
    if activations is None:
        activations = _from_data(data, "activations")
    if labels is None:
        labels = _from_data(data, "labels")
    if positive_activations is None:
        positive_activations = _from_data(data, "positive_activations")
    if negative_activations is None:
        negative_activations = _from_data(data, "negative_activations")
    if sample_weight is None:
        sample_weight = _from_data(data, "sample_weight")

    capture_learner: ContrastiveLearner | None = None
    if (
        activations is None
        and positive_activations is None
        and model is not None
        and isinstance(data, ContrastivePairs)
        and capture_site is not None
    ):
        capture_learner, metadata = _contrastive_capture_setup(
            sae=sae,
            model=model,
            data=data,
            site=capture_site,
            positions=capture_positions,
            batch_size=batch_size,
            seed=int(seed),
            metadata=metadata,
        )

    if (
        activations is None
        and positive_activations is None
        and capture_learner is not None
        and isinstance(data, ContrastivePairs)
    ):
        (
            positive_activations,
            negative_activations,
            captured_weights,
        ) = _capture_contrastive_activations(capture_learner, model, data)
        if sample_weight is None:
            sample_weight = captured_weights

    if activations is None and positive_activations is not None:
        if negative_activations is None:
            raise ValueError(
                "negative_activations are required with positive_activations"
            )
        positive = torch.as_tensor(positive_activations)
        negative = torch.as_tensor(negative_activations)
        if positive.shape[1:] != negative.shape[1:]:
            raise ValueError(
                "positive and negative activations must have matching non-batch shapes"
            )
        activations = torch.cat((positive, negative), dim=0)
        labels = torch.cat(
            (
                torch.ones(positive.shape[0]),
                torch.zeros(negative.shape[0]),
            )
        )

    if activations is None and model is not None:
        capture = getattr(model, "capture_sae_activations", None)
        if callable(capture):
            captured = capture(data, sae=sae)
            if isinstance(captured, Mapping):
                activations = captured.get("activations")
                labels = captured.get("labels", labels)
                sample_weight = captured.get("sample_weight", sample_weight)
            else:
                activations = captured

    normalized = criterion.lower().replace("-", "_")
    if normalized in {"supervised", "supervised_separation", "separation"}:
        if activations is None or labels is None:
            raise ValueError(
                "supervised feature selection needs activations and binary labels"
            )
        return select_supervised_features(
            sae,
            activations,
            labels,
            top_k=1,
            candidates=candidates,
            sample_weight=sample_weight,
            encoded=encoded,
            scoring=scoring,
            absolute=absolute,
            metadata=metadata,
        )[0]
    if normalized not in {"causal", "causal_effect", "causal_effects"}:
        raise ValueError("criterion must be 'supervised' or 'causal_effect'")

    causal = causal_evaluator if causal_evaluator is not None else evaluator
    if causal is None:
        causal = _from_data(data, "causal_effects")
    if causal is None and model is not None:
        evaluate = getattr(model, "evaluate_sae_feature", None)
        if callable(evaluate):

            def evaluate_feature(artifact: SAEFeatureArtifact) -> Any:
                return evaluate(artifact, data=data)

            causal = evaluate_feature
    if causal is None:
        raise ValueError(
            "causal feature selection needs evaluator=..., a causal_effects "
            "mapping, or model.evaluate_sae_feature()"
        )

    shortlist = candidates
    if shortlist is None and activations is not None and labels is not None:
        shortlist = tuple(
            artifact.feature_id
            for artifact in select_supervised_features(
                sae,
                activations,
                labels,
                top_k=top_k,
                sample_weight=sample_weight,
                encoded=encoded,
                scoring=scoring,
                absolute=absolute,
                metadata=metadata,
            )
        )
    elif shortlist is None:
        if isinstance(causal, Mapping):
            shortlist = tuple(int(key) for key in causal)
        else:
            raise ValueError(
                "causal selection needs candidates or labeled activations for "
                "a supervised shortlist"
            )
    return select_causal_features(
        sae,
        causal,
        candidates=shortlist,
        top_k=1,
        absolute=absolute,
        metadata=metadata,
    )[0]


__all__ = [
    "CausalEvaluator",
    "feature_artifact",
    "select_causal_features",
    "select_feature",
    "select_supervised_features",
    "supervised_feature_scores",
]
