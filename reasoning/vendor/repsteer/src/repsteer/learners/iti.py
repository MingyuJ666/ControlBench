"""Inference-Time Intervention (ITI) profile learning.

This follows Li et al. (NeurIPS 2023): train one binary linear probe per
attention query head, rank heads by held-out probe accuracy, then apply a
unit mass-mean direction scaled by that head's projected activation standard
deviation. The runtime edit itself lives at ``self_attn.o_proj.input``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import torch
from torch import Tensor

from repsteer.artifacts import ITIArtifact, ITIHead
from repsteer.capture import ActivationStore, CaptureRequest, capture_activations
from repsteer.data import ContrastivePairs
from repsteer.positions import LastNonPaddingToken
from repsteer.sites import head_result

from .base import (
    ContrastiveLearner,
    activation_matrix,
    artifact_metadata,
    sample_weights,
    weighted_mean,
)
from .probe import fit_logistic_probe


def _validation_count(samples: int, fraction: float) -> int:
    if samples < 2:
        raise ValueError(
            "ITI needs at least two positive and two negative examples to make a "
            "held-out probe split"
        )
    return min(max(1, round(samples * fraction)), samples - 1)


def _stratified_indices(
    positive_samples: int,
    negative_samples: int,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Return deterministic, class-stratified train and validation indices."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    positive = torch.randperm(positive_samples, generator=generator)
    negative = torch.randperm(negative_samples, generator=generator)
    positive_validation = _validation_count(positive_samples, validation_fraction)
    negative_validation = _validation_count(negative_samples, validation_fraction)
    train = torch.cat(
        (
            positive[positive_validation:],
            negative[negative_validation:] + positive_samples,
        )
    )
    validation = torch.cat(
        (
            positive[:positive_validation],
            negative[:negative_validation] + positive_samples,
        )
    )
    return train, validation


def _layer_count(model: Any) -> int:
    config = getattr(model, "config", None)
    for owner in (
        config,
        getattr(config, "text_config", None),
        getattr(config, "llm_config", None),
    ):
        value = getattr(owner, "num_hidden_layers", None)
        if value is None:
            continue
        if isinstance(value, bool) or int(value) <= 0:
            break
        return int(value)
    raise ValueError("ITI requires a model with positive config.num_hidden_layers")


@dataclass
class ITI(ContrastiveLearner):
    """Learn an ITI head profile from truthful (positive) and false examples.

    ``positive`` and ``negative`` inputs in :class:`ContrastivePairs` are
    treated as labels 1 and 0. The default ``mass_mean`` direction is the
    paper's best-performing direction: the positive activation mean minus the
    negative mean, normalized per selected head. ``probe_weight`` is available
    for the paper's ablation, while ranking always uses held-out probe accuracy.
    """

    site: Any = None
    positions: Any = field(default_factory=LastNonPaddingToken)
    pooling: Any = "last"
    normalize: str | None = "l2"
    top_k: int = 48
    validation_fraction: float = 0.2
    direction: Literal["mass_mean", "probe_weight"] = "mass_mean"
    std_correction: int = 1
    method: ClassVar[str] = "iti"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.site is not None:
            raise ValueError(
                "ITI learns all layers' head_result surfaces; do not pass site=..."
            )
        if self.normalize not in {"l2", "unit", "unit_l2"}:
            raise ValueError("ITI requires l2-normalized directions")
        if isinstance(self.top_k, bool) or int(self.top_k) <= 0:
            raise ValueError("ITI.top_k must be a positive integer")
        self.top_k = int(self.top_k)
        self.validation_fraction = float(self.validation_fraction)
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("ITI.validation_fraction must be strictly between 0 and 1")
        if self.direction not in {"mass_mean", "probe_weight"}:
            raise ValueError("ITI.direction must be 'mass_mean' or 'probe_weight'")
        if self.std_correction not in {0, 1}:
            raise ValueError("ITI.std_correction must be 0 (population) or 1 (sample)")

    def _capture_side(
        self,
        model: Any,
        data: ContrastivePairs,
        *,
        layer: int,
        positive: bool,
        store: ActivationStore | None,
    ) -> tuple[Tensor, Tensor]:
        examples = data.positive_examples if positive else data.negative_examples
        request = CaptureRequest(
            inputs=[example.input for example in examples],
            site=head_result(layer),
            positions=self.positions,
            pooling=self.pooling,
            batch_size=self.batch_size,
            sample_weights=[example.weight for example in examples],
            dataset_fingerprint=str(data.fingerprint),
            apply_chat_template=self._resolved_apply_chat_template(data),
            add_generation_prompt=self.add_generation_prompt,
            system_prompt=self.system_prompt,
            chat_template_kwargs=self.chat_template_kwargs,
            special_tokens=self.special_tokens,
            dtype=self.capture_dtype,
            seed=self.seed,
            metadata={
                "contrastive_side": "positive" if positive else "negative",
                "iti_layer": layer,
                "iti_surface": "self_attn.o_proj.input",
            },
        )
        captured = capture_activations(model, request, store=store)
        matrix = activation_matrix(
            captured,
            name=("positive" if positive else "negative") + " ITI head results",
        )
        return matrix, sample_weights(captured, matrix)

    def fit(
        self,
        model: Any,
        data: ContrastivePairs,
        *,
        store: ActivationStore | None = None,
        calibration_data: ContrastivePairs | None = None,
    ) -> ITIArtifact:
        """Fit probes/directions and return a portable selected-head profile.

        ``calibration_data`` optionally supplies the activations used only to
        estimate each selected direction's projected standard deviation. When
        omitted, the full learning dataset (train plus validation) is used, as
        in the reference implementation.
        """

        capabilities = getattr(getattr(model, "adapter", None), "capabilities", None)
        require = getattr(capabilities, "require", None)
        if not callable(require):
            raise TypeError(
                "ITI requires a model adapter declaring the head_result surface"
            )
        require("head_result")

        layers = _layer_count(model)
        first_site = head_result(0)
        try:
            head_dim = int(model.resolve_site(first_site).hidden_dim)
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError(
                "ITI requires resolvable per-layer head_result sites"
            ) from exc
        if head_dim <= 0:
            raise ValueError("resolved ITI head_result dimension must be positive")

        train_indices, validation_indices = _stratified_indices(
            len(data.positive_examples),
            len(data.negative_examples),
            validation_fraction=self.validation_fraction,
            seed=self.seed,
        )
        calibration = data if calibration_data is None else calibration_data
        candidates: list[tuple[float, int, int, Tensor, float]] = []
        expected_heads: int | None = None

        for layer in range(layers):
            resolved = model.resolve_site(head_result(layer))
            if int(resolved.hidden_dim) != head_dim:
                raise ValueError(
                    "ITI requires the same query-head dimension in every layer"
                )
            positive, positive_weights = self._capture_side(
                model, data, layer=layer, positive=True, store=store
            )
            negative, negative_weights = self._capture_side(
                model, data, layer=layer, positive=False, store=store
            )
            if positive.shape[-1] % head_dim or negative.shape[-1] % head_dim:
                raise ValueError(
                    "captured o_proj input width must be divisible by the resolved "
                    f"head dimension {head_dim}"
                )
            num_heads = positive.shape[-1] // head_dim
            if negative.shape[-1] // head_dim != num_heads:
                raise ValueError(
                    "positive and negative ITI captures disagree on head layout"
                )
            if expected_heads is None:
                expected_heads = num_heads
            elif expected_heads != num_heads:
                raise ValueError(
                    "ITI requires the same query-head count in every layer"
                )

            features = torch.cat((positive, negative), dim=0).reshape(
                positive.shape[0] + negative.shape[0], num_heads, head_dim
            )
            labels = torch.cat(
                (
                    torch.ones(positive.shape[0], dtype=torch.float32),
                    torch.zeros(negative.shape[0], dtype=torch.float32),
                )
            )
            weights = torch.cat((positive_weights, negative_weights)).to(
                device=features.device, dtype=features.dtype
            )
            calibration_positive = calibration_negative = None
            if calibration is not data:
                calibration_positive, _ = self._capture_side(
                    model, calibration, layer=layer, positive=True, store=store
                )
                calibration_negative, _ = self._capture_side(
                    model, calibration, layer=layer, positive=False, store=store
                )

            for head in range(num_heads):
                values = features[:, head, :]
                probe_weight, probe_bias = fit_logistic_probe(
                    values[train_indices],
                    labels[train_indices],
                    sample_weight=weights[train_indices],
                )
                validation_logits = (
                    values[validation_indices].cpu() @ probe_weight + probe_bias
                )
                accuracy = float(
                    ((validation_logits >= 0) == labels[validation_indices].cpu())
                    .float()
                    .mean()
                )
                if self.direction == "mass_mean":
                    positive_mean = weighted_mean(
                        positive.reshape(positive.shape[0], num_heads, head_dim)[
                            :, head
                        ],
                        positive_weights,
                    )
                    negative_mean = weighted_mean(
                        negative.reshape(negative.shape[0], num_heads, head_dim)[
                            :, head
                        ],
                        negative_weights,
                    )
                    vector = positive_mean - negative_mean
                else:
                    vector = probe_weight.to(device=values.device, dtype=values.dtype)
                norm = torch.linalg.vector_norm(vector)
                if not bool(torch.isfinite(norm)) or float(norm) == 0.0:
                    continue
                direction = vector / norm
                if calibration_positive is None or calibration_negative is None:
                    scale_values = values
                else:
                    if (
                        calibration_positive.shape[-1] != positive.shape[-1]
                        or calibration_negative.shape[-1] != positive.shape[-1]
                    ):
                        raise ValueError(
                            "calibration captures disagree with ITI head layout"
                        )
                    scale_values = torch.cat(
                        (calibration_positive, calibration_negative), dim=0
                    ).reshape(-1, num_heads, head_dim)[:, head, :]
                projected_std = float(
                    torch.std(
                        scale_values
                        @ direction.to(
                            device=scale_values.device, dtype=scale_values.dtype
                        ),
                        correction=self.std_correction,
                    )
                )
                if (
                    not torch.isfinite(torch.tensor(projected_std))
                    or projected_std <= 0
                ):
                    continue
                candidates.append(
                    (
                        accuracy,
                        layer,
                        head,
                        direction.detach().to(device="cpu", dtype=torch.float32),
                        projected_std,
                    )
                )

        if not candidates:
            raise ValueError("ITI found no finite, non-zero calibrated head directions")
        if self.top_k > len(candidates):
            raise ValueError(
                f"ITI.top_k={self.top_k} exceeds the {len(candidates)} usable heads"
            )
        selected = sorted(candidates, key=lambda item: (-item[0], item[1], item[2]))[
            : self.top_k
        ]
        heads = tuple(
            ITIHead(
                rank=rank,
                layer=layer,
                head=head,
                validation_accuracy=accuracy,
                projected_std=projected_std,
            )
            for rank, (accuracy, layer, head, _direction, projected_std) in enumerate(
                selected
            )
        )
        directions = torch.stack([item[3] for item in selected])
        metadata = artifact_metadata(
            learner=self,
            model=model,
            data=data,
            artifact_type="iti",
            hidden_size=head_dim,
            extra_config={
                "head_surface": "self_attn.o_proj.input",
                "head_axis": "query",
                "num_layers": layers,
                "num_attention_heads": expected_heads,
                "top_k": self.top_k,
                "validation_fraction": self.validation_fraction,
                "selection_metric": "held_out_accuracy",
                "direction": self.direction,
                "std_correction": self.std_correction,
                "calibration_dataset_fingerprint": str(calibration.fingerprint),
            },
        )
        return ITIArtifact(metadata, directions, heads)


__all__ = ["ITI"]
