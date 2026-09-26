"""Deterministic multimodal processor identity and preprocessing metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from repsteer.data import canonicalize, stable_fingerprint


def _get(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _first(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def processor_metadata(target: Any) -> dict[str, Any]:
    """Describe a wrapper's processor using the same fields as artifacts."""

    processor = _get(target, "processor")
    explicit_id = _get(target, "processor_id")
    explicit_revision = _get(target, "processor_revision")
    explicit_fingerprint = _get(target, "processor_preprocess_fingerprint")
    if processor is None:
        if not (explicit_id or explicit_revision or explicit_fingerprint):
            return {}
        return {
            "id": None if explicit_id is None else str(explicit_id),
            "revision": (None if explicit_revision is None else str(explicit_revision)),
            "preprocess_fingerprint": (
                None if explicit_fingerprint is None else str(explicit_fingerprint)
            ),
            "config": {},
        }
    init_kwargs = _get(processor, "init_kwargs")
    if not isinstance(init_kwargs, Mapping):
        init_kwargs = {}
    identifier = _first(
        explicit_id,
        _get(processor, "name_or_path"),
        _get(processor, "_name_or_path"),
        init_kwargs.get("name_or_path"),
        init_kwargs.get("pretrained_model_name_or_path"),
        type(processor).__qualname__,
    )
    revision = _first(
        explicit_revision,
        _get(processor, "_commit_hash"),
        _get(processor, "revision"),
        init_kwargs.get("_commit_hash"),
        init_kwargs.get("revision"),
    )
    image_processor = _get(processor, "image_processor")
    tokenizer = _get(processor, "tokenizer")
    preprocessing: dict[str, Any] = {
        "processor_type": type(processor).__qualname__,
        "image_processor_type": (
            None if image_processor is None else type(image_processor).__qualname__
        ),
    }
    for owner_name, owner in (
        ("processor", processor),
        ("image_processor", image_processor),
    ):
        if owner is None:
            continue
        for name in ("size", "crop_size", "resample", "image_mean", "image_std"):
            value = _get(owner, name)
            if value is not None:
                preprocessing[f"{owner_name}.{name}"] = canonicalize(value)
    chat_template = _first(
        _get(processor, "chat_template"), _get(tokenizer, "chat_template")
    )
    if chat_template is not None:
        preprocessing["chat_template_sha256"] = str(stable_fingerprint(chat_template))
    return {
        "id": str(identifier),
        "revision": None if revision is None else str(revision),
        "preprocess_fingerprint": (
            str(explicit_fingerprint)
            if explicit_fingerprint is not None
            else str(stable_fingerprint(preprocessing))
        ),
        "config": preprocessing,
    }


__all__ = ["processor_metadata"]
