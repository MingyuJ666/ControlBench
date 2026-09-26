"""Cache-key construction for activation capture."""

from __future__ import annotations

from typing import Any

from repsteer.data.fingerprint import Fingerprint, stable_fingerprint

from .request import CaptureRequest


def capture_cache_key(model: Any, request: CaptureRequest) -> Fingerprint:
    """Build a key that includes model, rendering, data, site and dtype state."""

    config = getattr(model, "config", None)
    tokenizer = getattr(model, "tokenizer", None)
    processor = getattr(model, "processor", None)
    payload = {
        "kind": "activation_capture",
        "model": {
            "id": _first(
                getattr(model, "model_id", None),
                getattr(model, "name_or_path", None),
                getattr(config, "_name_or_path", None),
                f"{type(model).__module__}.{type(model).__qualname__}",
            ),
            "revision": _first(
                getattr(model, "revision", None),
                getattr(config, "_commit_hash", None),
            ),
            "architecture": type(model).__qualname__,
            "training": getattr(model, "training", None),
            "dtype": _first(request.dtype, getattr(model, "dtype", None)),
        },
        "tokenizer": {
            "id": _first(
                getattr(tokenizer, "name_or_path", None),
                getattr(processor, "name_or_path", None),
            ),
            "revision": _first(
                getattr(tokenizer, "revision", None),
                getattr(processor, "revision", None),
            ),
            "chat_template": _first(
                getattr(tokenizer, "chat_template", None),
                getattr(processor, "chat_template", None),
            ),
        },
        "request": str(request.fingerprint),
    }
    return stable_fingerprint(payload)


def _first(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


__all__ = ["capture_cache_key"]
