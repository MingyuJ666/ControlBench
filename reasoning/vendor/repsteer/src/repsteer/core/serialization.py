"""Small JSON-safe conversion helpers used by public immutable objects."""

from __future__ import annotations

import dataclasses
import enum
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch


def json_safe(value: Any) -> Any:
    """Convert configuration data to JSON primitives without executing code."""

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, enum.Enum):
        return json_safe(value.value)
    if isinstance(value, torch.dtype | torch.device | Path):
        return str(value)
    if isinstance(value, slice):
        return {
            "__type__": "slice",
            "start": value.start,
            "stop": value.stop,
            "step": value.step,
        }
    if isinstance(value, torch.Tensor):
        if value.numel() > 1024:
            raise TypeError("Large tensors must be stored in tensors.safetensors")
        return value.detach().cpu().tolist()
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return json_safe(value.to_dict())
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(cast(Any, value)))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, set | frozenset):
        items = [json_safe(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ),
        )
    if isinstance(value, tuple | list):
        return [json_safe(item) for item in value]
    raise TypeError(
        f"Value of type {type(value).__name__} is not safely JSON serializable"
    )


__all__ = ["json_safe"]
