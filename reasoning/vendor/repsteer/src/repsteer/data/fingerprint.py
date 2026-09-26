"""Deterministic fingerprints for steering datasets and capture requests."""

from __future__ import annotations

import base64
import dataclasses
import enum
import functools
import hashlib
import json
import marshal
import math
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


class Fingerprint(str):
    """A string fingerprint that is also callable for API compatibility.

    Both ``dataset.fingerprint`` and ``dataset.fingerprint()`` therefore work.
    """

    def __call__(self) -> str:
        return str(self)


def canonicalize(value: Any) -> Any:
    """Convert *value* to a JSON-safe, deterministic representation.

    The function intentionally records Python/dataclass types.  Two selectors with
    identical fields fingerprint alike, while semantically different selector
    classes do not accidentally share cache entries.
    """

    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "inf" if value > 0 else "-inf"}
        # Normalize negative zero, whose distinction is irrelevant to configs.
        return 0.0 if value == 0.0 else value
    if isinstance(value, enum.Enum):
        return {
            "__enum__": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": canonicalize(value.value),
        }
    if isinstance(value, Path):
        return {"__path__": value.as_posix()}
    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "__tensor__": True,
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "values": canonicalize(tensor.tolist()),
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                field.name: canonicalize(getattr(value, field.name))
                for field in dataclasses.fields(value)
            },
        }
    if isinstance(value, Mapping):
        # JSON object keys need not be strings in user metadata.  Represent the
        # mapping as sorted key/value pairs so all key types remain deterministic.
        items = [(canonicalize(key), canonicalize(item)) for key, item in value.items()]
        items.sort(key=lambda pair: _json_dumps(pair[0]))
        return {"__mapping__": [[key, item] for key, item in items]}
    if isinstance(value, set | frozenset):
        items = [canonicalize(item) for item in value]
        items.sort(key=_json_dumps)
        return {"__set__": items}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [canonicalize(item) for item in value]

    # Functions need their implementation and bound values in the fingerprint.
    # Falling through to ``function.__dict__`` would make all ordinary lambdas
    # look identical and could return a cached activation pooled by different
    # code.  Marshal is only used as deterministic code-object bytes; nothing is
    # ever deserialized or executed here.
    if isinstance(value, types.FunctionType):
        closure: list[Any] = []
        for cell in value.__closure__ or ():
            try:
                captured = cell.cell_contents
            except ValueError:
                closure.append({"__empty_cell__": True})
                continue
            if captured is value:
                closure.append({"__self_reference__": True})
            elif callable(captured):
                reference = getattr(
                    captured, "__qualname__", type(captured).__qualname__
                )
                closure.append(
                    {
                        "__callable_reference__": (
                            f"{getattr(captured, '__module__', '')}.{reference}"
                        )
                    }
                )
            else:
                closure.append(canonicalize(captured))
        return {
            "__function__": f"{value.__module__}.{value.__qualname__}",
            "code_sha256": hashlib.sha256(marshal.dumps(value.__code__)).hexdigest(),
            "defaults": canonicalize(value.__defaults__),
            "kwdefaults": canonicalize(value.__kwdefaults__),
            "closure": closure,
        }
    if isinstance(value, types.MethodType):
        return {
            "__method__": canonicalize(value.__func__),
            "self": canonicalize(value.__self__),
        }
    if isinstance(value, functools.partial):
        return {
            "__partial__": canonicalize(value.func),
            "args": canonicalize(value.args),
            "keywords": canonicalize(value.keywords),
        }
    if isinstance(value, types.BuiltinFunctionType | types.BuiltinMethodType):
        return {
            "__builtin_callable__": (
                f"{getattr(value, '__module__', '')}."
                f"{getattr(value, '__qualname__', type(value).__qualname__)}"
            )
        }

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return {
                "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
                "value": canonicalize(to_dict()),
            }
        except (TypeError, ValueError):
            pass

    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, Mapping):
        public = {
            str(key): canonicalize(item)
            for key, item in attributes.items()
            if not str(key).startswith("_") and not callable(item)
        }
        return {
            "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
            "attributes": public,
        }

    # Avoid repr(), which commonly embeds a process-specific memory address.
    return {"__type__": f"{type(value).__module__}.{type(value).__qualname__}"}


def _json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_json(value: Any) -> str:
    """Return the stable JSON encoding used by :func:`stable_fingerprint`."""

    return _json_dumps(canonicalize(value))


def stable_fingerprint(value: Any, *, prefix: str = "sha256:") -> Fingerprint:
    """Hash arbitrary nested records without depending on insertion order."""

    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return Fingerprint(f"{prefix}{digest}")


__all__ = ["Fingerprint", "canonical_json", "canonicalize", "stable_fingerprint"]
