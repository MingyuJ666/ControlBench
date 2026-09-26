"""Lazy SAE provider registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .adapter import SAEAdapter, validate_adapter

SAELoader = Callable[..., SAEAdapter]
_LOADERS: dict[str, SAELoader] = {}


def _provider_name(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "").replace("_", "")
    if not normalized:
        raise ValueError("SAE provider cannot be empty")
    return normalized


def register_provider(
    provider: str,
    loader: SAELoader,
    *,
    overwrite: bool = False,
) -> None:
    """Register a project-local SAE provider without importing it eagerly."""

    name = _provider_name(provider)
    if not callable(loader):
        raise TypeError("SAE provider loader must be callable")
    if name in _LOADERS and not overwrite:
        raise ValueError(f"SAE provider {provider!r} is already registered")
    _LOADERS[name] = loader


def unregister_provider(provider: str) -> None:
    _LOADERS.pop(_provider_name(provider), None)


def available_providers() -> tuple[str, ...]:
    return tuple(sorted({"saelens", *_LOADERS}))


def load(*, provider: str = "saelens", **kwargs: Any) -> SAEAdapter:
    """Load an SAE through a named optional integration."""

    name = _provider_name(provider)
    if name == "saelens":
        from .saelens import SAELensAdapter

        return validate_adapter(SAELensAdapter.load(**kwargs))
    try:
        loader = _LOADERS[name]
    except KeyError as exc:
        supported = ", ".join(available_providers())
        raise ValueError(
            f"Unknown SAE provider {provider!r}; available providers: {supported}"
        ) from exc
    return validate_adapter(loader(**kwargs))


__all__ = [
    "SAELoader",
    "available_providers",
    "load",
    "register_provider",
    "unregister_provider",
]
