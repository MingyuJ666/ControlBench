from __future__ import annotations

from collections.abc import Iterable
from threading import RLock

from torch import nn

from .base import ArchitectureAdapter
from .gemma import GemmaAdapter
from .llama import LlamaAdapter
from .mistral import MistralAdapter
from .qwen2 import Qwen2Adapter
from .qwen3 import Qwen3Adapter
from .vlm import InternVLAdapter, Qwen2_5_VLAdapter


def _unsupported(message: str) -> Exception:
    try:
        from repsteer.core.errors import UnsupportedArchitectureError

        return UnsupportedArchitectureError(message)
    except (ImportError, AttributeError):
        return ValueError(message)


class AdapterRegistry:
    def __init__(self, adapters: Iterable[ArchitectureAdapter] = ()) -> None:
        self._adapters = list(adapters)
        self._lock = RLock()

    def register(self, adapter: ArchitectureAdapter, *, prepend: bool = False) -> None:
        if not isinstance(adapter, ArchitectureAdapter):
            # ABC isinstance gives a useful guard while still allowing users to
            # subclass the public base without importing Transformers.
            required = ("supports", "resolve", "hidden_size")
            if not all(callable(getattr(adapter, name, None)) for name in required):
                raise TypeError("adapter must implement supports/resolve/hidden_size")
        with self._lock:
            if prepend:
                self._adapters.insert(0, adapter)
            else:
                self._adapters.append(adapter)

    def unregister(self, adapter: ArchitectureAdapter) -> None:
        with self._lock:
            self._adapters.remove(adapter)

    def resolve(self, model: nn.Module) -> ArchitectureAdapter:
        with self._lock:
            adapters = tuple(self._adapters)
        for adapter in adapters:
            try:
                if adapter.supports(model):
                    return adapter
            except (AttributeError, TypeError):
                continue
        config = getattr(model, "config", None)
        model_type = getattr(config, "model_type", None)
        available = ", ".join(adapter.architecture_name for adapter in adapters)
        raise _unsupported(
            f"unsupported Hugging Face architecture {type(model).__name__} "
            f"(model_type={model_type!r}); registered adapters: {available}"
        )

    def adapters(self) -> tuple[ArchitectureAdapter, ...]:
        with self._lock:
            return tuple(self._adapters)


DEFAULT_ADAPTER_REGISTRY = AdapterRegistry(
    (
        GemmaAdapter(),
        LlamaAdapter(),
        MistralAdapter(),
        Qwen2_5_VLAdapter(),
        InternVLAdapter(),
        Qwen2Adapter(),
        Qwen3Adapter(),
    )
)


def register_adapter(adapter: ArchitectureAdapter, *, prepend: bool = False) -> None:
    DEFAULT_ADAPTER_REGISTRY.register(adapter, prepend=prepend)


def get_adapter(model: nn.Module) -> ArchitectureAdapter:
    return DEFAULT_ADAPTER_REGISTRY.resolve(model)


def registered_adapters() -> tuple[ArchitectureAdapter, ...]:
    return DEFAULT_ADAPTER_REGISTRY.adapters()
