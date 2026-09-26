"""Activation cache interfaces and an in-memory implementation."""

from __future__ import annotations

from collections import OrderedDict
from threading import RLock
from typing import Protocol, runtime_checkable

from .request import ActivationBatch


@runtime_checkable
class ActivationStore(Protocol):
    def get(self, key: str) -> ActivationBatch | None: ...

    def put(self, key: str, value: ActivationBatch) -> None: ...


class MemoryStore:
    """Thread-safe LRU-ish activation cache for reuse across learners."""

    def __init__(
        self, max_entries: int | None = None, *, clone_on_read: bool = False
    ) -> None:
        if max_entries is not None and max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self.clone_on_read = clone_on_read
        self._items: OrderedDict[str, ActivationBatch] = OrderedDict()
        self._lock = RLock()

    def get(self, key: str) -> ActivationBatch | None:
        with self._lock:
            value = self._items.get(key)
            if value is None:
                return None
            self._items.move_to_end(key)
            return value.clone() if self.clone_on_read else value

    def put(self, key: str, value: ActivationBatch) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            if self.max_entries is not None:
                while len(self._items) > self.max_entries:
                    self._items.popitem(last=False)

    def __getitem__(self, key: str) -> ActivationBatch:
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: ActivationBatch) -> None:
        self.put(key, value)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._items

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


InMemoryStore = MemoryStore


__all__ = ["ActivationStore", "InMemoryStore", "MemoryStore"]
