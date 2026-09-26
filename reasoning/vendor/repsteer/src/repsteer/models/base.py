from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Protocol, runtime_checkable

from .outputs import GenerationResult


@runtime_checkable
class SteerableModel(Protocol):
    model_id: str
    revision: str | None

    def resolve_site(self, site: Any) -> Any: ...

    def capture(self, request: Any) -> Any: ...

    def compile(self, plan: Any, **kwargs: Any) -> Any: ...

    def steer(self, plan: Any, **kwargs: Any) -> AbstractContextManager[None]: ...

    def generate(self, *args: Any, **kwargs: Any) -> GenerationResult: ...
