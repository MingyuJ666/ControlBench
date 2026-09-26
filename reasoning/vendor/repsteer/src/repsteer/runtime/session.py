from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Literal

from .compiled_plan import CompiledPlan


class SteeringSession(AbstractContextManager[None]):
    """Exception-safe context manager returned by ``model.steer``."""

    def __init__(self, manager: Any, compiled: CompiledPlan) -> None:
        self.manager = manager
        self.compiled = compiled
        self._entered = False

    def __enter__(self) -> None:
        if self._entered:
            raise RuntimeError("a SteeringSession object cannot be re-entered")
        self.manager.enter(self.compiled)
        self._entered = True
        return None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Literal[False]:
        if not self._entered:
            return False
        self._entered = False
        try:
            self.manager.exit(self.compiled)
        except BaseException:
            # Never hide the exception raised by the steered body.  HookManager
            # removal errors are surfaced when there is no primary exception.
            if exc is None:
                raise
        return False


__all__ = ["SteeringSession"]
