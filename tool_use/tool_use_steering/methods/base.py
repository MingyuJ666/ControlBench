"""The plug-in seam: what a steering method has to provide.

A method may intervene at three points, and implements only the ones it uses:

    system()        the system prompt the model is given
    hooks(dose)     a context manager active during generation (forward hooks)
    logits(dose)    a transformers LogitsProcessor

`dose` is the intervention strength. Its units are the method's own business --
raw norm units, a logit bias, a scaling factor. The only fixed rule is that
**dose 0 means no intervention**, so a dose-0 run is the unsteered baseline no
matter which method produced it. A negative dose is a real intervention (it
pushes the other way), not a no-op.

Subclassing registers the method: `class MyMethod(Method): name = "mine"` makes
`--method mine` work, with no registry to edit.
"""

from __future__ import annotations

import contextlib
from typing import Any

REGISTRY: dict[str, type["Method"]] = {}


class Method:
    """No intervention at any seam. Subclass and override what you use."""

    #: The name used on the command line. Must be unique. A subclass that
    #: leaves it empty is treated as an intermediate class and not registered.
    name = ""

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.name:
            return
        if cls.name in REGISTRY:
            raise ValueError(
                f"method name {cls.name!r} is already taken by "
                f"{REGISTRY[cls.name].__module__}")
        REGISTRY[cls.name] = cls

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    # ---- the three seams ------------------------------------------------

    def prepare(self, model, tokenizer) -> None:
        """One-time setup once the model is loaded (load vectors, probes, ...)."""

    def system(self) -> str | None:
        """System prompt to prepend, or None to leave the prompt alone."""
        return None

    def hooks(self, model, dose: float):
        """Context manager active during generation at this dose."""
        return contextlib.nullcontext()

    def logits(self, dose: float):
        """A transformers LogitsProcessor, or None."""
        return None

    # ---- optional -------------------------------------------------------

    def search_space(self) -> list[dict[str, Any]] | None:
        """Configurations this method wants evaluated on dev, or None.

        A method declares what it wants searched; it never runs the search
        itself. That is what lets the benchmark state a fair comparison as
        equal budget -- every method is given the same number of dev
        configurations -- rather than as an identical grid, which would only
        be fair if every method's axes meant the same thing.
        """
        return None

    def describe(self) -> dict:
        """Written into every result file, so a run can be traced to its setup."""
        return {"method": self.name, **self.config}
