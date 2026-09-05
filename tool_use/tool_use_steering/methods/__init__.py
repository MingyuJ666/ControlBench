"""Steering methods. Importing this package registers every built-in method;
adding one means adding a module here and importing it below.
"""

from .base import REGISTRY, Method
from . import none, prompting            # noqa: F401  single-file methods
from .dim import method as _dim          # noqa: F401  folder methods
from .probe import method as _probe      # noqa: F401
from .caa import method as _caa          # noqa: F401
from .cast import method as _cast        # noqa: F401
from .iti import method as _iti          # noqa: F401
from .sadi import method as _sadi        # noqa: F401
from .sae import method as _sae          # noqa: F401


def load(name: str, config: dict | None = None) -> Method:
    if name not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"unknown method {name!r}; available: {known}")
    return REGISTRY[name](config)


__all__ = ["Method", "REGISTRY", "load"]
