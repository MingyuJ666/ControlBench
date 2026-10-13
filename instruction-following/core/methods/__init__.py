"""Native method registry; layers and strengths are supplied by the caller."""

from .registry import METHODS, Method

BASELINES = ("unsteered", "prompted")

__all__ = ["BASELINES", "METHODS", "Method"]
