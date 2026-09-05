"""The unsteered baseline: the model as it comes, at every dose.

Every other method must agree with this one at dose 0. That is not a
convention -- it is what makes a dose curve readable, since the point at dose 0
is then the same measurement for all of them.
"""

from .base import Method


class NoSteering(Method):
    name = "none"
