"""No-op control method: the un-steered model, for the reference accuracy."""

from .base import SteeringMethod


class Baseline(SteeringMethod):
    name = "baseline"

    def steer_config(self):
        return None
