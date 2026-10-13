"""Apply a native direction, probe or rank-one PCA artifact with repsteer.

Strength is NormRelative: at every steered token the added vector has length
``ratio`` times that token's own residual norm. The same ratio therefore means
the same relative push at block 8 and block 32, on Qwen3 and on Llama.

Positions: the last prompt token during prefill, then every generated token
during decode. This is the read point the direction was learned at, carried
forward through the answer.
"""

import repsteer as rs
from repsteer.operators import Add
from repsteer.positions import GeneratedTokens, LastPromptToken
from repsteer.schedules import NormRelative


def add_plan(artifact, ratio: float) -> rs.SteeringPlan:
    strength = NormRelative(ratio=float(ratio))
    return rs.SteeringPlan([
        rs.Intervention(artifact=artifact, operator=Add(), positions=LastPromptToken(),
                        strength=strength, phase="prefill"),
        rs.Intervention(artifact=artifact, operator=Add(), positions=GeneratedTokens(),
                        strength=strength, phase="decode"),
    ])
