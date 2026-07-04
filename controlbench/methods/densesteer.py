"""DenseSteer (Ouyang et al., ICML 2026).

Steering vector at layer l:
    v_l = (1/N) * sum_i [ h_l(x_pos_i)[-1] - h_l(x_neg_i)[-1] ]
where h_l(x)[-1] is the residual-stream activation of the final token after
layer l, and (x_pos, x_neg) are dense/sparse reasoning traces for the same
problem. At inference the chosen layer's residual gets h + lambda * v_l.

This class holds the full per-layer vector bank [n_layers, d_model] and exposes
a SteerConfig for a chosen (layer, coeff) so the eval harness can sweep both.
"""

import torch
from .base import SteeringMethod, SteerConfig


class DenseSteer(SteeringMethod):
    name = "densesteer"

    def __init__(self, vectors, layer=None, coeff=0.0, normalize=False):
        """vectors: [n_layers, d_model] tensor (the vector bank)."""
        self.vectors = vectors
        self.layer = layer
        self.coeff = coeff
        self.normalize = normalize

    # --- calibration -----------------------------------------------------

    @staticmethod
    def build_vectors(model, pairs, show_progress=True):
        """Compute the per-layer steering-vector bank from contrastive pairs.

        pairs: list of {"question", "pos", "neg"} where pos/neg are trace texts.
        Returns a [n_layers, d_model] float tensor.
        """
        acc = torch.zeros(model.n_layers, model.d_model, dtype=torch.float32)
        n = 0
        for i, p in enumerate(pairs):
            h_pos = model.residual_last_token(model.full_text(p["question"], p["pos"]))
            h_neg = model.residual_last_token(model.full_text(p["question"], p["neg"]))
            acc += (h_pos - h_neg)
            n += 1
            if show_progress:
                print(f"  pair {i + 1}/{len(pairs)}", end="\r")
        if show_progress:
            print()
        return acc / max(n, 1)

    # --- inference-time config ------------------------------------------

    def steer_config(self):
        if self.layer is None or self.coeff == 0.0:
            return None
        v = self.vectors[self.layer].clone()
        if self.normalize:
            v = v / (v.norm() + 1e-8)
        return SteerConfig(layer=self.layer, vector=v, coeff=self.coeff)
