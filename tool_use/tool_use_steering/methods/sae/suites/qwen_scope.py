"""Qwen-Scope: Qwen's official SAEs for Qwen3-8B.

Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100: TopK-100, 65,536 features, one
checkpoint per residual layer 0..35. Encoding per their README:
pre-activations = residual @ W_enc.T + b_enc, keep the top 100.
"""

from __future__ import annotations

import torch

REPO = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100"


def load(sae_layer: int) -> dict:
    from huggingface_hub import hf_hub_download
    sae = torch.load(hf_hub_download(REPO, f"layer{sae_layer}.sae.pt"),
                     map_location="cpu")
    sae = {k: v.float() for k, v in sae.items()}

    def encode(residual: torch.Tensor) -> torch.Tensor:
        pre = residual @ sae["W_enc"].T + sae["b_enc"]
        values, indices = pre.topk(100, dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, indices, values.clamp_min(0))
        return acts

    return {"repo": REPO, "W_dec": sae["W_dec"], "encode": encode}
