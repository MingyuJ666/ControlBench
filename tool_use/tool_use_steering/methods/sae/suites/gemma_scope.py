"""Gemma Scope: DeepMind's SAEs for Gemma 2 9B (arXiv:2408.05147).

google/gemma-scope-9b-pt-res: JumpReLU, 16,384 features (width_16k), one
folder per post-block residual layer, several sparsity levels per layer.
The sparsity is picked to match the Qwen suite's L0 of 100 as closely as
the release allows (SPARSITY below); the choice is recorded in the
artifact via the repo path. Encoding per their README: pre = x @ W_enc +
b_enc, then JumpReLU with the per-feature threshold vector -- no input
normalization; params.npz stores W_dec with feature ROWS, transposed here
to the suite contract's feature columns.
"""

from __future__ import annotations

import torch

REPO = "google/gemma-scope-9b-pt-res"

#: sae_layer -> the release's average-l0 folder nearest the Qwen suite's 100.
SPARSITY = {13: 132, 20: 68, 27: 118}


def load(sae_layer: int) -> dict:
    from huggingface_hub import hf_hub_download
    import numpy as np
    if sae_layer not in SPARSITY:
        raise KeyError(f"no sparsity pick for layer {sae_layer}; extend "
                       "gemma_scope.SPARSITY from the repo's average_l0_* dirs")
    folder = f"layer_{sae_layer}/width_16k/average_l0_{SPARSITY[sae_layer]}"
    z = np.load(hf_hub_download(REPO, f"{folder}/params.npz"))
    w_enc = torch.from_numpy(z["W_enc"]).float()         # [d_model, d_sae]
    b_enc = torch.from_numpy(z["b_enc"]).float()
    w_dec = torch.from_numpy(z["W_dec"]).float().T.contiguous()  # -> [d_model, d_sae]
    threshold = torch.from_numpy(z["threshold"]).float() # per-feature

    def encode(residual: torch.Tensor) -> torch.Tensor:
        pre = residual @ w_enc + b_enc
        return pre * (pre > threshold)

    return {"repo": f"{REPO}/{folder}", "W_dec": w_dec, "encode": encode}
