"""Llama Scope: OpenMOSS SAEs for Llama-3.1-8B (arXiv:2410.20526).

fnlp/Llama3_1-8B-Base-LXR-8x: JumpReLU, 32,768 features, one folder per
post-block residual layer 0..31. Encoding per the official inference code
(OpenMOSS llamascopium@0e85797 models/sae.py): the residual is scaled
dataset-wise (sqrt(d_model)/avg_norm), pre = x @ W_enc.T + b_enc, the
per-feature decoder column norms are folded in for the threshold
comparison and divided back out of the surviving activations
(sparsity_include_decoder_norm); avg_norm and the scalar release
threshold ship in hyperparams.json. The shipped checkpoints use
encoder.weight [d_sae, d_model] / decoder.weight [d_model, d_sae] naming
(verified against final.safetensors), not the current library's W_E/W_D.
"""

from __future__ import annotations

import json

import torch

REPO = "fnlp/Llama3_1-8B-Base-LXR-8x"


def load(sae_layer: int) -> dict:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    folder = f"Llama3_1-8B-Base-L{sae_layer}R-8x"
    weights = load_file(hf_hub_download(
        REPO, f"{folder}/checkpoints/final.safetensors"))
    hp = json.load(open(hf_hub_download(REPO, f"{folder}/hyperparams.json")))

    w_enc = weights["encoder.weight"].float()        # [d_sae, d_model]
    b_enc = weights["encoder.bias"].float()
    w_dec = weights["decoder.weight"].float()        # [d_model, d_sae]
    scale = (hp["d_model"] ** 0.5) / hp["dataset_average_activation_norm"]["in"]
    dec_norm = w_dec.norm(dim=0)
    threshold = hp["jump_relu_threshold"]

    def encode(residual: torch.Tensor) -> torch.Tensor:
        pre = (residual * scale) @ w_enc.T + b_enc
        # JumpReLU with the decoder norms folded into the comparison only:
        # official code multiplies, thresholds, then divides back out.
        return pre * (pre * dec_norm > threshold)

    return {"repo": REPO, "W_dec": w_dec, "encode": encode}
