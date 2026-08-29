"""Load one layer of the Qwen-Scope sparse autoencoder.

Two numbering systems meet here. The benchmark indexes hidden states as
``L in 1..n``, where ``L`` is the output of decoder block ``L-1``; the
checkpoint names its files by the 0-based block index, ``layer0..layer35``,
at hook point ``resid_post``. They therefore differ by one, and the mapping is
explicit in one place so that ``--layer 9`` cannot mean one site for CAA and
another for this method.

Files are ~2.15 GB each and there are 36 of them, so only requested layers are
downloaded, and a loaded layer is released as soon as its measurements are
taken.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ID = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100"
HOOK_POINT = "resid_post"


def checkpoint_layer(hidden_state_index: int) -> int:
    """Map our hidden-state index to the checkpoint's 0-based block index."""
    if hidden_state_index < 1:
        raise ValueError(
            "hidden-state index 0 is the embedding output; the checkpoint "
            "starts at the first decoder block"
        )
    return hidden_state_index - 1


def hidden_state_index(checkpoint_layer_index: int) -> int:
    """Inverse of :func:`checkpoint_layer`, for reading the artifact back."""
    return checkpoint_layer_index + 1


@dataclass(frozen=True)
class LoadedSae:
    """One layer's encoder/decoder, with the identity it came from."""

    encoder_weight: torch.Tensor      # [d_sae, d_model]
    encoder_bias: torch.Tensor        # [d_sae]
    decoder_weight: torch.Tensor      # [d_model, d_sae]
    decoder_bias: torch.Tensor        # [d_model]
    k: int
    hidden_state_index: int
    checkpoint_layer: int
    revision: str
    file_sha256: str

    @property
    def d_sae(self) -> int:
        return self.encoder_weight.shape[0]

    @property
    def d_model(self) -> int:
        return self.encoder_weight.shape[1]

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        """Return TopK feature activations for ``[n, d_model]`` inputs.

        Top-k ReLU, as the paper names it and the demo implements it:
        rectify the pre-activations, then keep the largest k. Inference has to
        match training, and the paper describes what was trained. The model
        card's snippet omits the rectification and disagrees with both.

        The decoder bias is not subtracted before encoding, which other SAE
        families do and this checkpoint does not.
        """
        pre = activations @ self.encoder_weight.T + self.encoder_bias
        rectified = torch.relu(pre)
        values, indices = torch.topk(rectified, self.k, dim=-1)
        features = torch.zeros_like(pre)
        return features.scatter(-1, indices, values)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.decoder_weight.T + self.decoder_bias

    def reconstruct(self, activations: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(activations))

    def feature_direction(self, feature: int) -> torch.Tensor:
        """The unit decoder column steering along this feature adds."""
        column = self.decoder_weight[:, feature].float()
        return column / torch.linalg.vector_norm(column)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_config() -> dict:
    from huggingface_hub import hf_hub_download

    return json.loads(
        Path(hf_hub_download(REPO_ID, "config.json")).read_text(
            encoding="utf-8"
        )
    )


def load_layer(index: int, dtype: torch.dtype = torch.float32) -> LoadedSae:
    """Download (if absent) and load the SAE for one hidden-state index."""
    from huggingface_hub import HfApi, hf_hub_download

    config = repo_config()
    if config.get("hook_point") != HOOK_POINT:
        raise ValueError(
            f"checkpoint hook point {config.get('hook_point')!r} is not "
            f"{HOOK_POINT!r}; the injection site would not match"
        )
    layer = checkpoint_layer(index)
    if layer not in config.get("layers", []):
        raise ValueError(
            f"hidden-state index {index} maps to checkpoint layer {layer}, "
            f"which the checkpoint does not provide"
        )

    filename = f"layer{layer}.sae.pt"
    path = Path(hf_hub_download(REPO_ID, filename))
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state) if isinstance(state, dict) else state

    def take(*names: str) -> torch.Tensor:
        for name in names:
            if name in state:
                return state[name].to(dtype)
        raise KeyError(
            f"{filename} has none of {names}; keys are {sorted(state)[:12]}"
        )

    loaded = LoadedSae(
        encoder_weight=take("encoder.weight", "W_enc", "encoder_weight"),
        encoder_bias=take("encoder.bias", "b_enc", "encoder_bias"),
        decoder_weight=take("decoder.weight", "W_dec", "decoder_weight"),
        decoder_bias=take("decoder.bias", "b_dec", "decoder_bias"),
        k=int(config["k"]),
        hidden_state_index=index,
        checkpoint_layer=layer,
        revision=HfApi().repo_info(REPO_ID).sha,
        file_sha256=_sha256(path),
    )
    if loaded.d_model != int(config["d_model"]):
        raise ValueError(
            f"{filename} has d_model {loaded.d_model}, config says "
            f"{config['d_model']}"
        )
    if loaded.d_sae != int(config["d_sae"]):
        raise ValueError(
            f"{filename} has d_sae {loaded.d_sae}, config says "
            f"{config['d_sae']}"
        )
    return loaded
