"""Gemma Scope instruction-model single-feature decoder addition."""

import gc
from pathlib import Path
import numpy as np
import torch
from .data import load, read
from .storage import save_tensor, read_tensor
from .learners import weights

SPEC = {
    "repo": "google/gemma-scope-9b-it-res",
    "revision": "e86af97a5b6fbbccca28ab654f2fda1b0768f770",
    "layers": {
        9: "layer_9/width_16k/average_l0_88/params.npz",
        20: "layer_20/width_16k/average_l0_91/params.npz",
        31: "layer_31/width_16k/average_l0_76/params.npz",
    },
    "format": "gemma_jumprelu",
}


def dictionary_record(root, layer):
    root = Path(root)
    record = read(root / f"layer-{layer}.json")
    for key in ("repo", "revision", "format"):
        if record[key] != SPEC[key]:
            raise ValueError("Unexpected dictionary specification")
    if record["block_index"] != layer or record["filename"] != SPEC["layers"][layer]:
        raise ValueError("Unexpected dictionary layer")
    record["path"] = str((root / record["path"]).resolve())
    if not Path(record["path"]).is_file():
        raise FileNotFoundError(record["path"])
    return record


class Dictionary:
    def __init__(self, record, device):
        self.kind = record["format"]
        path = record["path"]
        if self.kind == "gemma_jumprelu":
            with np.load(path, allow_pickle=False) as state:
                self.enc = torch.from_numpy(state["W_enc"].copy()).float().to(device)
                self.dec = torch.from_numpy(state["W_dec"].copy()).float().to(device)
                self.be = torch.from_numpy(state["b_enc"].copy()).float().to(device)
                self.bd = torch.from_numpy(state["b_dec"].copy()).float().to(device)
                self.threshold = (
                    torch.from_numpy(state["threshold"].copy()).float().to(device)
                )
        else:
            raise ValueError(
                "Only Gemma instruction JumpReLU dictionaries are supported"
            )
        if (
            self.enc.shape[::-1] != self.dec.shape
            or self.enc.shape[0] != len(self.bd)
            or self.enc.shape[1] != len(self.be)
        ):
            raise ValueError("SAE matrix shapes")

    def encode(self, x):
        z = x.float()
        # Gemma Scope encoding does not subtract decoder bias.
        z = z @ self.enc + self.be
        return z.relu() * (z > self.threshold)

    def decode(self, f):
        return f @ self.dec + self.bd

    def reconstruct(self, x):
        return self.decode(self.encode(x))


@torch.no_grad()
def statistics(dictionary, pos, neg, labels, feature=None):
    device = dictionary.enc.device
    w = weights(labels).to(device)
    total = w.sum()
    width = dictionary.enc.shape[1]
    diff = torch.zeros(width, device=device)
    sums = torch.zeros_like(diff)
    squares = torch.zeros_like(diff)
    error = torch.zeros((), device=device)
    second = torch.zeros((), device=device)
    first = torch.zeros(pos.shape[1], device=device)
    for start in range(0, len(pos), 16):
        p = pos[start : start + 16].to(device)
        n = neg[start : start + 16].to(device)
        ww = w[start : start + 16, None]
        pf = dictionary.encode(p)
        nf = dictionary.encode(n)
        diff += ((pf - nf) * ww).sum(0)
        sums += ((pf + nf) * ww).sum(0)
        squares += ((pf.square() + nf.square()) * ww).sum(0)
        error += (
            (
                (dictionary.decode(pf) - p).square()
                + (dictionary.decode(nf) - n).square()
            )
            * ww
        ).sum()
        first += ((p + n) * ww).sum(0)
        second += ((p.square() + n.square()) * ww).sum()
    diff /= total
    mu = sums / (2 * total)
    sd = (squares / (2 * total) - mu.square()).clamp_min(0).sqrt()
    if feature is None:
        feature = int(diff.abs().argmax())
    variance = second - first.square().sum() / (2 * total)
    fvu = float(error / variance) if variance > 0 else None
    return {
        "feature": feature,
        "difference": float(diff[feature]),
        "feature_sd": float(sd[feature]),
        "fvu": fvu,
    }


def fit_sae(a, provenance):
    from .pipeline import read_cache, event

    if a.model != "gemma2_9b" or "sae" not in a.spec["methods"]:
        raise ValueError("SAE requires the Gemma SAE configuration")
    destination = a.out / "artifacts/sae.pt"
    if destination.exists():
        read_tensor(destination, provenance)
        return
    rows, cached = read_cache(a, provenance)
    fit, _ = load(a.root, a.model, "fit")
    val, _ = load(a.root, a.model, "head_validation")
    index = {r["sample_id"]: i for i, r in enumerate(rows)}
    ii = [index[r["sample_id"]] for r in fit]
    vi = [index[r["sample_id"]] for r in val]
    layers = {}
    for layer in a.spec["sae_layers"]:
        checkpoint = a.out / "fit" / f"sae-{layer}.pt"
        if checkpoint.exists():
            layers[layer] = read_tensor(checkpoint, provenance)["result"]
            continue
        record = dictionary_record(a.sae_dir, layer)
        import os

        device = (
            "cuda"
            if torch.cuda.is_available() and os.environ.get("SLURM_JOB_ID")
            else "cpu"
        )
        dictionary = Dictionary(record, device)
        pos = torch.stack([r["positive"]["residual"][layer] for r in cached])
        neg = torch.stack([r["negative"]["residual"][layer] for r in cached])
        fit_stats = statistics(dictionary, pos[ii], neg[ii], [r["label"] for r in fit])
        val_stats = statistics(
            dictionary,
            pos[vi],
            neg[vi],
            [r["label"] for r in val],
            fit_stats["feature"],
        )
        sign = 1 if fit_stats["difference"] > 0 else -1
        if (
            fit_stats["difference"] == 0
            or fit_stats["feature_sd"] <= 0
            or val_stats["difference"] * sign <= 0
        ):
            raise ValueError(
                f"Layer {layer}: invalid feature scale or inconsistent held-out sign"
            )
        vector = (
            sign * fit_stats["feature_sd"] * dictionary.dec[fit_stats["feature"]]
        ).cpu()
        if (
            vector.shape != pos.shape[1:]
            or not torch.isfinite(vector).all()
            or vector.norm() <= 0
        ):
            raise ValueError("Invalid SAE decoder vector")
        result = dict(
            dictionary=record,
            fit=fit_stats,
            validation=val_stats,
            fvu_policy="diagnostic_only",
            vector=vector,
            delta_norm_alpha1=float(vector.norm()),
        )
        save_tensor(checkpoint, dict(provenance=provenance, result=result))
        layers[layer] = result
        event(
            a,
            "sae_fit",
            layer=layer,
            feature=fit_stats["feature"],
            validation_fvu=val_stats["fvu"],
        )
        del dictionary
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    save_tensor(
        destination,
        dict(
            provenance=provenance,
            method="sae",
            layers=layers,
            fit_sample_ids=[r["sample_id"] for r in fit],
            validation_sample_ids=[r["sample_id"] for r in val],
        ),
    )
