"""Gemma Scope IT: contrastive top-1 feature, steered along its decoder direction.

The dictionary is pretrained; only feature selection uses benchmark training
pairs. Generation uses the portable repsteer SAEFeatureArtifact and needs no
SAE encoder. Gemma Scope's raw residual encoder does NOT subtract b_dec.
"""

import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from repsteer.learners.base import ContrastiveLearner, activation_matrix, artifact_metadata, sample_weights
from repsteer.positions import LastNonPaddingToken
from repsteer.sae import select_supervised_features, validate_feature_id

from .learn import DEFAULT_BATCH
from ..paths import output_root, site_slug

MODEL = "google/gemma-2-9b-it"
MODEL_REVISION = "11c9b309abf73637e4b6f9a3fa1e92e615547819"
REPOSITORY = "google/gemma-scope-9b-it-res"
REVISION = "e86af97a5b6fbbccca28ab654f2fda1b0768f770"
SUPPORTED_DOMAINS = ("instruction_following", "tool_use", "tool_use_multi_turn")
# Official IT checkpoints: 16K features, L0 closest to 100 at each available layer.
CHECKPOINTS = {
    9: (88, "6a5970f9d01fe7d50ce072679d04616b12295c690474d8444e407764d1f4bc78"),
    20: (91, "bbd770b6f8b92a2fe7498e05bd6274c6cfa89ebc08fb972c0e842840737f1a82"),
    31: (76, "03df4fa793f20666797c5ed87c811c5aab722e4b84708de28d0fcaad0e1587aa"),
}


def validate_run(model, revision, blocks):
    if model != MODEL:
        raise ValueError(f"SAE currently supports only {MODEL}")
    if revision != MODEL_REVISION:
        raise ValueError(f"SAE requires --revision {MODEL_REVISION} for the validated Gemma model")
    if not blocks or any(b not in CHECKPOINTS for b in blocks):
        raise ValueError("Gemma Scope IT SAE requires --layers from 9,20,31 (zero-based resid_post)")


def checkpoint(layer):
    if layer not in CHECKPOINTS:
        raise ValueError(f"No pinned Gemma Scope IT checkpoint for layer {layer}; choose 9,20,31")
    l0, digest = CHECKPOINTS[layer]
    return {"repository": REPOSITORY, "revision": REVISION,
            "filename": f"layer_{layer}/width_16k/average_l0_{l0}/params.npz", "sha256": digest}


def protocol_config():
    # Independent of the layer subset, so one-layer artifacts remain reusable.
    return {"version": 1, "repository": REPOSITORY, "revision": REVISION,
            "checkpoints": {str(k): checkpoint(k) for k in CHECKPOINTS},
            "selection": "positive_mean_minus_negative_mean", "top_k": 1,
            "absolute": False, "positions": "last_non_padding_token",
            "intervention": "decoder_direction_add", "strength_unit": "residual_norm_ratio"}


def fetch_checkpoint(layer):
    from huggingface_hub import hf_hub_download
    spec = checkpoint(layer)
    cache = Path(os.environ.get("CONTROLBENCH_SAE_CACHE", output_root("instruction_following") / "cache/sae"))
    path = Path(hf_hub_download(spec['repository'], spec['filename'], revision=spec['revision'], cache_dir=cache))
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(block)
    digest = hasher.hexdigest()
    if digest != spec['sha256']:
        raise ValueError(f"SAE checkpoint checksum mismatch: {path}")
    return path


class GemmaScopeAdapter:
    """Float32 JumpReLU inference matching the official raw NPZ format."""

    def __init__(self, weights):
        # Releases use W_enc/W_dec or w_enc/w_dec; accept both without ambiguity.
        keys = [k.lower() if k.startswith("W_") else k for k in weights]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate SAE weight aliases")
        weights = dict(zip(keys, weights.values()))
        expected = {"w_enc", "w_dec", "b_enc", "b_dec", "threshold"}
        if set(weights) != expected:
            raise ValueError(f"Unexpected Gemma Scope weights: {sorted(set(weights))}")
        tensors = {k: torch.as_tensor(v, dtype=torch.float32, device='cpu') for k, v in weights.items()}
        self.w_enc, self.w_dec = tensors['w_enc'], tensors['w_dec']
        self.b_enc, self.b_dec, self.threshold = tensors['b_enc'], tensors['b_dec'], tensors['threshold']
        if self.w_enc.ndim != 2:
            raise ValueError("SAE w_enc must be a matrix")
        self.input_dim, self.num_features = self.w_enc.shape
        shapes = {'w_dec': (self.num_features, self.input_dim), 'b_enc': (self.num_features,),
                  'b_dec': (self.input_dim,), 'threshold': (self.num_features,)}
        if any(tuple(tensors[k].shape) != shape for k, shape in shapes.items()):
            raise ValueError("Inconsistent Gemma Scope weight dimensions")
        if any(not torch.isfinite(v).all() for v in tensors.values()) or (self.threshold < 0).any():
            raise ValueError("SAE weights must be finite and thresholds nonnegative")

    @classmethod
    def from_checkpoint(cls, layer):
        path = fetch_checkpoint(layer)
        with np.load(path, allow_pickle=False) as archive:
            adapter = cls({k: archive[k] for k in archive.files})
        if (adapter.input_dim, adapter.num_features) != (3584, 16384):
            raise ValueError("Pinned Gemma 9B IT SAE must have shape 3584 x 16384")
        adapter.release = REPOSITORY
        adapter.sae_id = checkpoint(layer)['filename']
        return adapter

    @torch.no_grad()
    def encode(self, x):
        if x.shape[-1] != self.input_dim:
            raise ValueError("SAE input width mismatch")
        pre = x.detach().to(device='cpu', dtype=torch.float32) @ self.w_enc + self.b_enc
        return pre.relu() * (pre > self.threshold)

    @torch.no_grad()
    def decode(self, z):
        if z.shape[-1] != self.num_features:
            raise ValueError("SAE latent width mismatch")
        return z.detach().to(device='cpu', dtype=torch.float32) @ self.w_dec + self.b_dec

    def decoder_direction(self, feature_id):
        return self.w_dec[validate_feature_id(self, feature_id)]


def learn(wrapper, domain, target, sites, log=print, *, batch_size=DEFAULT_BATCH, pairs=None):
    if domain.NAME not in SUPPORTED_DOMAINS:
        raise ValueError(f"SAE is not supported for {domain.NAME}")
    validate_run(wrapper.model_id, getattr(wrapper, 'requested_revision', None), [s.layer for s in sites])
    pairs = domain.pairs(target) if pairs is None else pairs
    artifacts = {}
    for site in sites:
        if site.component != 'resid_post':
            raise ValueError("Gemma Scope IT checkpoints require resid_post capture")
        learner = ContrastiveLearner(site=site, positions=LastNonPaddingToken(), normalize=None,
                                     batch_size=batch_size, capture_dtype='float32',
                                     **domain.rendering().learner_kwargs())
        log(f"SAE layer {site.layer}: capturing {len(pairs)} training pairs")
        positive, negative = learner._capture_pair(wrapper, pairs, store=None)
        pos, neg = activation_matrix(positive), activation_matrix(negative)
        values = torch.cat([pos.cpu(), neg.cpu()])
        labels = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
        weights = torch.cat([sample_weights(positive, pos).cpu(), sample_weights(negative, neg).cpu()])
        log(f"SAE layer {site.layer}: captured {len(pos)} positive / {len(neg)} negative activations")
        adapter = GemmaScopeAdapter.from_checkpoint(site.layer)
        metadata = artifact_metadata(learner=learner, model=wrapper, data=pairs,
                    artifact_type='sae_feature', hidden_size=adapter.input_dim,
                    extra_config={'sae_config': protocol_config(), 'sae_checkpoint': checkpoint(site.layer)})
        artifact, = select_supervised_features(adapter, values, labels, sample_weight=weights,
                    top_k=1, absolute=False, scoring='mean_difference', metadata=metadata.with_updates(method='sae'))
        if artifact.score <= 0 or float(artifact.decoder_direction.norm()) == 0:
            raise ValueError("No positive nonzero SAE feature found in the training pairs")
        if abs(float(artifact.decoder_direction.norm()) - 1.) > 1e-3:
            raise ValueError('SAE decoder direction must have unit norm for residual-norm strengths')
        artifacts[site_slug(site)] = artifact
        log(f"SAE layer {site.layer}: selected feature {artifact.feature_id}, train mean gap={artifact.score:.6g}")
    return artifacts
