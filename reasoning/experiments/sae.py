"""Gemma Scope JumpReLU feature learning for the unified suite."""
import re
from pathlib import Path


def validate_weights(path, layer):
    import numpy as np
    match = re.search(r'(?:^|/)layer_(\d+)(?:/|$)', str(Path(path)))
    if match and int(match.group(1)) != layer:
        raise ValueError('SAE checkpoint layer differs from --sae-layer')
    with np.load(path, allow_pickle=False) as arrays:
        required = {'W_enc', 'W_dec', 'b_enc', 'b_dec', 'threshold'}
        if not required <= set(arrays.files):
            raise ValueError('Expected Gemma Scope JumpReLU parameters')
        enc = arrays['W_enc']
        if enc.ndim != 2 or enc.shape[0] != 3584 or enc.shape[1] < 1:
            raise ValueError('Expected Gemma-2-9B SAE hidden dimension 3584')
        hidden, features = enc.shape
        shapes = dict(W_enc=(hidden, features), W_dec=(features, hidden),
                      b_enc=(features,), b_dec=(hidden,), threshold=(features,))
        for name, shape in shapes.items():
            if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
                raise ValueError(f'Invalid SAE parameter: {name}')
        return hidden, features


def fit_sae(backend, data, layer, weights):
    import numpy as np
    import torch
    from dataclasses import replace
    from reasoning.common import digest
    from reasoning.steering import rs
    from repsteer.learners.base import ContrastiveLearner, artifact_metadata

    hidden, features = validate_weights(weights, layer)
    with np.load(weights, allow_pickle=False) as arrays:
        params = {k: torch.from_numpy(arrays[k].copy()).float()
                  for k in ('W_enc', 'W_dec', 'b_enc', 'b_dec', 'threshold')}

    def encode(x):
        pre = x.detach().cpu().float() @ params['W_enc'] + params['b_enc']
        return torch.relu(pre) * (pre > params['threshold'])

    adapter = rs.sae.FunctionalSAEAdapter(hidden, features, encode,
        lambda z: z.cpu().float() @ params['W_dec'] + params['b_dec'],
        lambda i: params['W_dec'][i], site=rs.sites.resid_post(layer))
    learner = ContrastiveLearner(site=adapter.site,
        positions=rs.positions.LastNonPaddingToken(), normalize=None,
        batch_size=1, apply_chat_template=False, special_tokens=False,
        capture_dtype=torch.float32)
    positive, negative = learner._capture_pair(backend.model, data, store=None)
    metadata = artifact_metadata(learner=learner, model=backend.model, data=data,
        artifact_type='sae_feature', hidden_size=hidden,
        extra_config=dict(sae_sha256=digest(weights), normalization='l2',
                          feature_selection='supervised mean_difference')).with_updates(method='sae')
    artifact = rs.sae.select_feature(sae=adapter,
        positive_activations=positive.activations, negative_activations=negative.activations,
        metadata=metadata, criterion='supervised', scoring='mean_difference')
    direction = artifact.decoder_direction
    if artifact.score <= 0 or not torch.isfinite(direction).all() or direction.norm() <= 0:
        raise ValueError('Expected positive SAE feature score and finite nonzero direction')
    artifact = replace(artifact, decoder_direction=direction / direction.norm())
    artifact = backend.tag_extraction_position(artifact)
    artifact.bind(backend.model)
    return artifact
