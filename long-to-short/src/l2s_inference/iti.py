"""All-layer ITI with problem-grouped selection, without changing repsteer.

The library's ITI.fit has no grouped-split argument. This benchmark-owned fit
uses its public logistic solver, artifact and runtime recipe. Direction and
projected-standard-deviation formulas match repsteer ITI for equal-weight pairs.
Capture remains layer-by-layer; completed profiles are never refitted by sweeps.
"""

from .artifacts import save_learned_artifact
from .config import ITI_PROTOCOL


def grouped_split(pairs, *, seed=0, validation_fraction=0.2):
    """Return trace indices for [all short, all long], grouped by pair ID."""
    import torch

    ids = [pair["pair_id"] for pair in pairs]
    count = len(ids)
    if count < 2 or len(set(ids)) != count:
        raise RuntimeError("ITI requires at least two distinct pair IDs")
    if not 0 < validation_fraction < 1:
        raise ValueError("Invalid ITI validation fraction")
    # Canonical ordering makes the split independent of input-file row order.
    canonical = sorted(range(count), key=lambda i: ids[i])
    permutation = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
    order = [canonical[i] for i in permutation.tolist()]
    held_out = min(max(1, round(count * validation_fraction)), count - 1)
    validation_pairs, train_pairs = order[:held_out], order[held_out:]
    train = torch.tensor(train_pairs + [i + count for i in train_pairs])
    validation = torch.tensor(validation_pairs + [i + count for i in validation_pairs])
    metadata = dict(
        train_pair_ids=[ids[i] for i in train_pairs],
        validation_pair_ids=[ids[i] for i in validation_pairs],
        train_trace_count=len(train),
        validation_trace_count=len(validation),
    )
    return train, validation, metadata


def learn_profile(model, pairs, capture, *, top_k=48, seed=0):
    """Fit one global profile; capture(layer) supplies pooled head-result rows."""
    import torch
    from repsteer.learners.probe import fit_logistic_probe

    import repsteer as rs

    train, validation, split = grouped_split(pairs, seed=seed)
    config = model.config
    if not hasattr(config, "num_hidden_layers"):
        config = config.text_config
    layers = int(config.num_hidden_layers)
    model.adapter.capabilities.require("head_result")
    head_dim = int(model.resolve_site(rs.sites.head_result(0)).hidden_dim)
    heads = int(config.num_attention_heads)
    count = len(pairs)
    labels = torch.cat((torch.ones(count), torch.zeros(count)))
    candidates, diagnostics = [], []
    for layer in range(layers):
        if model.resolve_site(rs.sites.head_result(layer)).hidden_dim != head_dim:
            raise RuntimeError("ITI requires consistent head dimensions")
        pooled = capture(layer)
        positive, negative = (
            pooled[side].float().cpu() for side in ("positive", "negative")
        )
        if (
            positive.shape != (count, heads * head_dim)
            or negative.shape != positive.shape
        ):
            raise RuntimeError("ITI head capture has an unexpected query-head layout")
        if not torch.isfinite(positive).all() or not torch.isfinite(negative).all():
            raise RuntimeError("Nonfinite ITI capture")
        features = torch.cat((positive, negative)).reshape(2 * count, heads, head_dim)
        for head in range(heads):
            values = features[:, head, :]
            weight, bias = fit_logistic_probe(
                values[train], labels[train], sample_weight=torch.ones(len(train))
            )
            accuracy = float(
                ((values[validation] @ weight + bias >= 0) == labels[validation])
                .float()
                .mean()
            )
            # As in repsteer: the internal holdout selects heads only. All fitting
            # pairs contribute to the mass-mean direction and calibration scale.
            vector = values[:count].mean(0) - values[count:].mean(0)
            norm = vector.norm()
            record = dict(layer=layer, head=head, validation_accuracy=accuracy)
            if not torch.isfinite(norm) or norm <= 0:
                diagnostics.append(
                    dict(record, usable=False, reason="zero/nonfinite direction")
                )
                continue
            direction = vector / norm
            scale = float(torch.std(values @ direction, correction=1))
            if not torch.isfinite(torch.tensor(scale)) or scale <= 0:
                diagnostics.append(
                    dict(record, usable=False, reason="zero/nonfinite scale")
                )
                continue
            diagnostics.append(dict(record, usable=True, projected_std=scale))
            candidates.append((accuracy, layer, head, direction, scale))
        print(
            f"ITI probes layer={layer}: {heads} heads, grouped validation", flush=True
        )
    if not 0 < top_k <= len(candidates):
        raise RuntimeError(f"ITI needs {top_k} usable heads; found {len(candidates)}")
    selected = sorted(candidates, key=lambda item: (-item[0], item[1], item[2]))[:top_k]
    metadata = rs.ArtifactMetadata(
        model_id=model.model_id,
        model_revision=model.revision,
        architecture=model.architecture,
        hidden_size=head_dim,
        site=None,
        method="iti",
        normalization="l2",
        seed=seed,
        dataset_fingerprint=f"l2s-{len(pairs)}-pairs",
        tokenizer={"id": model.model_id, "revision": model.revision},
        config=dict(
            ITI_PROTOCOL,
            top_k=top_k,
            seed=seed,
            pair_count=count,
            num_layers=layers,
            num_attention_heads=heads,
            head_axis="query",
            positive="short",
            negative="long",
            grouped_split=split,
            candidate_scores=diagnostics,
        ),
        provenance={
            "implementation": "l2s_inference.iti.learn_profile",
            "repsteer_modified": False,
        },
    )
    return rs.ITIArtifact(
        metadata,
        torch.stack([entry[3] for entry in selected]),
        tuple(
            rs.ITIHead(rank, layer, head, accuracy, scale)
            for rank, (accuracy, layer, head, _, scale) in enumerate(selected)
        ),
    )


def fit_iti(model, name, pairs, directory):
    """Reuse the complete artifact, or fit once after the 64-pair runtime gate."""
    import repsteer as rs

    from .fit import capture_layer

    if len(pairs) != 64:
        raise RuntimeError("ITI fitting requires exactly 64 frozen pairs")
    path = directory / "ITI" / "global"
    if path.exists():
        artifact = rs.load_artifact(path)
        artifact.bind(model)
        print(f"ITI reusing fitted profile: {path}", flush=True)
        return artifact
    artifact = learn_profile(
        model,
        pairs,
        lambda layer: capture_layer(
            model, name, pairs, layer, directory / "capture" / "iti", head_results=True
        ),
        top_k=ITI_PROTOCOL["top_k"],
        seed=ITI_PROTOCOL["seed"],
    )
    artifact.bind(model)
    save_learned_artifact(artifact, path)
    return artifact
