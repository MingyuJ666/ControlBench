"""Thin adapters over repsteer; no local implementations of its learners."""
METHODS = ("diffmean", "lat", "probe")


def fit(model, pairs, method, layer):
    import torch
    import repsteer as rs

    if not 0 <= layer < model.raw.config.num_hidden_layers:
        raise ValueError("Layer must be a zero-based transformer block index")
    options = dict(site=rs.sites.resid_post(layer=layer),
                   positions=rs.positions.LastNonPaddingToken(), normalize="l2",
                   batch_size=1, apply_chat_template=False, special_tokens=False,
                   capture_dtype=torch.float32)
    learners = {"diffmean": lambda: rs.learners.DiffMean(**options),
                "lat": lambda: rs.learners.LAT(n_components=1, center=True, **options),
                "probe": lambda: rs.learners.LinearProbe(l2=0.1, max_iter=25, **options)}
    records = []
    for row in pairs:
        prefix = model.prompt(row["problem"], tokenize=False)
        records.append({"id": row["id"], "positive": prefix + row["positive"],
                        "negative": prefix + row["negative"]})
    artifact = learners[method]().fit(
        model.wrapped, rs.data.ContrastivePairs.from_records(records),
        store=rs.capture.MemoryStore(),
    )
    direction = artifact.direction if hasattr(artifact, "direction") else artifact.weight
    if not torch.isfinite(direction).all() or not torch.isclose(
            direction.norm(), direction.new_tensor(1.0), atol=1e-4):
        raise ValueError("Expected a finite unit-norm steering vector")
    return artifact


def intervention(artifact, strength):
    import repsteer as rs

    return rs.Intervention(artifact=artifact, operator=rs.operators.Add(),
                           positions=rs.positions.GeneratedTokens(),
                           strength=rs.schedules.Constant(strength), phase="decode")
