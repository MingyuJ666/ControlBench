"""Teacher-forced span capture and public repsteer learners."""

from l2s.matching import split_trace
from l2s.storage import write_json

from .artifacts import load_tensors, save_learned_artifact, save_tensors
from .config import LAYERS, RESIDUAL_METHODS, SAE_LAYERS
from .generation import render_prompt


def capture_layer(model, name, pairs, layer, directory, *, head_results=False):
    import torch

    import repsteer as rs

    pooled_path = directory / f"layer_{layer}.safetensors"
    if pooled_path.exists():
        return load_tensors(pooled_path)
    pooled = {}
    for side, column in [("positive", "short_trace"), ("negative", "long_trace")]:
        values = []
        for index, pair in enumerate(pairs):
            sample_path = (
                directory / f"layer_{layer}" / f"{side}_{index:02d}.safetensors"
            )
            if sample_path.exists():
                tensors = load_tensors(sample_path)
                values.append(tensors["mean"])
                continue
            reasoning, _ = split_trace(pair[column])
            prompt = render_prompt(model.tokenizer, name, pair["problem"])
            full = prompt + pair[column]
            if not reasoning:
                raise RuntimeError("Empty reasoning span in extraction trace")
            encoded = model.tokenizer(
                full, add_special_tokens=False, return_offsets_mapping=True
            )
            start = len(prompt) + pair[column].find(reasoning)
            end = start + len(reasoning)
            indices = [
                i
                for i, (a, b) in enumerate(encoded["offset_mapping"])
                if b > start and a < end and b > a
            ]
            if not indices:
                raise RuntimeError("Could not align reasoning tokens")
            maximum = getattr(model.config, "max_position_embeddings", 0)
            if maximum and len(encoded["input_ids"]) > maximum:
                raise RuntimeError(
                    "Teacher-forced input exceeds context window; do not truncate pairs"
                )
            captured = rs.capture.capture_activations(
                model,
                inputs=[full],
                site=rs.sites.head_result(layer)
                if head_results
                else rs.sites.resid_post(layer),
                positions=rs.positions.TokenIndices(indices, strict=True),
                pooling="identity",
                special_tokens=False,
                apply_chat_template=False,
                batch_size=1,
                seed=0,
            )
            raw = captured.activations.squeeze(0).detach().cpu()
            if raw.ndim != 2 or raw.shape[0] != len(indices):
                raise RuntimeError(
                    "Capture did not return the requested reasoning span"
                )
            tensors = {"mean": raw.float().mean(dim=0)}
            if not head_results and layer in SAE_LAYERS[name]:
                tensors["tokens"] = raw
            save_tensors(sample_path, tensors)
            values.append(tensors["mean"])
            print(f"capture {name} layer={layer} {side} {index + 1}/64", flush=True)
        pooled[side] = torch.stack(values)
    save_tensors(pooled_path, pooled)
    return pooled


class PooledCaptureModel:
    """Public model.capture protocol backed by the same frozen pooled matrices.

    Learners retain their public .fit API and metadata path. No library private
    methods are overridden, and no extra subject-model forwards occur during fit.
    """

    def __init__(self, model, samples, positive, negative):
        self.model = model
        self.lookup = {
            f"{side}:{index}": matrix[index]
            for side, matrix in [("positive", positive), ("negative", negative)]
            for index in range(samples)
        }

    def __getattr__(self, key):
        return getattr(self.model, key)

    def capture(self, request):
        import torch

        import repsteer as rs

        return rs.capture.ActivationBatch(
            torch.stack([self.lookup[key] for key in request.inputs]),
            request=request,
            pooled=True,
        )


def fit_residual(model, pairs, pooled, layer, method):
    import torch

    import repsteer as rs

    proxy = PooledCaptureModel(
        model, len(pairs), pooled["positive"], pooled["negative"]
    )
    data = rs.data.ContrastivePairs.from_records(
        [
            dict(
                id=pair["pair_id"],
                positive=f"positive:{i}",
                negative=f"negative:{i}",
            )
            for i, pair in enumerate(pairs)
        ]
    )
    learner_class = {
        "CAA": rs.learners.CAA,
        "PCA": rs.learners.PCA,
        "LAT": rs.learners.LAT,
        "Linear-Probe": rs.learners.LinearProbe,
    }[method]
    options = dict(
        site=rs.sites.resid_post(layer),
        positions=rs.positions.AllTokens(),
        pooling="mean",
        normalize="l2",
        seed=0,
    )
    if method in ("PCA", "LAT"):
        options["n_components"] = 1
    artifact = learner_class(**options).fit(proxy, data)
    # Add accepts directions; convert the one-component PCA / probe explicitly.
    vector = (
        artifact.basis[0]
        if method == "PCA"
        else artifact.weight
        if method == "Linear-Probe"
        else artifact.direction
    )
    delta = pooled["positive"].mean(0) - pooled["negative"].mean(0)
    if torch.dot(vector, delta) < 0:
        vector = -vector
    norm = vector.norm()
    if not torch.isfinite(vector).all() or norm <= 0:
        raise RuntimeError(f"{method} produced an invalid direction")
    metadata = artifact.metadata.with_updates(
        artifact_type="direction",
        config={
            **dict(artifact.metadata.config),
            "actual_capture": "mean of teacher-forced reasoning tokens",
            "pair_count": len(pairs),
            "positive": "short",
            "negative": "long",
        },
        normalization="l2",
    )
    return rs.DirectionArtifact(metadata=metadata, direction=vector / norm)


def fit_non_iti(root, name, pairs, model):
    """Fit shared residual captures and Gemma SAE without head captures."""
    from .sae import fit_sae

    directory = root / "artifacts" / name
    capture_dir = directory / "capture"
    for layer in sorted(set(LAYERS[name] + SAE_LAYERS[name])):
        pooled = capture_layer(model, name, pairs, layer, capture_dir)
        if layer in LAYERS[name]:
            for method in RESIDUAL_METHODS:
                path = directory / method / f"layer_{layer}"
                if path.exists():
                    import repsteer as rs

                    artifact = rs.load_artifact(path)
                    artifact.bind(model)
                    continue
                artifact = fit_residual(
                    model, pairs, pooled, layer, method
                )
                save_learned_artifact(artifact, path)
        if layer in SAE_LAYERS[name]:
            fit_sae(model, directory, layer, pairs)


def fit_all(root, name, pairs, run_config, model):
    track = run_config["track"]
    directory = root / "artifacts" / name
    if track in ("all", "non-iti"):
        fit_non_iti(root, name, pairs, model)
    if track in ("all", "iti"):
        from .iti import fit_iti

        fit_iti(model, name, pairs, directory)
    write_json(
        directory / "FIT_COMPLETE.json",
        dict(
            track=track,
            complete=True,
            blocked={},
        ),
    )


def fit_and_zero_check(root, name, pairs, run_config, model, items):
    """Reuse one loaded model; completed captures/artifacts survive restarts."""
    from .evaluate import zero_check

    fit_all(root, name, pairs, run_config, model)
    zero_check(root, name, model, run_config, items)
