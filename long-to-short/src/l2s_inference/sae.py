"""Gemma-only SAE selection: encode tokens first, then pool feature activations."""

from l2s.storage import write_json

from .artifacts import load_tensors, save_learned_artifact, save_tensors
from .config import SAE_PATHS, SAE_RELEASE, SAE_REPO


def load_gemma_sae(layer, device):
    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

    import repsteer as rs

    sae_id = f"layer_{layer}/width_16k/canonical"
    release = get_pretrained_saes_directory()[SAE_RELEASE]
    if release.repo_id != SAE_REPO or release.saes_map[sae_id] != SAE_PATHS[layer]:
        raise RuntimeError(
            "SAELens canonical checkpoint mapping differs from the frozen protocol"
        )
    adapter = rs.sae.load(
        provider="saelens",
        release=SAE_RELEASE,
        sae_id=sae_id,
        site=rs.sites.resid_post(layer),
        device=device,
        dtype="float32",
    )
    if adapter.num_features != 16384:
        raise RuntimeError("Gemma SAE has an unexpected feature width")
    adapter.sae.eval()
    return adapter, dict(
        repo=SAE_REPO,
        release=SAE_RELEASE,
        sae_id=sae_id,
        path=SAE_PATHS[layer],
        layer=layer,
        width=16384,
        k=None,
        architecture="JumpReLU",
    )


def mean_features(adapter, tokens, device, chunk_size=128):
    import torch

    total = torch.zeros(adapter.num_features, dtype=torch.float32, device=device)
    with torch.inference_mode():
        for chunk in tokens.split(chunk_size):
            # SAE(mean(residual)) is scientifically different from mean(SAE(residual)).
            total += (
                adapter.encode(chunk.to(device=device, dtype=torch.float32))
                .float()
                .sum(0)
            )
    return (total / len(tokens)).cpu()


def fit_sae(model, directory, layer, pairs):
    import torch

    import repsteer as rs

    path = directory / "SAE" / f"layer_{layer}"
    if path.exists():
        rs.load_artifact(path).bind(model)
        return
    adapter, info = load_gemma_sae(layer, str(model.device))
    features = []
    for side in ("positive", "negative"):
        for index in range(len(pairs)):
            sample_path = (
                directory
                / "capture"
                / f"layer_{layer}"
                / f"{side}_{index:02d}.safetensors"
            )
            tokens = load_tensors(sample_path)["tokens"]
            features.append(mean_features(adapter, tokens, model.device))
    encoded = torch.stack(features)
    labels = torch.cat([torch.ones(len(pairs)), torch.zeros(len(pairs))])
    metadata = rs.ArtifactMetadata(
        model_id=model.model_id,
        model_revision=model.revision,
        architecture=model.architecture,
        hidden_size=adapter.input_dim,
        site=rs.sites.resid_post(layer),
        method="sae",
        seed=0,
        config=info,
        dataset_fingerprint=f"l2s-{len(pairs)}-pairs",
        tokenizer={"id": model.model_id, "revision": model.revision},
        provenance={
            "pooling": "per-trace mean feature activation over reasoning tokens",
            "pair_count": len(pairs),
        },
    )
    artifact = rs.sae.select_supervised_features(
        adapter,
        encoded,
        labels,
        top_k=1,
        encoded=True,
        scoring="mean_difference",
        absolute=False,
        metadata=metadata,
    )[0]
    save_learned_artifact(artifact, path)
    save_tensors(
        directory / "capture" / f"sae_features_{layer}.safetensors",
        {"features": encoded, "labels": labels},
    )
    write_json(
        directory / "SAE" / f"layer_{layer}_selection.json",
        dict(
            info,
            feature_id=artifact.feature_id,
            criterion="short minus long mean activation",
        ),
    )
    del adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
