"""Fit and persist native repsteer layer artifacts and model-wide profiles."""

from dataclasses import replace
import shutil
import tempfile
from pathlib import Path
from typing import Any

from repsteer.artifacts import SteeringArtifact, load_artifact as load_native, save_artifact as save_native
from repsteer.core.errors import ArtifactFormatError, UnsafeArtifactError
from repsteer.capture import MemoryStore
from repsteer.data import ContrastivePairs
from repsteer.positions import LastNonPaddingToken

from ..paths import PIPELINE_ID, artifact_dir, site_slug
from ..runtime.rendering import Rendering

DEFAULT_BATCH = 32
CAPTURE_DTYPE = "float32"


def learn_artifacts(wrapper, learner_cls, pairs: ContrastivePairs, sites,
                    rendering: Rendering, *, batch_size: int = DEFAULT_BATCH,
                    learner_kwargs: dict[str, Any] | None = None,
                    store: MemoryStore | None = None) -> dict[str, SteeringArtifact]:
    """Call the native learner without changing its fitted tensors or type."""
    out = {}
    for site in sites:
        learner = learner_cls(
            site=site, positions=LastNonPaddingToken(), batch_size=batch_size,
            capture_dtype=CAPTURE_DTYPE, **rendering.learner_kwargs(),
            **(learner_kwargs or {}),
        )
        out[site_slug(site)] = learner.fit(wrapper, pairs, store=store)
    return out


def with_config(artifact, **fields: Any):
    """Attach benchmark provenance through the native metadata interface."""
    return replace(artifact, metadata=artifact.metadata.with_updates(
        config={**artifact.metadata.config, **fields}))


def artifact_key(artifact) -> str:
    if artifact.metadata.site is not None:
        return site_slug(artifact.metadata.site)
    if artifact.metadata.artifact_type == "iti":
        from .iti import profile_key
        return profile_key(len(artifact.heads), artifact.metadata.seed)
    raise ValueError("artifact has neither a layer site nor an ITI profile")


def save_artifact(artifact, *, domain: str, model: str, method: str, target: str,
                  force: bool = False, training: str | None = None) -> Path:
    """Save the native artifact under this pipeline's output directory."""
    destination = artifact_dir(domain, model, method, target, artifact_key(artifact), training=training)
    if destination.exists():
        if not force:
            raise FileExistsError(f"{destination} exists; pass force=True to replace it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as temporary:
        bundle = Path(temporary) / "artifact"
        save_native(with_config(artifact, domain=domain, target=target,
                                method=method, pipeline=PIPELINE_ID), bundle)
        if destination.exists():
            if not force:
                raise FileExistsError(f"{destination} exists; pass force=True to replace it")
            shutil.rmtree(destination)
        bundle.replace(destination)
    return destination


def load_artifact(domain: str, model: str, method: str, target: str, site,
                  *, training: str | None = None):
    """Let repsteer restore the native type and verify artifact checksums."""
    return load_native(artifact_dir(domain, model, method, target, site, training=training))


def artifact_complete(domain: str, model: str, method: str, target: str, site,
                      *, expected_config: dict | None = None, training: str | None = None) -> bool:
    """Check the complete native bundle and its benchmark provenance."""
    slug = site if isinstance(site, str) else site_slug(site)
    try:
        artifact = load_artifact(domain, model, method, target, slug, training=training)
        key = artifact_key(artifact)
    except (OSError, ValueError, ArtifactFormatError, UnsafeArtifactError):
        return False
    metadata = artifact.metadata
    config = metadata.to_dict()["config"]
    return (key == slug
            and metadata.model_id == model
            and all(config.get(key) == value for key, value in {
                "pipeline": PIPELINE_ID, "domain": domain, "method": method, "target": target,
                **(expected_config or {}),
            }.items()))
