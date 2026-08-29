"""Produce a LAT-PCA artifact: the first principal component per layer.

Signs alternate across pairs before the decomposition, following the paper's
construction ``{(-1)^i (A+(i) - A-(i))}``. That is what makes the first
principal component the axis the two conditions separate along: with every
difference pointing the same way the set has a large mean, and removing it —
as the reference implementation does — would delete the separation itself and
leave only the residual variation around it. Alternating first makes the mean
almost zero, so recentring changes little and the component is the separation.

The sign of a principal component is arbitrary and steering is not, so it is
fixed afterwards by asking which way the positives actually lie: the majority
of pairs must project positively, a vote rather than an average because one
extreme pair should not decide the direction.

Two diagnostics are recorded beside each direction: the fraction of the
differences' variance the component explains, and the cosine between the
components fitted on each half of the pairs. A low explained-variance ratio
means the differences have no single dominant axis; a low split-half cosine
means the axis is a property of these carriers rather than of the contrast.
"""

from pathlib import Path
from uuid import uuid4

import torch

from baselines.shared.capture import pair_state_batches, task_pairs

ARTIFACT_METHOD = "lat_pca_first_component"
# Names the recipe, not just the method. An artifact made before the signs
# alternated holds a different direction under the same method name, and the
# loader has no way to see that in the tensor itself.
CONSTRUCTION = "sign-alternated differences, first principal component"
BATCH = 32


def _atomic_torch_save(record: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        torch.save(record, temporary)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def alternate_signs(differences: torch.Tensor) -> torch.Tensor:
    """Flip every other difference, as ``(-1)^i (A+(i) - A-(i))`` does."""
    signs = torch.ones(differences.shape[0], 1, dtype=differences.dtype)
    signs[1::2] = -1.0
    return differences * signs


def first_component(differences: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Return the leading principal axis of the differences, and its share.

    ``differences`` are the positive-minus-negative activations in pair order;
    the alternation and the sign convention are applied here so callers cannot
    apply one and forget the other.
    """
    if differences.ndim != 2 or differences.shape[0] < 2:
        raise ValueError(
            f"need at least two difference rows; got {tuple(differences.shape)}"
        )
    alternated = alternate_signs(differences)
    centred = alternated - alternated.mean(dim=0, keepdim=True)
    total_variance = float((centred ** 2).sum())
    if total_variance == 0:
        raise ValueError("differences have no variance to decompose")

    # economy SVD: right singular vectors are the principal axes
    _, singular_values, components = torch.linalg.svd(
        centred.double(), full_matrices=False
    )
    direction = components[0].float()
    direction = direction / torch.linalg.vector_norm(direction)
    if not torch.isfinite(direction).all():
        raise ValueError("principal component is not finite")

    # Which way do the positives lie? A vote over pairs, on the differences as
    # they were before alternation.
    votes = (differences @ direction) > 0
    if float(votes.float().mean()) < 0.5:
        direction = -direction
    explained = float(singular_values[0] ** 2) / total_variance
    return direction, explained


def produce_task(model, tok, model_slug: str, task: str, batch: int = BATCH,
                 force: bool = False) -> Path:
    """Fit and atomically save one task's all-layer principal directions."""
    from baselines.lat_pca import artifact_path

    if batch < 1:
        raise ValueError(f"batch must be positive; got {batch}")

    destination = artifact_path(model_slug, task)
    if destination.exists() and not force:
        raise FileExistsError(
            f"LAT-PCA artifact already exists: {destination}; pass force=True "
            "to replace it"
        )

    pairs = task_pairs(task)
    print(f"  {task}: {len(pairs)} pairs")

    positives, negatives = [], []
    for start, positive, negative in pair_state_batches(
            model, tok, pairs, batch):
        positives.append(positive)
        negatives.append(negative)
        processed = min(start + batch, len(pairs))
        print(f"\r    {processed}/{len(pairs)} pairs", end="", flush=True)
    print()

    positive_states = torch.cat(positives)
    negative_states = torch.cat(negatives)
    all_differences = positive_states - negative_states
    n_indices, hidden = all_differences.shape[1], all_differences.shape[2]
    half = all_differences.shape[0] // 2

    directions = torch.zeros(n_indices, hidden)
    sigmas = torch.zeros(n_indices)
    explained = []
    split_half_cosine = []
    without_contrast = []

    for index in range(n_indices):
        differences = all_differences[:, index, :]
        if bool((differences == 0).all()):
            # Index 0 is the embedding of a template token both prompts end
            # with: identical rows, no axis to find. Recorded, not steerable.
            without_contrast.append(index)
            explained.append(None)
            split_half_cosine.append(None)
            continue

        direction, ratio = first_component(differences)
        first, _ = first_component(differences[:half])
        second, _ = first_component(differences[half:2 * half])
        cosine = float(torch.nn.functional.cosine_similarity(
            first, second, dim=0
        ))

        states = torch.cat([
            positive_states[:, index, :], negative_states[:, index, :]
        ])
        sigma = float((states @ direction).std(unbiased=True))
        if sigma <= 0 or not torch.isfinite(torch.tensor(sigma)):
            raise ValueError(
                f"projection at hidden-state index {index} has no spread"
            )

        directions[index] = direction
        sigmas[index] = sigma
        explained.append(ratio)
        split_half_cosine.append(cosine)
        print(f"\r    decomposing layer {index}/{n_indices - 1}",
              end="", flush=True)
    print()
    if without_contrast:
        print(f"    no contrast at index/indices {without_contrast}")

    record = {
        "direction": directions,
        "sigma": sigmas,
        "method": ARTIFACT_METHOD,
        "task": task,
        "model": model.config._name_or_path,
        "n_pairs": len(pairs),
        "extraction_position": "last_prompt_token",
        "explained_variance_ratio": explained,
        "split_half_cosine": split_half_cosine,
        "indices_without_contrast": without_contrast,
        "producer_config": {
            "construction": CONSTRUCTION,
            "centred": True,
            "sign_convention": "majority vote over pairs",
            "decomposition": "economy SVD of the sign-alternated differences",
        },
    }
    _atomic_torch_save(record, destination)

    last = n_indices - 1
    probes = sorted({max(1, last // 4), max(1, last // 2),
                     max(1, 3 * last // 4), last})
    print(
        "    explained variance  "
        + "  ".join(f"L{i}:{explained[i]:.3f}" for i in probes
                    if explained[i] is not None)
    )
    print(
        "    split-half cosine   "
        + "  ".join(f"L{i}:{split_half_cosine[i]:.3f}" for i in probes
                    if split_half_cosine[i] is not None)
    )
    print(f"    -> {destination}")
    return destination
