"""Produce a probe artifact: one logistic-regression direction per layer.

Both members of a pair share a carrier question and differ only by the hidden
instruction, so a classifier trained with one member in training and the other
in validation could identify the *question* rather than the instruction. All
capture here is streamed into one training set with no internal split, and the
regularization is fixed in advance rather than selected — there is no held-out
set to select on, and dev exists to choose steering configurations, never to
train the direction.

Recorded with the artifact: the unit direction per hidden-state index, the
standard deviation of the activation projection along it (the unit strength is
expressed in), the solver and regularization, the seed, and the identity of
the extraction pairs the probe was fit on.
"""

from pathlib import Path
from uuid import uuid4

import torch

from baselines.shared.capture import has_contrast, pair_state_batches, task_pairs

ARTIFACT_METHOD = "probe_logistic_regression"
BATCH = 32
# Fixed in advance and recorded: with no held-out split there is nothing to
# select these on, and selecting them on dev would train the direction on the
# split that is supposed to only choose steering configurations.
REGULARIZATION_C = 1.0
SOLVER = "lbfgs"
MAX_ITER = 1000
SEED = 20260720


def _atomic_torch_save(record: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        torch.save(record, temporary)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def fit_layer(positive: torch.Tensor, negative: torch.Tensor,
              seed: int = SEED) -> tuple[torch.Tensor, float, float]:
    """Fit one layer's probe; return its unit direction, sigma and accuracy.

    ``sigma`` is the standard deviation of the projection of all activations
    onto the unit direction, which is what turns an abstract direction into a
    strength unit the intervention can use.
    """
    from sklearn.linear_model import LogisticRegression

    features = torch.cat([positive, negative]).double().numpy()
    labels = torch.cat([
        torch.ones(positive.shape[0]), torch.zeros(negative.shape[0])
    ]).numpy()

    model = LogisticRegression(
        C=REGULARIZATION_C, solver=SOLVER, max_iter=MAX_ITER,
        random_state=seed,
    ).fit(features, labels)

    weights = torch.tensor(model.coef_[0], dtype=torch.float32)
    norm = torch.linalg.vector_norm(weights)
    if not torch.isfinite(norm) or norm == 0:
        raise ValueError("probe produced a zero or non-finite direction")
    direction = weights / norm

    projections = torch.cat([positive, negative]).float() @ direction
    sigma = float(projections.std(unbiased=True))
    if not torch.isfinite(torch.tensor(sigma)) or sigma <= 0:
        raise ValueError("probe projection has zero or non-finite spread")
    return direction, sigma, float(model.score(features, labels))


def produce_task(model, tok, model_slug: str, task: str, batch: int = BATCH,
                 force: bool = False) -> Path:
    """Fit and atomically save one task's all-layer probe directions."""
    from baselines.probe import artifact_path

    if batch < 1:
        raise ValueError(f"batch must be positive; got {batch}")

    destination = artifact_path(model_slug, task)
    if destination.exists() and not force:
        raise FileExistsError(
            f"Probe artifact already exists: {destination}; pass force=True "
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
    n_indices = positive_states.shape[1]

    directions = torch.zeros(n_indices, positive_states.shape[2])
    sigmas = torch.zeros(n_indices)
    accuracies = []
    without_contrast = []
    for index in range(n_indices):
        layer_positive = positive_states[:, index, :]
        layer_negative = negative_states[:, index, :]
        if not has_contrast(layer_positive, layer_negative):
            # Left as a zero direction with zero sigma, and recorded: an index
            # that cannot be steered along must not look like one that can.
            without_contrast.append(index)
            accuracies.append(None)
            continue
        try:
            direction, sigma, accuracy = fit_layer(
                layer_positive, layer_negative
            )
        except ValueError as error:
            raise ValueError(
                f"probe fit failed at hidden-state index {index}: {error}"
            ) from error
        directions[index] = direction
        sigmas[index] = sigma
        accuracies.append(accuracy)
        print(f"\r    fitting layer {index}/{n_indices - 1}", end="", flush=True)
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
        "train_accuracy": accuracies,
        "indices_without_contrast": without_contrast,
        "producer_config": {
            "C": REGULARIZATION_C, "solver": SOLVER, "max_iter": MAX_ITER,
            "standardize": False, "internal_validation": None,
            "pair_grouping": "all pairs in one training set, no internal split",
        },
        "seed": SEED,
    }
    _atomic_torch_save(record, destination)

    last = n_indices - 1
    probes = sorted({max(1, last // 4), max(1, last // 2),
                     max(1, 3 * last // 4), last})
    print(
        "    train accuracy  "
        + "  ".join(
            f"L{index}:{accuracies[index]:.3f}" for index in probes
            if accuracies[index] is not None
        )
    )
    print(f"    -> {destination}")
    return destination
