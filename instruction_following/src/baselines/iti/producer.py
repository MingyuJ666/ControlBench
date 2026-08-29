"""Produce an ITI artifact: which attention heads to steer, and along what.

One probe is fitted per head over all 36x32 of them. Heads are ranked by how
well their probe separates the two prompt sets on pairs it was not fitted on,
because a head that only separates the pairs it saw tells us about those
carriers rather than about the instruction.

Ranking is what the probe is for. The direction a selected head is moved
along is the mass-mean shift — the difference between the two conditions'
mean activations — which is what the paper uses for its main results after
finding it beats the probe-weight direction. Both are stored, so the
comparison the paper reports can be repeated here without new capture.

The split is by pair, never within one: both members of a pair share a carrier
question and differ only by the hidden instruction, so putting the positive in
training and the negative in validation would let a probe score well by
recognising the question.

Directions are refitted on all pairs once the ranking is settled. Ranking and
direction estimation answer different questions — which head carries the
signal, and where that head's signal points — and only the first needs held-out
data.
"""

from pathlib import Path
from uuid import uuid4

import torch

from baselines.iti.heads import head_layout
from baselines.shared.capture import has_contrast, task_pairs, validate_pairs
from kernel.generate import render_chat

ARTIFACT_METHOD = "iti_head_probes"
BATCH = 16
# Fixed in advance and recorded, as for the other probe-based baseline.
REGULARIZATION_C = 1.0
SOLVER = "lbfgs"
MAX_ITER = 1000
MAX_SEARCHABLE_HEADS = 128     # the largest head count the search plan offers
SEED = 20260720
VALIDATION_FRACTION = 0.2


def _atomic_torch_save(record: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        torch.save(record, temporary)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def pair_split(n_pairs: int, fraction: float = VALIDATION_FRACTION
               ) -> tuple[list[int], list[int]]:
    """Split pair indices into (train, validation), keeping pairs whole.

    The order of the extraction split is already fixed and shuffled once when
    it was released, so taking a contiguous tail needs no second shuffle and
    gives the same split every run without carrying a seed.
    """
    if n_pairs < 2:
        raise ValueError(f"need at least two pairs; got {n_pairs}")
    n_validation = max(1, int(round(n_pairs * fraction)))
    if n_validation >= n_pairs:
        raise ValueError("validation fraction leaves no pairs to fit on")
    cut = n_pairs - n_validation
    return list(range(cut)), list(range(cut, n_pairs))


def mass_mean_direction(positive: torch.Tensor, negative: torch.Tensor
                        ) -> torch.Tensor:
    """The unit vector from the negatives' mean to the positives' mean."""
    shift = positive.mean(dim=0) - negative.mean(dim=0)
    norm = torch.linalg.vector_norm(shift)
    if not torch.isfinite(norm) or norm == 0:
        raise ValueError("the two conditions have the same mean at this head")
    return shift / norm


def fit_head(positive: torch.Tensor, negative: torch.Tensor,
             seed: int = SEED):
    """Fit one head's probe; return its unit direction and accuracy."""
    from sklearn.linear_model import LogisticRegression

    features = torch.cat([positive, negative]).double().numpy()
    labels = torch.cat([
        torch.ones(positive.shape[0]), torch.zeros(negative.shape[0])
    ]).numpy()
    model = LogisticRegression(
        C=REGULARIZATION_C, solver=SOLVER, max_iter=MAX_ITER, random_state=seed,
    ).fit(features, labels)
    weights = torch.tensor(model.coef_[0], dtype=torch.float32)
    norm = torch.linalg.vector_norm(weights)
    if not torch.isfinite(norm) or norm == 0:
        raise ValueError("probe produced a zero or non-finite direction")
    return weights / norm, model


def score_head(model, positive: torch.Tensor, negative: torch.Tensor) -> float:
    """Accuracy of a fitted probe on activations it was not fitted on."""
    features = torch.cat([positive, negative]).double().numpy()
    labels = torch.cat([
        torch.ones(positive.shape[0]), torch.zeros(negative.shape[0])
    ]).numpy()
    return float(model.score(features, labels))


@torch.no_grad()
def capture_pairs(model, tok, pairs: list[dict], batch: int):
    """Return per-head last-token states for both sides of every pair."""
    from baselines.iti.heads import capture_last_token

    positives, negatives = [], []
    for start in range(0, len(pairs), batch):
        chunk = pairs[start:start + batch]
        for field, sink in (("positive", positives), ("negative", negatives)):
            rendered = [render_chat(tok, pair[field]) for pair in chunk]
            encoded = tok(
                rendered, return_tensors="pt", padding=True, padding_side="left",
            ).to(model.device)
            sink.append(capture_last_token(model, encoded))
        print(f"\r    {min(start + batch, len(pairs))}/{len(pairs)} pairs",
              end="", flush=True)
    print()
    return torch.cat(positives), torch.cat(negatives)


def produce_task(model, tok, model_slug: str, task: str, batch: int = BATCH,
                 force: bool = False) -> Path:
    """Rank every head, then save directions and scales for all of them."""
    from baselines.iti import artifact_path

    if batch < 1:
        raise ValueError(f"batch must be positive; got {batch}")
    destination = artifact_path(model_slug, task)
    if destination.exists() and not force:
        raise FileExistsError(
            f"ITI artifact already exists: {destination}; pass force=True "
            "to replace it"
        )

    pairs = task_pairs(task)
    validate_pairs(pairs, task)
    n_layers, n_heads, head_dim = head_layout(model)
    print(f"  {task}: {len(pairs)} pairs, {n_layers}x{n_heads} heads")

    positive, negative = capture_pairs(model, tok, pairs, batch)
    train, validation = pair_split(len(pairs))

    directions = torch.zeros(n_layers, n_heads, head_dim)
    probe_directions = torch.zeros(n_layers, n_heads, head_dim)
    sigmas = torch.zeros(n_layers, n_heads)
    validation_accuracy = torch.zeros(n_layers, n_heads)

    # Not every one of the 1152 heads can be fitted. Some carry identical
    # activations for both members of a pair; on others the probe converges to
    # a zero coefficient vector, which is the same statement — nothing here
    # separates the instruction. Either way the head is left at zero and
    # recorded, rather than aborting a production that has already spent the
    # whole capture pass. A residual-stream producer stops instead, and should:
    # it has 37 sites, not 1152, so one failure there is not a detail.
    unusable = []

    for layer in range(n_layers):
        for head in range(n_heads):
            head_positive = positive[:, layer, head, :]
            head_negative = negative[:, layer, head, :]
            if not has_contrast(head_positive, head_negative):
                unusable.append([layer, head, "identical activations"])
                continue
            try:
                _, ranking_model = fit_head(
                    head_positive[train], head_negative[train]
                )
                probe_direction, _ = fit_head(head_positive, head_negative)
            except ValueError as error:
                unusable.append([layer, head, str(error)])
                continue
            validation_accuracy[layer, head] = score_head(
                ranking_model, head_positive[validation],
                head_negative[validation],
            )
            probe_directions[layer, head] = probe_direction
            direction = mass_mean_direction(head_positive, head_negative)
            directions[layer, head] = direction
            projections = torch.cat([head_positive, head_negative]) @ direction
            sigmas[layer, head] = projections.std(unbiased=True)
        print(f"\r    fitting layer {layer}/{n_layers - 1}", end="", flush=True)
    print()

    n_usable = n_layers * n_heads - len(unusable)
    if unusable:
        layers_hit = sorted({row[0] for row in unusable})
        print(f"    {len(unusable)} head(s) carried no direction, "
              f"in layer(s) {layers_hit}")
    if n_usable < MAX_SEARCHABLE_HEADS:
        raise SystemExit(
            f"only {n_usable} attention heads separated the instruction for "
            f"{task}, fewer than the {MAX_SEARCHABLE_HEADS} the search may "
            "select; the iti baseline reports that rather than a score"
        )
    flat = validation_accuracy.reshape(-1)
    order = torch.argsort(flat, descending=True)
    ranking = [
        (int(index) // n_heads, int(index) % n_heads, float(flat[index]))
        for index in order
    ]

    record = {
        "direction": directions,
        "probe_direction": probe_directions,
        "sigma": sigmas,
        "validation_accuracy": validation_accuracy,
        "ranking": ranking,
        "heads_without_direction": unusable,
        "method": ARTIFACT_METHOD,
        "task": task,
        "model": model.config._name_or_path,
        "n_pairs": len(pairs),
        "extraction_position": "last_prompt_token",
        "site": "attention head output before o_proj",
        "producer_config": {
            "C": REGULARIZATION_C, "solver": SOLVER, "max_iter": MAX_ITER,
            "validation_fraction": VALIDATION_FRACTION,
            "pair_grouping": "pairs kept whole; a contiguous tail is held out",
            "direction": "mass-mean shift, the paper's main-result choice",
            "probe_direction": "recorded but unused; the paper's other variant",
            "directions_estimated_on": "all pairs, after ranking",
        },
        "seed": SEED,
    }
    _atomic_torch_save(record, destination)

    best = ranking[:5]
    print("    top heads (layer, head, held-out accuracy):")
    for layer, head, accuracy in best:
        print(f"      L{layer} H{head}  {accuracy:.3f}")
    print(f"    -> {destination}")
    return destination
