"""Rank heads by how specific they are to this instruction, then steer them.

The direction added at a head is unchanged from ``iti``: the mass-mean shift
from the bare prompt to the instructed one, scaled by that head's own sigma.
Only the ranking differs.

``iti`` ranks by a probe's accuracy at separating instructed from bare
prompts. Two things go wrong with that here. Accuracy is a proportion, so it
stops discriminating once a head separates the groups at all — 279 of 1152
heads reached 1.000, and the top-K was then decided by index order. And the
question it asks is answered by any head that notices a sentence was added,
which every task's prompt does.

The score here is an effect size between this task's instructed prompts and
the other four tasks': continuous, so it keeps ordering heads that accuracy
has saturated on, and zero for a head that responds to instructions in
general.
"""

from pathlib import Path

import torch

from baselines.iti.producer import (BATCH, VALIDATION_FRACTION,
                                    _atomic_torch_save, capture_pairs,
                                    mass_mean_direction, pair_split)
from baselines.shared.capture import has_contrast, task_pairs
from baselines.shared.discriminative import standardized_gap
from baselines.tasks_order import other_tasks

MAX_SEARCHABLE_HEADS = 128


def head_specificity(target: torch.Tensor, rivals: torch.Tensor) -> float:
    """Effect size between this instruction and the others, along their axis.

    Both groups are projected onto the direction that separates their means,
    and the score is the standardised gap along it. A head that treats every
    instruction alike has means that coincide, so the gap is about zero
    whatever it does against the bare prompt.
    """
    shift = target.mean(dim=0) - rivals.mean(dim=0)
    norm = torch.linalg.vector_norm(shift)
    if not torch.isfinite(norm) or norm == 0:
        return 0.0
    axis = shift / norm
    gap = standardized_gap(
        (target @ axis).unsqueeze(1), (rivals @ axis).unsqueeze(1)
    )
    return float(gap[0])


@torch.no_grad()
def produce_task(model, tok, model_slug: str, task: str, batch: int = BATCH,
                 force: bool = False) -> Path:
    """Rank every head by instruction specificity and save the artifact."""
    from baselines.iti_disc import ARTIFACT_METHOD, artifact_path

    if batch < 1:
        raise ValueError(f"batch must be positive; got {batch}")
    destination = artifact_path(model_slug, task)
    if destination.exists() and not force:
        raise FileExistsError(
            f"artifact already exists: {destination}; pass force=True to "
            "replace it"
        )

    pairs = task_pairs(task)
    rivals = other_tasks(task)
    print(f"  {task}: {len(pairs)} pairs, contrasted against {rivals}")

    positive, negative = capture_pairs(model, tok, pairs, batch)
    rival_states = []
    for name in rivals:
        print(f"    {name}:")
        rival_states.append(
            capture_pairs(model, tok, task_pairs(name), batch)[0]
        )
    rival_pool = torch.cat(rival_states)

    n_layers, n_heads, head_dim = positive.shape[1:]
    train, validation = pair_split(len(pairs))

    directions = torch.zeros(n_layers, n_heads, head_dim)
    sigmas = torch.zeros(n_layers, n_heads)
    specificity = torch.zeros(n_layers, n_heads)
    unusable: list[list] = []

    for layer in range(n_layers):
        for head in range(n_heads):
            head_positive = positive[:, layer, head, :]
            head_negative = negative[:, layer, head, :]
            if not has_contrast(head_positive, head_negative):
                unusable.append([layer, head, "identical activations"])
                continue
            # Scored on held-out pairs only, for the same reason iti holds out
            # a validation split: the best of 1152 heads fitted to all of the
            # data would be chosen partly on noise.
            head_rivals = rival_pool[:, layer, head, :]
            specificity[layer, head] = head_specificity(
                head_positive[validation], head_rivals
            )
            direction = mass_mean_direction(head_positive, head_negative)
            directions[layer, head] = direction
            projections = torch.cat([head_positive, head_negative]) @ direction
            sigmas[layer, head] = projections.std(unbiased=True)
        print(f"\r    scoring layer {layer}/{n_layers - 1}", end="", flush=True)
    print()

    n_usable = n_layers * n_heads - len(unusable)
    if unusable:
        print(f"    {len(unusable)} head(s) carried no direction, in "
              f"layer(s) {sorted({row[0] for row in unusable})}")
    if n_usable < MAX_SEARCHABLE_HEADS:
        raise SystemExit(
            f"only {n_usable} heads separated {task} from the other "
            f"instructions, fewer than the {MAX_SEARCHABLE_HEADS} the search "
            "may select; the iti_disc baseline reports that rather than a score"
        )

    flat = specificity.reshape(-1)
    order = torch.argsort(flat, descending=True)
    ranking = [
        (int(index) // n_heads, int(index) % n_heads, float(flat[index]))
        for index in order
    ]
    ties = int((flat == flat.max()).sum())

    _atomic_torch_save({
        "direction": directions,
        "sigma": sigmas,
        "specificity": specificity,
        "ranking": ranking,
        "heads_without_direction": unusable,
        "method": ARTIFACT_METHOD,
        "task": task,
        "model": model.config._name_or_path,
        "n_pairs": len(pairs),
        "extraction_position": "last_prompt_token",
        "site": "attention head output before o_proj",
        "contrasted_against": rivals,
        "n_tied_at_top": ties,
        "producer_config": {
            "validation_fraction": VALIDATION_FRACTION,
            "pair_grouping": "pairs kept whole; a contiguous tail is held out",
            "ranking": "standardised gap between this task's instructed "
                       "prompts and the other tasks', on held-out pairs",
            "direction": "mass-mean shift from bare to instructed, as in iti",
            "scale": "standard deviation of the head's own projections",
        },
    }, destination)
    print(f"    -> {destination}  (top score held by {ties} head(s))")
    return destination
