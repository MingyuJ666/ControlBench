"""Produce an SAE artifact: one feature per admitted layer, and its scale.

For each layer the gate admitted, every extraction pair is encoded and each
feature is scored by how far its activation separates the two conditions.
Picking the best of 65,536 features on the same data that scored them would
find something whatever the data said, so the pairs are split: features are
ranked on one half and the winner's separation is confirmed on the other. The
confirmed score is recorded beside the chosen feature, so a feature that only
looked good on the half that chose it is visible rather than hidden.

The scale stored with each feature is its maximum activation over the
extraction split. Strength then multiplies something the data has actually
shown, rather than an arbitrary unit.
"""

from pathlib import Path
from uuid import uuid4

import torch

from baselines.sae import checkpoint
from baselines.shared.capture import pair_state_batches, task_pairs

ARTIFACT_METHOD = "sae_decoder_feature"
BATCH = 16
SELECTION_FRACTION = 0.5


def _atomic_torch_save(record: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        torch.save(record, temporary)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def pair_split(n_pairs: int, fraction: float = SELECTION_FRACTION
               ) -> tuple[list[int], list[int]]:
    """Split pair indices into (selection, confirmation), pairs kept whole."""
    if n_pairs < 2:
        raise ValueError(f"need at least two pairs; got {n_pairs}")
    cut = int(round(n_pairs * fraction))
    cut = min(max(cut, 1), n_pairs - 1)
    return list(range(cut)), list(range(cut, n_pairs))


def separation(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    """Per-feature mean activation difference, the paper's ranking.

    Dividing by the within-group spread would be the more careful statistic
    and would choose different features; it is not what Qwen-Scope does, and
    this baseline exists to represent that method. The normalized score is
    recorded beside this one as a diagnostic.
    """
    return positive.mean(dim=0) - negative.mean(dim=0)


def standardized_separation(positive: torch.Tensor, negative: torch.Tensor
                            ) -> torch.Tensor:
    """The same difference in units of the pooled within-group spread."""
    spread = torch.sqrt(
        0.5 * (positive.var(dim=0, unbiased=True)
               + negative.var(dim=0, unbiased=True))
    )
    return separation(positive, negative) / spread.clamp_min(1e-6)


def select_feature(positive: torch.Tensor, negative: torch.Tensor,
                   selection: list[int], confirmation: list[int]) -> dict:
    """Rank features on one half of the pairs and judge the winner on the other.

    ``confirmed`` decides, it does not merely describe: a feature that does
    not still separate in the same direction on pairs it was not chosen on is
    the best of 65,536 fitted to half the data, and the layer is left without
    one. The rule is fixed here rather than after seeing which layers survive.
    """
    ranked = separation(positive[selection], negative[selection])
    feature = int(torch.argmax(ranked))
    confirmed = float(separation(
        positive[confirmation], negative[confirmation]
    )[feature])
    return {
        "feature": feature,
        "selection": float(ranked[feature]),
        "confirmation": confirmed,
        "standardized": float(standardized_separation(
            positive[confirmation], negative[confirmation]
        )[feature]),
        "confirmed": confirmed > 0,
    }


@torch.no_grad()
def encode_pairs(sae, positive_states: torch.Tensor,
                 negative_states: torch.Tensor, index: int, batch: int = 64):
    """Feature activations for both conditions at one hidden-state index."""
    def encode(states: torch.Tensor) -> torch.Tensor:
        rows = []
        for start in range(0, states.shape[0], batch):
            chunk = states[start:start + batch, index, :].to(
                sae.encoder_weight.device
            )
            rows.append(sae.encode(chunk).cpu())
        return torch.cat(rows)

    return encode(positive_states), encode(negative_states)


def produce_task(model, tok, model_slug: str, task: str, batch: int = BATCH,
                 force: bool = False) -> Path:
    """Select one feature per admitted layer and save the artifact."""
    from baselines.sae import artifact_path, passing_layers

    if batch < 1:
        raise ValueError(f"batch must be positive; got {batch}")
    destination = artifact_path(model_slug, task)
    if destination.exists() and not force:
        raise FileExistsError(
            f"SAE artifact already exists: {destination}; pass force=True "
            "to replace it"
        )

    layers = passing_layers(model_slug)
    pairs = task_pairs(task)
    print(f"  {task}: {len(pairs)} pairs, admitted layers {layers}")

    positives, negatives = [], []
    for start, positive, negative in pair_state_batches(
            model, tok, pairs, batch):
        positives.append(positive)
        negatives.append(negative)
        print(f"\r    {min(start + batch, len(pairs))}/{len(pairs)} pairs",
              end="", flush=True)
    print()
    positive_states = torch.cat(positives)
    negative_states = torch.cat(negatives)

    n_indices, hidden = positive_states.shape[1], positive_states.shape[2]
    directions = torch.zeros(n_indices, hidden)
    scales = torch.zeros(n_indices)
    features = torch.full((n_indices,), -1, dtype=torch.long)
    selection_score = torch.zeros(n_indices)
    confirmation_score = torch.zeros(n_indices)
    standardized_score = torch.zeros(n_indices)
    rejected: list[int] = []
    identity = {"repo": checkpoint.REPO_ID, "revision": None, "files": {}}
    selection, confirmation = pair_split(len(pairs))

    for index in layers:
        print(f"    layer {index}: loading checkpoint layer "
              f"{checkpoint.checkpoint_layer(index)}", flush=True)
        loaded = checkpoint.load_layer(index)
        on_device = checkpoint.LoadedSae(**{
            field: (value.to(model.device)
                    if isinstance(value, torch.Tensor) else value)
            for field, value in vars(loaded).items()
        })
        identity["revision"] = loaded.revision
        identity["files"][f"layer{loaded.checkpoint_layer}.sae.pt"] = (
            loaded.file_sha256
        )
        del loaded

        positive_features, negative_features = encode_pairs(
            on_device, positive_states, negative_states, index
        )
        chosen = select_feature(
            positive_features, negative_features, selection, confirmation
        )
        feature = chosen["feature"]
        selection_score[index] = chosen["selection"]
        confirmation_score[index] = chosen["confirmation"]
        standardized_score[index] = chosen["standardized"]
        if not chosen["confirmed"]:
            rejected.append(index)
            print(f"      feature {feature}: separation "
                  f"{chosen['selection']:.2f} selecting but "
                  f"{chosen['confirmation']:.2f} confirming — no feature here")
            del on_device
            torch.cuda.empty_cache()
            continue

        activations = torch.cat([
            positive_features[:, feature], negative_features[:, feature]
        ])
        max_act = float(activations.max())
        if max_act <= 0:
            raise ValueError(
                f"the selected feature never activates at index {index}"
            )

        directions[index] = on_device.feature_direction(feature).cpu()
        scales[index] = max_act
        features[index] = feature
        print(f"      feature {feature}: separation {chosen['selection']:.2f} "
              f"selecting, {chosen['confirmation']:.2f} confirming, "
              f"max_act {max_act:.2f}")
        del on_device
        torch.cuda.empty_cache()

    usable = [index for index in layers if index not in rejected]
    if not usable:
        raise SystemExit(
            f"no admitted layer kept a feature on the confirmation half for "
            f"{task}; the sae baseline reports that rather than a score"
        )
    without_direction = [index for index in range(n_indices)
                         if index not in usable]
    record = {
        "direction": directions,
        "max_act": scales,
        "feature": features,
        "selection_separation": selection_score,
        "confirmation_separation": confirmation_score,
        "standardized_separation": standardized_score,
        "layers_without_confirmed_feature": rejected,
        "method": ARTIFACT_METHOD,
        "task": task,
        "model": model.config._name_or_path,
        "n_pairs": len(pairs),
        "extraction_position": "last_prompt_token",
        "admitted_layers": layers,
        "usable_layers": usable,
        "indices_without_direction": without_direction,
        "checkpoint": identity,
        "producer_config": {
            "selection_fraction": SELECTION_FRACTION,
            "pair_grouping": "pairs kept whole; the first half selects",
            "score": "mean activation difference, as the paper ranks",
            "confirmation_rule": "the difference must stay positive on the "
                                 "half that did not choose the feature",
            "scale": "maximum activation of the chosen feature over extraction",
        },
    }
    _atomic_torch_save(record, destination)
    print(f"    -> {destination}")
    return destination
