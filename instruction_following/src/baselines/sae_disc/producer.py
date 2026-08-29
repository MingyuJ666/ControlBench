"""Choose, per admitted layer, the feature that carries *this* instruction.

Everything except the ranking is ``baselines.sae.producer``: the same
checkpoint, the same encoder, the same decoder column as the direction, the
same maximum activation as the dose, the same held-out confirmation.

The ranking asks a different question. ``sae`` scores a feature by how much
more it activates when the instruction is present than when it is absent —
and the prompts differ by a whole added sentence, so a feature that merely
notices the prompt changed scores well. This scores a feature by how much
more it activates for this task's instruction than for the other four, which
that feature cannot do.

The other tasks' positives are the same 1014 questions with a different
sentence appended, so no new data is needed — but they do have to be captured,
which makes this producer about three times the forward passes of ``sae``'s.
"""

from pathlib import Path

import torch

from baselines.sae import checkpoint
from baselines.sae.producer import (BATCH, SELECTION_FRACTION,
                                    _atomic_torch_save, encode_pairs,
                                    pair_split)
from baselines.shared.capture import pair_state_batches, task_pairs
from baselines.shared.discriminative import (discriminative_separation,
                                             other_task_mean)
from baselines.tasks_order import other_tasks


def select_feature(target: torch.Tensor, others: list[torch.Tensor],
                   selection: list[int], confirmation: list[int]) -> dict:
    """Rank on one half of the pairs and judge the winner on the other.

    The confirmation half decides, as it does for ``sae``: the best of 65,536
    features fitted to half the data will separate something, and a feature
    whose margin over the other instructions does not survive on pairs it was
    not chosen on has not shown that it carries this one.
    """
    ranked = discriminative_separation(
        target[selection], [other[selection] for other in others]
    )
    feature = int(torch.argmax(ranked))
    confirmed = float(discriminative_separation(
        target[confirmation], [other[confirmation] for other in others]
    )[feature])
    return {
        "feature": feature,
        "selection": float(ranked[feature]),
        "confirmation": confirmed,
        "confirmed": confirmed > 0,
    }


@torch.no_grad()
def produce_task(model, tok, model_slug: str, task: str, batch: int = BATCH,
                 force: bool = False) -> Path:
    """Select one discriminative feature per admitted layer and save it."""
    from baselines.sae import passing_layers
    from baselines.sae_disc import ARTIFACT_METHOD, artifact_path

    if batch < 1:
        raise ValueError(f"batch must be positive; got {batch}")
    destination = artifact_path(model_slug, task)
    if destination.exists() and not force:
        raise FileExistsError(
            f"artifact already exists: {destination}; pass force=True to "
            "replace it"
        )

    layers = passing_layers(model_slug)
    pairs = task_pairs(task)
    rivals = other_tasks(task)
    print(f"  {task}: {len(pairs)} pairs, admitted layers {layers}, "
          f"contrasted against {rivals}")

    def capture_positive(for_task: str) -> torch.Tensor:
        rows = []
        for start, positive, _negative in pair_state_batches(
                model, tok, task_pairs(for_task), batch):
            rows.append(positive)
            print(f"\r    {for_task}: {min(start + batch, len(pairs))}"
                  f"/{len(pairs)} pairs", end="", flush=True)
        print()
        return torch.cat(rows)

    target_states = capture_positive(task)
    rival_states = [capture_positive(name) for name in rivals]

    n_indices, hidden = target_states.shape[1], target_states.shape[2]
    directions = torch.zeros(n_indices, hidden)
    scales = torch.zeros(n_indices)
    features = torch.full((n_indices,), -1, dtype=torch.long)
    selection_score = torch.zeros(n_indices)
    confirmation_score = torch.zeros(n_indices)
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

        target_features, _ = encode_pairs(
            on_device, target_states, target_states, index
        )
        rival_features = [
            encode_pairs(on_device, states, states, index)[0]
            for states in rival_states
        ]
        chosen = select_feature(
            target_features, rival_features, selection, confirmation
        )
        feature = chosen["feature"]
        selection_score[index] = chosen["selection"]
        confirmation_score[index] = chosen["confirmation"]

        if not chosen["confirmed"]:
            rejected.append(index)
            print(f"      feature {feature}: margin {chosen['selection']:.2f} "
                  f"selecting but {chosen['confirmation']:.2f} confirming — "
                  "no feature here")
            del on_device
            torch.cuda.empty_cache()
            continue

        max_act = float(target_features[:, feature].max())
        if max_act <= 0:
            raise ValueError(
                f"the selected feature never activates at index {index}"
            )
        directions[index] = on_device.feature_direction(feature).cpu()
        scales[index] = max_act
        features[index] = feature
        rival_level = float(other_task_mean(rival_features)[feature])
        print(f"      feature {feature}: margin {chosen['selection']:.2f} "
              f"selecting, {chosen['confirmation']:.2f} confirming, "
              f"max_act {max_act:.2f}, other-instruction mean {rival_level:.2f}")
        del on_device
        torch.cuda.empty_cache()

    usable = [index for index in layers if index not in rejected]
    if not usable:
        raise SystemExit(
            f"no admitted layer kept a feature specific to {task}; the "
            "sae_disc baseline reports that rather than a score"
        )
    without_direction = [index for index in range(n_indices)
                         if index not in usable]

    _atomic_torch_save({
        "direction": directions,
        "max_act": scales,
        "feature": features,
        "selection_separation": selection_score,
        "confirmation_separation": confirmation_score,
        "layers_without_confirmed_feature": rejected,
        "method": ARTIFACT_METHOD,
        "task": task,
        "model": model.config._name_or_path,
        "n_pairs": len(pairs),
        "extraction_position": "last_prompt_token",
        "admitted_layers": layers,
        "usable_layers": usable,
        "indices_without_direction": without_direction,
        "contrasted_against": rivals,
        "checkpoint": identity,
        "producer_config": {
            "selection_fraction": SELECTION_FRACTION,
            "pair_grouping": "pairs kept whole; the first half selects",
            "score": "mean activation for this instruction minus the mean "
                     "over the other tasks' instructions",
            "confirmation_rule": "the margin must stay positive on the half "
                                 "that did not choose the feature",
            "scale": "maximum activation of the chosen feature over extraction",
            "direction": "decoder column, as in the sae baseline",
        },
    }, destination)
    print(f"    -> {destination}")
    return destination
