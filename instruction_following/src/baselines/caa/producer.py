"""Produce a CAA artifact: the mean positive-minus-negative activation.

For each task, every extraction carrier gives a positive prompt containing the
canonical instruction and a negative prompt containing only the carrier. At
every hidden-state index the task direction is the mean difference at the
final rendered input token, immediately before generation begins.

One artifact stores every index so calibration can choose a layer without
rerunning extraction. A split-half cosine is recorded beside the direction:
computed from the two halves of the pairs, it says whether the direction is a
stable property of the contrast or an artifact of these particular carriers.
"""

from pathlib import Path
from uuid import uuid4

import torch

from baselines.shared.capture import pair_state_batches, task_pairs

ARTIFACT_METHOD = "caa_diff_in_means"
BATCH = 32


def _range_sum(differences: torch.Tensor, start: int, low: int, high: int):
    """Sum the rows of this batch that fall inside ``[low, high)`` globally."""
    first = max(low - start, 0)
    last = min(high - start, differences.shape[0])
    if first >= last:
        return None, 0
    window = differences[first:last]
    return window.sum(dim=0), window.shape[0]


def _atomic_torch_save(record: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        torch.save(record, temporary)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def produce_task(model, tok, model_slug: str, task: str, batch: int = BATCH,
                 force: bool = False) -> Path:
    """Extract and atomically save one task's all-layer CAA direction."""
    from baselines.caa import artifact_path

    if batch < 1:
        raise ValueError(f"batch must be positive; got {batch}")

    destination = artifact_path(model_slug, task)
    if destination.exists() and not force:
        raise FileExistsError(
            f"Vector artifact already exists: {destination}; pass force=True "
            "to replace it"
        )

    pairs = task_pairs(task)
    print(f"  {task}: {len(pairs)} pairs")

    total_sum = None
    first_sum = None
    second_sum = None
    first_count = 0
    second_count = 0
    half = len(pairs) // 2

    for start, positive, negative in pair_state_batches(
            model, tok, pairs, batch):
        differences = positive - negative
        if not torch.isfinite(differences).all():
            raise ValueError("Activation differences contain NaN or Inf")

        chunk_sum = differences.sum(dim=0)
        total_sum = chunk_sum if total_sum is None else total_sum + chunk_sum

        contribution, count = _range_sum(differences, start, 0, half)
        if contribution is not None:
            first_sum = (
                contribution if first_sum is None else first_sum + contribution
            )
            first_count += count

        contribution, count = _range_sum(differences, start, half, 2 * half)
        if contribution is not None:
            second_sum = (
                contribution
                if second_sum is None
                else second_sum + contribution
            )
            second_count += count

        processed = min(start + batch, len(pairs))
        print(f"\r    {processed}/{len(pairs)} pairs", end="", flush=True)
    print()

    if first_count != half or second_count != half:
        raise RuntimeError(
            "Internal split-half accounting error: "
            f"{first_count} and {second_count}, expected {half} each"
        )

    vector = total_sum / len(pairs)
    first_mean = first_sum / first_count
    second_mean = second_sum / second_count
    cosine = torch.nn.functional.cosine_similarity(
        first_mean, second_mean, dim=-1
    )
    if not torch.isfinite(vector).all():
        raise ValueError("Extracted vector contains NaN or Inf")
    if not torch.isfinite(cosine).all():
        raise ValueError("Split-half cosine contains NaN or Inf")

    record = {
        "vector": vector,
        "method": ARTIFACT_METHOD,
        "task": task,
        "model": model.config._name_or_path,
        "n_pairs": len(pairs),
        "extraction_position": "last_prompt_token",
        "split_half_cosine": cosine.tolist(),
    }
    _atomic_torch_save(record, destination)

    last_layer = vector.shape[0] - 1
    probes = sorted({
        max(1, last_layer // 4),
        max(1, last_layer // 2),
        max(1, 3 * last_layer // 4),
        last_layer,
    })
    print(
        "    split-half cosine  "
        + "  ".join(f"L{layer}:{cosine[layer]:.3f}" for layer in probes)
    )
    print(f"    -> {destination}")
    return destination


# The producer contract every method's CLI dispatches through; CAA's own name
# for it predates the registry and stays for the tests and docs that use it.
extract_task = produce_task
