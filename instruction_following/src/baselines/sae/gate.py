"""The reconstruction gate this method must pass before it may be scored.

The checkpoint was trained on Qwen3-8B-Base and we apply it to the
post-trained chat model. Whether that transfer holds is measured, not
assumed, and the thresholds were written into RESTRUCTURE.md before any
number was produced, so a disappointing result cannot move the line.

Two measurements per candidate layer, over a fixed set of extraction prompts
at every token position:

``loss recovered``  substitute the reconstruction into the forward pass and
    express the added language-model loss as a fraction of the loss that
    zero-ablating the same site would add. 1.0 means the model works exactly
    as well through the reconstruction; 0.0 means the reconstruction is worth
    no more than deleting the site.
``fvu``  the residual's share of the activations' variance, both measured
    against one mean over every token. Taking each batch's own mean instead
    would make the number depend on how the prompts happened to be grouped —
    with the same activations, batches of two and of four can differ by two
    orders of magnitude.

A gate is necessary, not sufficient: it says the SAE describes these
activations, not that any feature encodes the instruction we want to steer.
"""

from dataclasses import dataclass, field

import torch

from baselines.shared.residual_add import check_layer

# Names how the numbers were measured, not just what they are: an earlier
# report used per-batch variance, which is a different quantity under the same
# field name, and nothing in the file showed that.
MEASURE = "global-mean fvu, label-weighted loss recovered"
LOSS_RECOVERED_MIN = 0.80
FVU_MAX = 0.25
N_PROMPTS = 200


@dataclass
class GateResult:
    """One layer's measurements and the verdict they imply."""

    hidden_state_index: int
    loss_recovered: float
    fvu: float
    cosine: float
    n_prompts: int
    n_tokens: int
    thresholds: dict = field(default_factory=lambda: {
        "loss_recovered_min": LOSS_RECOVERED_MIN, "fvu_max": FVU_MAX,
    })

    @property
    def passed(self) -> bool:
        return (
            self.loss_recovered >= LOSS_RECOVERED_MIN and self.fvu <= FVU_MAX
        )

    def as_record(self) -> dict:
        return {
            "hidden_state_index": self.hidden_state_index,
            "loss_recovered": self.loss_recovered,
            "fvu": self.fvu,
            "cosine": self.cosine,
            "n_prompts": self.n_prompts,
            "n_tokens": self.n_tokens,
            "thresholds": dict(self.thresholds),
            "passed": self.passed,
        }


def _replace_hook(transform):
    """Wrap a tensor transform as a forward hook on one decoder block."""
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        replaced = transform(hidden)
        return (
            (replaced, *output[1:]) if isinstance(output, tuple) else replaced
        )
    return hook


@torch.no_grad()
def _mean_loss(model, batches, layer: int, transform=None) -> float:
    """Mean language-model loss, optionally with the site's output replaced.

    Batches are weighted by the number of positions the loss was actually
    averaged over — labels shift by one and padding is ignored — so batches of
    different lengths combine into the same number a single batch would give.
    """
    module = model.model.layers[layer - 1]
    handle = (
        module.register_forward_hook(_replace_hook(transform))
        if transform is not None else None
    )
    try:
        total, count = 0.0, 0
        for encoded in batches:
            labels = encoded["input_ids"].clone()
            labels[encoded["attention_mask"] == 0] = -100
            loss = model(**encoded, labels=labels).loss
            scored = int((labels[:, 1:] != -100).sum())
            total += float(loss) * scored
            count += scored
        return total / count
    finally:
        if handle is not None:
            handle.remove()


@torch.no_grad()
def measure_layer(model, sae, batches, index: int) -> GateResult:
    """Measure one layer's reconstruction quality on the given batches.

    ``batches`` are pre-tokenized and already on the model's device; they are
    iterated several times, so a list rather than a generator is required.
    """
    check_layer(model, index)

    def reconstruct(hidden: torch.Tensor) -> torch.Tensor:
        flat = hidden.reshape(-1, hidden.shape[-1]).float()
        return sae.reconstruct(flat).reshape(hidden.shape).to(hidden.dtype)

    clean = _mean_loss(model, batches, index)
    reconstructed = _mean_loss(model, batches, index, reconstruct)
    ablated = _mean_loss(model, batches, index, lambda hidden: hidden * 0)

    denominator = ablated - clean
    if denominator <= 0:
        raise ValueError(
            f"zero-ablating layer {index} did not increase loss "
            f"({clean:.4f} -> {ablated:.4f}); the reference is meaningless"
        )
    loss_recovered = 1.0 - (reconstructed - clean) / denominator

    # Two passes: the first accumulates the mean over every token, the second
    # measures against it. One pass with per-batch means would answer a
    # different question for every batch size.
    residual_sq, variance_sq, cosines, n_tokens = 0.0, 0.0, 0.0, 0
    captured = []
    module = model.model.layers[index - 1]
    handle = module.register_forward_hook(
        lambda _m, _i, output: captured.append(
            (output[0] if isinstance(output, tuple) else output).detach()
        )
    )

    def activations(encoded):
        captured.clear()
        model(**encoded)
        hidden = captured[0]
        mask = encoded["attention_mask"].reshape(-1).bool()
        return hidden.reshape(-1, hidden.shape[-1]).float()[mask]

    try:
        total = None
        for encoded in batches:
            flat = activations(encoded)
            batch_sum = flat.sum(dim=0)
            total = batch_sum if total is None else total + batch_sum
            n_tokens += int(flat.shape[0])
        overall_mean = total / n_tokens

        for encoded in batches:
            flat = activations(encoded)
            rebuilt = sae.reconstruct(flat)
            residual_sq += float(((flat - rebuilt) ** 2).sum())
            variance_sq += float(((flat - overall_mean) ** 2).sum())
            cosines += float(torch.nn.functional.cosine_similarity(
                flat, rebuilt, dim=-1
            ).sum())
    finally:
        handle.remove()

    return GateResult(
        hidden_state_index=index,
        loss_recovered=float(loss_recovered),
        fvu=residual_sq / variance_sq,
        cosine=cosines / n_tokens,
        n_prompts=sum(int(batch["input_ids"].shape[0]) for batch in batches),
        n_tokens=n_tokens,
    )
