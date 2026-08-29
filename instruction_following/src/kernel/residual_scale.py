"""How large the residual stream already is, at every site a method may steer.

A dose is only interpretable against what it is added to. ``alpha`` means a
different physical quantity in each baseline — raw magnitude for CAA, standard
deviations of a projection for the probe, a feature's peak activation for the
SAE — so the numbers cannot be compared to each other. What can be compared is
the ratio the ``dose`` records make possible:

    residual_rms_ratio = effective_delta_norm / residual_rms(layer)

the size of the nudge against the size of the signal already there. The
denominator belongs to the model, not to any baseline, so it is measured once
per model and shared, which also keeps every baseline's ratio on one scale.

    python src/kernel/residual_scale.py                 # measure and store
    python src/kernel/residual_scale.py --prompts 400
"""

# Run directly as a script as well as imported: the package root
# is this file's ancestor, not its own directory.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402
import json  # noqa: E402
from uuid import uuid4  # noqa: E402

import torch  # noqa: E402

from kernel.config import BASELINES, TARGET_MODEL, slug  # noqa: E402

DEFAULT_PROMPTS = 200


def scale_path(model_slug: str) -> Path:
    return BASELINES / model_slug / "residual_scale.json"


def measure(model, tok, prompts: list[str], batch: int = 8) -> list[float]:
    """Root-mean-square of the residual stream at each hidden-state index.

    Read at the final prompt token, the site the directions were learned at,
    with the same capture the producers use — so the ratio compares a dose to
    the stream as that dose's own extraction saw it.
    """
    from baselines.shared.capture import last_token_states

    total = None
    count = 0
    for start in range(0, len(prompts), batch):
        states = last_token_states(model, tok, prompts[start:start + batch])
        squares = states.double().pow(2).mean(dim=-1).sum(dim=0)
        total = squares if total is None else total + squares
        count += states.shape[0]
    if not count:
        raise ValueError("residual scale needs at least one prompt")
    return (total / count).sqrt().tolist()


def load(model_slug: str) -> list[float] | None:
    """The measured scale, or None when it has not been measured yet."""
    path = scale_path(model_slug)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["residual_rms"]


def ratio(dose: dict, scale: list[float] | None) -> float | None:
    """The dimensionless dose, or None when either part is unavailable."""
    if not dose or scale is None:
        return None
    norm = dose.get("effective_delta_norm")
    layer = (dose.get("site") or {}).get("layer")
    if norm is None or layer is None or not 0 <= int(layer) < len(scale):
        return None
    denominator = scale[int(layer)]
    return None if denominator == 0 else float(norm) / denominator


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=TARGET_MODEL)
    parser.add_argument("--prompts", type=int, default=DEFAULT_PROMPTS)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    from kernel.generate import GenConfig, Generator
    from kernel.splits import eval_items
    from kernel.tasks import TASKS

    model_slug = slug(args.model)
    path = scale_path(model_slug)
    if path.exists() and not args.force:
        raise SystemExit(f"{path} already measured; pass --force to replace")

    items = eval_items(next(iter(TASKS)), "dev")[:args.prompts]
    generator = Generator(GenConfig(model=args.model, batch_size=args.batch))
    values = measure(
        generator.model, generator.tok,
        [item["base_prompt"] for item in items], args.batch,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps({
        "model": args.model,
        "n_prompts": len(items),
        "extraction_position": "last_prompt_token",
        "residual_rms": values,
    }, indent=2), encoding="utf-8")
    temporary.replace(path)

    print(f"residual RMS over {len(items)} prompts:")
    for index in (1, len(values) // 4, len(values) // 2, len(values) - 1):
        print(f"  index {index:3d}: {values[index]:.2f}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
