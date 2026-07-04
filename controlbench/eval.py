"""Method-agnostic evaluation harness.

Given a SteerableModel, a dataset (list of {'question','gold'}), and a
SteeringMethod, generate a completion per problem (optionally under the method's
steering intervention) and report exact-match accuracy.
"""

import contextlib
from .data import is_correct, extract_pred_answer


def evaluate(model, dataset, method, max_new_tokens=512, verbose=False):
    cfg = method.steer_config()
    correct, records = 0, []
    for i, ex in enumerate(dataset):
        if cfg is None:
            ctx = contextlib.nullcontext()
        else:
            ctx = model.steer(cfg.layer, cfg.vector, cfg.coeff)
        with ctx:
            completion = model.generate(ex["question"], max_new_tokens=max_new_tokens)
        ok = is_correct(completion, ex["gold"])
        correct += int(ok)
        records.append({
            "question": ex["question"],
            "gold": ex["gold"],
            "pred": extract_pred_answer(completion),
            "correct": ok,
            "completion": completion,
        })
        if verbose:
            print(f"  [{i + 1}/{len(dataset)}] acc={correct/(i+1):.3f}", end="\r")
    if verbose:
        print()
    return {
        "method": method.name,
        "n": len(dataset),
        "accuracy": correct / max(len(dataset), 1),
        "records": records,
    }
