"""Extraction/Eval GPU acceptance test; never emits benchmark scores."""

import json
import os
import time
import torch
from .data import load, write_new, read
from .runtime import (
    prepare_prefix,
    prompt_ids,
    completion_ids,
    capture,
    head_layout,
    intervene,
    encoded,
    predict,
    predict_batch,
)
from .provenance import model_for_run


def validation_layers(a):
    return (
        a.spec["sae_layers"]
        if "sae" in a.spec["methods"]
        else a.spec["layer_grid"][a.model]
    )


def smoke(a):
    info = read(a.out / "preflight.json")["inputs"]
    native = info["model"]["config"]["max_position_embeddings"]
    limit = info["effective_max_context"]
    model, tok = model_for_run(a)
    extraction, _ = load(a.root, a.model, "extraction")
    evaluation, _ = load(a.root, a.model, "eval")
    torch.set_num_threads(4)
    rows = [next(r for r in extraction if r["label"] == label) for label in [0, 1]]
    longest = max(extraction, key=lambda r: len(prompt_ids(tok, r, a.model)))
    if len(prompt_ids(tok, longest, a.model, True)) + 8 > limit:
        raise ValueError("Context overflow needs approval")
    if longest["sample_id"] not in {r["sample_id"] for r in rows}:
        rows.append(longest)
    result = {
        "kind": "extraction_eval_engineering_smoke",
        "model": a.model,
        "job_id": os.environ["SLURM_JOB_ID"],
        "gpu": torch.cuda.get_device_name(),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "head_layout": head_layout(model),
        "records": [],
        "checks": [],
        "native_max_context": native,
        "effective_max_context": limit,
    }
    for row in rows:
        t = time.monotonic()
        pos, boundary = completion_ids(tok, row, a.model, row["label"])
        neg, _ = completion_ids(tok, row, a.model, 1 - row["label"])
        a_pos = capture(model, pos, validation_layers(a), heads=True)
        a_neg = capture(model, neg, validation_layers(a), heads=True)
        differences = {
            str(layer): float(
                (a_pos["residual"][layer] - a_neg["residual"][layer]).norm()
            )
            for layer in validation_layers(a)
        }
        if not all(v > 0 for v in differences.values()):
            raise ValueError("Contrast does not reach extraction site")
        result["records"].append(
            {
                "sample_id": row["sample_id"],
                "tokens": len(pos),
                "pair_capture_seconds": time.monotonic() - t,
                "difference_norms": differences,
                "boundary": boundary,
            }
        )
    row = rows[0]
    ids = prompt_ids(tok, row, a.model)
    layer = validation_layers(a)[0]

    def logits():
        with torch.no_grad():
            return (
                model(**encoded(model, ids), logits_to_keep=1)
                .logits[0, -1]
                .float()
                .cpu()
            )

    base = logits()
    events = []
    zero = torch.zeros(model.config.hidden_size)
    with intervene(model, {layer: zero}, telemetry=events):
        same = logits()
    torch.testing.assert_close(base, same, atol=0, rtol=0)
    direction = torch.zeros_like(zero)
    direction[0] = 1
    with intervene(model, {layer: direction}, telemetry=events):
        changed = logits()
    if torch.equal(base, changed):
        raise ValueError("Residual intervention did not affect first logits")
    width = head_layout(model)[layer][2]
    head_delta = torch.zeros(width)
    with intervene(model, head_deltas={layer: head_delta}):
        same_head = logits()
    torch.testing.assert_close(base, same_head, atol=0, rtol=0)
    head_delta[0] = 1
    with intervene(model, head_deltas={layer: head_delta}, telemetry=events):
        changed_head = logits()
    if torch.equal(base, changed_head):
        raise ValueError("Head intervention did not affect first logits")
    for prompted in [False, True]:
        prediction = predict(model, tok, evaluation[0], a.model, prompted=prompted)
        result.setdefault("eval_predictions", []).append(
            {"prompted": prompted, **prediction}
        )
    eval_batch = evaluation[:4]
    batch_base = predict_batch(model, tok, eval_batch, a.model)
    batch_zero = predict_batch(model, tok, eval_batch, a.model, residual={layer: zero})
    if [r["generated_token_ids"] for r in batch_base] != [
        r["generated_token_ids"] for r in batch_zero
    ]:
        raise ValueError("Batched zero steering not equivalent")
    result["batch_throughput"] = {
        "n": len(eval_batch),
        "total_seconds": sum(r["elapsed_seconds"] for r in batch_base),
        "prompt_tokens": sum(r["prompt_tokens"] for r in batch_base),
        "zero_dose_equal": True,
    }
    result.update(
        checks=[
            "nonzero lexical-answer contrasts",
            "bitwise zero-residual logit equivalence",
            "bitwise zero-head logit equivalence",
            "prefill residual and head hooks change first logits",
            "longest Extraction capture",
            "native templates both conditions",
        ],
        hook_events=events,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        status="passed",
    )
    write_new(a.out / "smoke.json", result)
    print(json.dumps(result), flush=True)


def prefix_check(a):
    model, tok = model_for_run(a)
    evaluation = load(a.root, a.model, "eval")[0]
    extraction = load(a.root, a.model, "extraction")[0]
    longest = max(extraction, key=lambda r: len(prompt_ids(tok, r, a.model)))
    if (
        len(prompt_ids(tok, longest, a.model, True)) + 8
        > model.config.max_position_embeddings
    ):
        longest = max(evaluation, key=lambda r: len(prompt_ids(tok, r, a.model)))
    batches = [evaluation[:4], [longest]]
    result = {
        "kind": "engineering_prefix_cache_validation",
        "model": a.model,
        "checks": [],
        "gpu": torch.cuda.get_device_name(),
    }
    for bi, batch in enumerate(batches):
        for prompted in [False, True]:
            start = time.monotonic()
            prepared = prepare_prefix(model, tok, batch, a.model, prompted)
            prefix_seconds = time.monotonic() - start
            for kind in ["zero", "residual_low", "residual_high", "head"]:
                layer = (
                    validation_layers(a)[-1]
                    if kind == "residual_high"
                    else validation_layers(a)[0]
                )
                residual = {}
                heads = {}
                if kind == "head":
                    vector = torch.zeros(head_layout(model)[layer][2])
                    vector[0] = 3
                    heads[layer] = vector
                else:
                    vector = torch.zeros(model.config.hidden_size)
                    vector[0] = 0 if kind == "zero" else 3
                    residual[layer] = vector
                full = predict_batch(
                    model,
                    tok,
                    batch,
                    a.model,
                    prompted=prompted,
                    residual=residual,
                    head_deltas=heads,
                )
                cached = predict_batch(
                    model,
                    tok,
                    batch,
                    a.model,
                    prompted=prompted,
                    residual=residual,
                    head_deltas=heads,
                    prepared=prepared,
                )
                same = [r["generated_token_ids"] for r in full] == [
                    r["generated_token_ids"] for r in cached
                ]
                result["checks"].append(
                    {
                        "batch": bi,
                        "sample_ids": [r["sample_id"] for r in batch],
                        "prompted": prompted,
                        "kind": kind,
                        "equal": same,
                        "prefix_seconds": prefix_seconds,
                        "full_seconds": sum(r["elapsed_seconds"] for r in full),
                        "cached_seconds": sum(r["elapsed_seconds"] for r in cached),
                    }
                )
            del prepared
    result["passed"] = all(c["equal"] for c in result["checks"])
    result["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
    write_new(a.out / "prefix-validation.json", result)
    print(json.dumps(result), flush=True)


def validate(a):
    from .provenance import verified_preflight

    verified_preflight(a)
    receipt = a.out / "validation.json"
    if receipt.exists():
        value = read(receipt)
        if not value["passed"] or value["preflight"] != read(a.out / "preflight.json"):
            raise ValueError("Stale acceptance checks")
        return
    if not (a.out / "smoke.json").exists():
        smoke(a)
    if not (a.out / "prefix-validation.json").exists():
        prefix_check(a)
    if (
        read(a.out / "smoke.json")["status"] != "passed"
        or not read(a.out / "prefix-validation.json")["passed"]
    ):
        raise ValueError("Extraction/Eval acceptance failed")
    write_new(
        receipt,
        dict(
            passed=True,
            preflight=read(a.out / "preflight.json"),
            smoke=read(a.out / "smoke.json"),
            prefix=read(a.out / "prefix-validation.json"),
        ),
    )
