"""Candidate execution, completion records and cache validation."""

import hashlib
import json
import shutil
import time
from pathlib import Path

from repsteer.selection import Candidate

from .generate import generate_texts
from .rendering import Rendering
from ..paths import PIPELINE_ID, run_dir
from ..methods.plans import add_plan


def candidate(site: str, strength: float, key: str = "r") -> Candidate:
    """``resid_post_20_r0.4``: site, then the method's strength key and value."""
    layer = int(site.rsplit("_", 1)[1])
    return Candidate(id=f"{site}_{key}{strength:g}",
                     parameters={"site": site, "layer": layer, "strength": float(strength)})


def candidates(sites: list[str], strengths, key: str = "r") -> list[Candidate]:
    return [candidate(s, v, key) for s in sites for v in strengths]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def input_fingerprint(items: list[dict]) -> str:
    return hashlib.sha256(json.dumps(items, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def generations_complete(out: Path, items: list[dict], *, expected: dict | None = None) -> bool:
    """Check row coverage and the request that produced a completion set."""
    try:
        rows = read_jsonl(out / "generations.jsonl")
        metadata = json.loads((out / "meta.json").read_text())
        wanted = [item["item_id"] for item in items]
        actual = [row["item_id"] for row in rows]
        if actual != wanted or len(set(actual)) != len(actual):
            return False
        if any(not isinstance(row["output"], str) or type(row["truncated"]) is not bool
               or type(row["n_new_tokens"]) is not int or row["n_new_tokens"] < 0 for row in rows):
            return False
        for row in rows:
            played = row.get("turns")
            if played is None:
                continue
            if not isinstance(played, list) or not played or any(
                    not isinstance(turn.get("output"), str) or type(turn.get("truncated")) is not bool
                    or type(turn.get("n_new_tokens")) is not int for turn in played):
                return False
        request = {"pipeline": PIPELINE_ID, "n_items": len(items),
                   "input_fingerprint": input_fingerprint(items), **(expected or {})}
        return all(metadata.get(key) == value for key, value in request.items())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def generate_candidate(wrapper, domain, *, model: str, experiment: str, method: str,
                       target: str, cand: Candidate, artifact, items: list[dict],
                       batch_size: int, max_new_tokens: int, force: bool = False,
                       log=print, plan_fn=None, messages_fn=None) -> Path:
    """Generate one candidate's outputs; skip if they already exist.

    A domain with a `next_turn` hook plays a conversation: after each round it
    is asked, per item, for the next prompt, and answers None when that item is
    done. Every round runs under the same plan, so a steered run stays steered
    for the whole conversation. Without the hook this is one round, which is
    what it has always been.
    """
    out = run_dir(domain.NAME, model, experiment, method, target, cand.id)
    next_turn = getattr(domain, "next_turn", None)
    if next_turn is not None and messages_fn is not None:
        raise ValueError(f"{domain.NAME}: next_turn renders its own prompts, so it cannot be "
                         "combined with a messages hook")
    strength = cand.parameters["strength"]
    build = plan_fn or (lambda art, params: add_plan(art, params["strength"]))
    plan = build(artifact, dict(cand.parameters)) if strength != 0 else None
    request = {
        "plan": plan.to_dict() if plan is not None else None,
        "candidate": cand.to_dict(),
        "turns": next_turn is not None,
        "rendering": (domain.rendering() if messages_fn is not None else Rendering.strings()).generation_kwargs(),
        "model": {"id": model, "revision": getattr(wrapper, "revision", None),
                  "requested_revision": getattr(wrapper, "requested_revision", None)},
        "artifact_fingerprint": artifact.fingerprint() if artifact is not None else None,
        "generation": {"max_new_tokens": max_new_tokens, "batch_size": batch_size, "do_sample": False},
    }
    protocol = getattr(domain, "PROTOCOL_VERSION", None)
    if protocol is not None:
        request["protocol_version"] = protocol
    generation_version = getattr(domain, "GENERATION_VERSION", None)
    if generation_version is not None:
        request["generation_version"] = generation_version
    if not force and generations_complete(out, items, expected=request):
        log(f"  {cand.id}: complete, skipped")
        return out
    clean = getattr(domain, "clean", None)
    texts = [domain.render(wrapper.tokenizer, it) for it in items]
    messages = [messages_fn(it) for it in items] if messages_fn is not None else None
    t0 = time.time()
    turns: list[list[dict]] = [[] for _ in items]
    live = list(range(len(items)))
    while live:
        outputs = generate_texts(
            wrapper, [texts[i] for i in live], plan, batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            messages=[messages[i] for i in live] if messages is not None else None,
            rendering=domain.rendering() if messages is not None else None)
        for index, output in zip(live, outputs):
            turns[index].append({"output": clean(output["output"]) if clean else output["output"],
                                 "truncated": output["truncated"],
                                 "n_new_tokens": output["n_new_tokens"]})
        if next_turn is None:
            break
        still = []
        for index in live:
            prompt = next_turn(wrapper.tokenizer, items[index],
                               [turn["output"] for turn in turns[index]])
            if prompt is not None:
                texts[index] = prompt
                still.append(index)
        live = still
    rows = []
    for item, played in zip(items, turns):
        row = {"item_id": item["item_id"], **played[0]}
        if len(played) > 1:
            row["turns"] = played
        # Optional domain-owned evidence (e.g. actual multi-turn tool results).
        record_hook = getattr(domain, "generation_record", None)
        if record_hook is not None:
            evidence = record_hook(item, [turn["output"] for turn in played])
            if set(evidence) & set(row):
                raise ValueError(f"{domain.NAME}: generation_record cannot replace output fields")
            row.update(evidence)
        rows.append(row)
    for old in (out / "scores.jsonl", out / "summary.json", out.parent / "selection.json", out.parent / "summary.json"):
        old.unlink(missing_ok=True)
    judging = out.parent / "judging"
    if judging.exists():
        shutil.rmtree(judging)
    write_jsonl(out / "generations.jsonl", rows)
    write_json(out / "meta.json", {
        "pipeline": PIPELINE_ID,
        **request,
        "input_fingerprint": input_fingerprint(items),
        "artifact": str(getattr(getattr(artifact, "metadata", None), "site", None) or cand.parameters["site"]) if artifact is not None else None,
        "n_items": len(items),
        "seconds": round(time.time() - t0, 1),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    log(f"  {cand.id}: {len(rows)} outputs in {time.time() - t0:.0f}s")
    return out
