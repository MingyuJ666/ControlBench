"""Run the blind semantic judge with a local language model.

Input JSONL shards are produced by ``judge.prepare``. Each completed shard is
written atomically, so an interrupted run can skip finished shards and resume
the rest. A run manifest records the judge model, exact model revision, prompt,
and decoding environment used for the verdicts.
"""

# Run directly as a script as well as imported: the package root
# is this file's ancestor, not its own directory.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import hashlib
import json
import re
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from kernel.config import JUDGE_MODEL
from kernel.judge import JUDGE_INSTRUCTIONS

# 64 was too small: the judge model writes uid/coherent/answers_question and
# then a free-text note, and a note cut by the cap leaves unclosed JSON. With
# greedy decoding the truncation is deterministic, so affected items can never
# be rejudged into validity — the cap must leave room for the note.
MAX_NEW = 256
BATCH = 8
RUN_SCHEMA_VERSION = 1


def blind_label(uid: str) -> str:
    """An echo token for one item that says nothing about how it was made.

    Our uid is ``task|split|config|item``, so rendering it into the prompt
    told the judge the layer and the strength behind every answer it graded —
    and that grade is what selects the configuration. The judge still echoes
    a token so a verdict can be matched to its item and a malformed response
    rejected; it is now a hash, which serves that purpose and no other.
    """
    return hashlib.sha256(uid.encode("utf-8")).hexdigest()[:12]


def build_prompt(item: dict) -> str:
    """Render one prepared judge item without exposing steering metadata."""
    label = blind_label(item["uid"])
    return (f"{JUDGE_INSTRUCTIONS}\n\n"
            f"---\nid: {label}\n"
            f"QUESTION:\n{item['question']}\n\n"
            f"ANSWER ({item['n_words']} words):\n{item['answer']}\n---\n\n"
            f"Your JSON verdict for id {label}:")


def _json_objects(text: str):
    """Yield complete JSON objects embedded in a model response."""
    decoder = json.JSONDecoder()
    for offset, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def _verdict_fields(row: dict) -> dict | None:
    """The two decisions, if this object carries both as booleans."""
    if type(row.get("coherent")) is not bool:
        return None
    if type(row.get("answers_question")) is not bool:
        return None
    note = row.get("note", "")
    if not isinstance(note, str):
        return None
    return {"coherent": row["coherent"],
            "answers_question": row["answers_question"], "note": note}


def parse(text: str, uid: str) -> dict | None:
    """Return a strict verdict object, or ``None`` for an invalid response.

    The verdict is matched on the blind label the judge was shown; the real
    uid is attached here, where it never passed through the model.

    A response whose label does not match is still accepted when it carries
    exactly one decided verdict, with the mismatch recorded. The label is a
    twelve-character hash and the judge sometimes copies it wrong — one
    character inserted, in the case this allowance was written for — while
    deciding the item correctly. Nothing depends on the label being right:
    one prompt carries one item and responses are paired to items by
    position, so the echo is a check, not the join. Rejecting on it threw
    away a sound decision that greedy decoding would reproduce identically,
    which is to say permanently.
    """
    label = blind_label(uid)
    for row in _json_objects(text):
        if row.get("id") != label:
            continue
        if type(row.get("coherent")) is not bool:
            continue
        if type(row.get("answers_question")) is not bool:
            continue
        note = row.get("note", "")
        if not isinstance(note, str):
            continue
        return {
            "uid": uid,
            "coherent": row["coherent"],
            "answers_question": row["answers_question"],
            "note": note,
            "raw_response": text.strip(),
        }

    # The label-anchored paths go first, including the salvage: a response can
    # hold a complete object the judge copied from the instructions and the
    # real decision truncated after it, and the label is what tells them apart.
    salvaged = _parse_truncated(text, uid)
    if salvaged is not None:
        return salvaged

    decided = [
        fields for fields in (_verdict_fields(row) for row in _json_objects(text))
        if fields is not None
    ]
    if len(decided) == 1:
        return {"uid": uid, **decided[0], "id_mismatch": True,
                "raw_response": text.strip()}
    return None


_JSON_BOOL = {"true": True, "false": False}


def _parse_truncated(text: str, uid: str) -> dict | None:
    """Salvage a verdict whose free-text note hit the generation cap.

    The requested format places both boolean decisions before the note, so a
    response truncated inside the note still contains complete decisions.
    Fields are read only after the exact label so instruction examples
    earlier in the response cannot supply them.
    """
    anchor = re.search(rf'"id"\s*:\s*"{re.escape(blind_label(uid))}"', text)
    if anchor is None:
        return None
    tail = text[anchor.start():]
    decisions = {}
    for field in ("coherent", "answers_question"):
        match = re.search(rf'"{field}"\s*:\s*(true|false)', tail)
        if match is None:
            return None
        decisions[field] = _JSON_BOOL[match.group(1)]
    return {
        "uid": uid,
        "coherent": decisions["coherent"],
        "answers_question": decisions["answers_question"],
        "note": "",
        "raw_response": text.strip(),
    }


def _out_path(in_path: Path) -> Path:
    return in_path.with_name(in_path.name.replace("judge_in_", "judge_out_"))


def _unparsed_path(in_path: Path) -> Path:
    return in_path.with_name(
        in_path.name.replace("judge_in_", "judge_unparsed_")
    )


def _atomic_write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _run_spec(in_paths: list[Path], model_name: str, batch: int) -> dict:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "model": model_name,
        "prompt_sha256": hashlib.sha256(
            JUDGE_INSTRUCTIONS.encode("utf-8")
        ).hexdigest(),
        "max_new_tokens": MAX_NEW,
        "do_sample": False,
        "dtype": "bfloat16",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "batch_size": batch,
        "input_shards": [path.name for path in in_paths],
    }


def _load_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"judge run manifest must be a JSON object: {path}")
    return value


def _mismatches(manifest: dict, expected: dict, fields) -> list[str]:
    return [field for field in fields if manifest.get(field) != expected[field]]


@torch.no_grad()
def run(in_paths: list[Path], model_name: str = JUDGE_MODEL,
        batch: int = BATCH, force: bool = False) -> None:
    """Judge all missing shards, reusing only compatible completed shards."""
    in_paths = sorted(in_paths)
    if not in_paths:
        raise ValueError("no judge_in_*.jsonl files were found")
    if batch <= 0:
        raise ValueError(f"batch must be positive, got {batch}")
    parents = {path.parent for path in in_paths}
    if len(parents) != 1:
        raise ValueError("all judge input shards must be in one directory")

    out_dir = in_paths[0].parent
    manifest_path = out_dir / "judge_run.json"
    out_paths = [_out_path(path) for path in in_paths]
    unparsed_paths = [_unparsed_path(path) for path in in_paths]
    if force:
        for path in [*out_paths, *unparsed_paths, manifest_path]:
            path.unlink(missing_ok=True)

    existing = [path for path in out_paths if path.exists()]
    manifest = _load_manifest(manifest_path)
    expected = _run_spec(in_paths, model_name, batch)
    semantic_fields = (
        "schema_version", "model", "prompt_sha256", "max_new_tokens",
        "do_sample", "dtype", "input_shards",
    )

    if existing:
        if manifest is None:
            raise RuntimeError(
                "judge outputs exist without judge_run.json; use --force to "
                "regenerate them with provenance"
            )
        mismatched = _mismatches(manifest, expected, semantic_fields)
        if mismatched:
            raise RuntimeError(
                "existing judge outputs use a different run specification "
                f"({', '.join(mismatched)}); use --force to rejudge"
            )
        if len(existing) == len(out_paths):
            print(f"judged already: {out_dir}")
            return
        mismatched = _mismatches(manifest, expected, expected)
        if mismatched:
            raise RuntimeError(
                "partial judge outputs cannot mix runtime settings "
                f"({', '.join(mismatched)}); use --force to rejudge"
            )

    model_config = AutoConfig.from_pretrained(model_name)
    revision = getattr(model_config, "_commit_hash", None)
    if existing and manifest.get("model_revision") != revision:
        raise RuntimeError(
            "partial judge outputs use a different model revision; "
            "use --force to rejudge"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, config=model_config, dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    if not existing:
        manifest = {**expected, "model_revision": revision}
        _atomic_write_json(manifest_path, manifest)

    devices = set(model.hf_device_map.values())
    print(f"judge={model_name} sharded over {len(devices)} device(s)")

    for in_path, out_path, unparsed_path in zip(
            in_paths, out_paths, unparsed_paths):
        if out_path.exists():
            print(f"  completed already: {out_path.name}")
            continue

        with in_path.open(encoding="utf-8") as handle:
            items = [json.loads(line) for line in handle]
        verdicts = []
        unparsed = []
        for index in range(0, len(items), batch):
            chunk = items[index:index + batch]
            texts = [tokenizer.apply_chat_template(
                        [{"role": "user", "content": build_prompt(item)}],
                        tokenize=False, add_generation_prompt=True)
                     for item in chunk]
            encoded = tokenizer(
                texts, return_tensors="pt", padding=True
            ).to(model.device)
            generated = model.generate(
                **encoded, max_new_tokens=MAX_NEW, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = generated[:, encoded["input_ids"].shape[1]:]
            for row, item in zip(new_tokens, chunk):
                raw_response = tokenizer.decode(row, skip_special_tokens=True)
                verdict = parse(raw_response, item["uid"])
                if verdict is None:
                    unparsed.append({
                        "uid": item["uid"],
                        "raw_response": raw_response.strip(),
                    })
                else:
                    verdicts.append(verdict)
            print(
                f"\r  {in_path.name}: "
                f"{min(index + batch, len(items))}/{len(items)}",
                end="", flush=True,
            )
        print()
        _atomic_write_jsonl(out_path, verdicts)
        _atomic_write_jsonl(unparsed_path, unparsed)
        print(
            f"  -> {out_path.name}  {len(verdicts)} verdicts, "
            f"{len(unparsed)} unparsed"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path)
    parser.add_argument("--model", default=JUDGE_MODEL)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument(
        "--force", action="store_true",
        help="discard completed shards and regenerate all verdicts",
    )
    args = parser.parse_args()
    try:
        run(
            sorted(args.dir.glob("judge_in_*.jsonl")),
            args.model, args.batch, args.force,
        )
    except (ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
