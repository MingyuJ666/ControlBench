"""Stage A CLI for building a resumable 64-pair extraction dataset."""

import argparse
import fcntl
import json
import os
import random
from pathlib import Path

from .runtime import load_environment
from .storage import read_json, write_json

DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "data" / "l2s"


def load_pairs(directory):
    from .pairs import TARGET_PAIRS

    path = Path(directory) / "extraction.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != TARGET_PAIRS or len({row["math_id"] for row in rows}) != TARGET_PAIRS:
        raise RuntimeError("Extraction file does not contain exactly 64 distinct pairs")
    return rows


def candidate_name(item):
    return item["math_id"].replace("/", "__").replace(" ", "_")


def run(args):
    load_environment()
    from transformers import AutoTokenizer

    from .pairs import TARGET_PAIRS, craft_pair
    from .sources import TOKENIZER, load_sources, resolve_revisions
    from .teacher import Teacher

    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another pair-construction process owns this output directory") from None
        extraction = directory / "extraction.jsonl"
        if extraction.exists():
            load_pairs(directory)
            print("Stage A already complete: 64 pairs.", flush=True)
            return

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
        encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
        queue_path = directory / "source_queue.json"
        if queue_path.exists():
            queue = read_json(queue_path)
        else:
            queue, _ = load_sources(resolve_revisions())
            if len(queue) < TARGET_PAIRS:
                raise RuntimeError("Insufficient eligible source problems")
            write_json(queue_path, queue)
        if args.prepare_only:
            print(f"Prepared {len(queue)} candidates; no teacher calls made.", flush=True)
            return

        accepted, tried = [], 0
        for item in queue:
            key = candidate_name(item)
            result_path = directory / "candidates" / f"{key}.json"
            if result_path.exists():
                result = read_json(result_path)
            else:
                if args.max_candidates is not None and tried >= args.max_candidates:
                    break
                teacher = Teacher(directory / "raw" / key)
                pair, audit = craft_pair(item, teacher, encode)
                result = dict(math_id=item["math_id"], pair=pair, audit=audit)
                write_json(result_path, result)
                tried += 1
            if result["pair"]:
                accepted.append(result["pair"])
            print(f'{item["math_id"]}: {"keep" if result["pair"] else "drop"}; {len(accepted)}/64', flush=True)
            if len(accepted) == TARGET_PAIRS:
                break

        if len(accepted) != TARGET_PAIRS:
            raise RuntimeError(f"Stage A incomplete: {len(accepted)}/64 pairs. Resume this command; no extraction file published.")
        accepted.sort(key=lambda row: row["math_id"])
        random.Random(0).shuffle(accepted)
        for index, pair in enumerate(accepted):
            pair["pair_id"] = f"l2s-{index:06d}"
        temporary = extraction.with_suffix(".jsonl.tmp")
        with temporary.open("w") as handle:
            for pair in accepted:
                handle.write(json.dumps(pair, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(extraction)
        load_pairs(directory)
        print(f"Published 64 pairs in {directory}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepare-only", action="store_true", help="Prepare sources without teacher calls")
    parser.add_argument("--max-candidates", type=int, help="Limit NEW candidates this invocation; official target remains 64")
    args = parser.parse_args()
    if args.max_candidates is not None and args.max_candidates < 1:
        parser.error("--max-candidates must be positive")
    try:
        run(args)
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted; completed teacher responses and candidates are resumable.\n")
    except Exception as error:
        if isinstance(error, RuntimeError):
            parser.exit(1, f"{error}\n")
        parser.exit(1, f"Stage A stopped: {type(error).__name__}. Check connectivity, dependencies, and configured cache access.\n")


if __name__ == "__main__":
    main()
