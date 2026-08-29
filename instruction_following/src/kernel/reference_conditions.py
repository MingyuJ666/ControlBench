"""Generate and store unsteered and prompted reference conditions.

For one evaluation split, the unsteered carrier prompts are generated once
and shared across all tasks.  All task-specific prompted inputs are flattened
into one second generation call.  The resulting per-task files are
self-contained so later calibration, evaluation, and judging can consume a
single stable schema.
"""

# Run directly as a script as well as imported: the package root
# is this file's ancestor, not its own directory.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path
from typing import Any

from kernel.config import REFERENCE_CONDITIONS, TARGET_MODEL, slug
from kernel.splits import EVAL_SPLITS, eval_items
from kernel.tasks import TASKS
from kernel.verifiers import score


CONDITIONS = ("unsteered", "prompted")


def _storage_root(root: Path | None) -> Path:
    return REFERENCE_CONDITIONS if root is None else Path(root)


def _validate_coordinates(split: str, task: str | None = None,
                          condition: str | None = None) -> None:
    if split not in EVAL_SPLITS:
        raise ValueError(
            f"Evaluation split must be one of {EVAL_SPLITS}, got {split!r}"
        )
    if task is not None and task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {tuple(TASKS)}")
    if condition is not None and condition not in CONDITIONS:
        raise ValueError(
            f"Unknown reference condition {condition!r}; "
            f"expected one of {CONDITIONS}"
        )


def condition_dir(model_slug: str, split: str, task: str, condition: str,
                  root: Path | None = None) -> Path:
    """Directory containing one task/condition's reference artifacts."""
    _validate_coordinates(split, task, condition)
    return _storage_root(root) / model_slug / split / task / condition


def outputs_path(model_slug: str, split: str, task: str, condition: str,
                 root: Path | None = None) -> Path:
    return condition_dir(model_slug, split, task, condition, root) / "outputs.jsonl"


def summary_path(model_slug: str, split: str, task: str, condition: str,
                 root: Path | None = None) -> Path:
    return condition_dir(model_slug, split, task, condition, root) / "summary.json"


def run_path(model_slug: str, split: str, root: Path | None = None) -> Path:
    _validate_coordinates(split)
    return _storage_root(root) / model_slug / split / "run.json"


def load_reference_records(model_slug: str, split: str, task: str,
                           condition: str, root: Path | None = None,
                           required: bool = False) -> list[dict] | None:
    """Load records, or return None when this optional reference is absent."""
    path = outputs_path(model_slug, split, task, condition, root)
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def load_reference_summary(model_slug: str, split: str, task: str,
                           condition: str, root: Path | None = None,
                           required: bool = False) -> dict | None:
    """Load a summary, or return None when this optional reference is absent.

    ``run.json`` is written last and is therefore the completion marker, so a
    per-task summary is only read once it exists. A run that died halfway
    leaves complete-looking files for the tasks it reached; reading those
    would put a partial run's numbers in the table under the same name as a
    finished one, and the reader would have no way to tell.
    """
    marker = run_path(model_slug, split, root)
    path = summary_path(model_slug, split, task, condition, root)
    if not marker.exists() or not path.exists():
        if required:
            raise FileNotFoundError(path if marker.exists() else marker)
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def summarize_records(records: list[dict]) -> dict:
    """Aggregate rule and truncation facts over every record.

    Same denominator as every other condition: a reference whose rate was
    taken over a different subset could not sit in the same table.
    """
    n_total = len(records)
    n_rule_pass = sum(row["rule_pass"] is True for row in records)
    n_truncated = sum(bool(row["truncated"]) for row in records)
    return {
        "n_total": n_total,
        "n_rule_scored": n_total,
        "n_rule_pass": n_rule_pass,
        "n_truncated": n_truncated,
        "rule_pass_rate": n_rule_pass / n_total if n_total else None,
        "truncation_rate": n_truncated / n_total if n_total else None,
    }


def validate_alignment(items_by_task: dict[str, list[dict]]) -> None:
    """Require every task view to contain the same carriers in the same order."""
    first_task = next(iter(items_by_task), None)
    if first_task is None:
        raise ValueError("No task items were provided")
    expected = [
        (row["prompt_id"], row["base_prompt"])
        for row in items_by_task[first_task]
    ]
    for task, items in items_by_task.items():
        actual = [(row["prompt_id"], row["base_prompt"]) for row in items]
        if actual != expected:
            raise ValueError(
                f"Evaluation carrier alignment differs between "
                f"{first_task!r} and {task!r}"
            )


def make_record(item: dict, condition: str, generated: dict,
                split: str) -> dict:
    """Combine immutable input identity with generation and rule facts."""
    _validate_coordinates(split, item["task"], condition)
    input_prompt = (
        item["base_prompt"]
        if condition == "unsteered"
        else item["prompted_prompt"]
    )
    rule_facts = score(
        generated["output"], item["verifier"], generated["truncated"]
    )
    return {
        "uid": f"{item['item_id']}_{condition}",
        "item_id": item["item_id"],
        "prompt_id": item["prompt_id"],
        "task": item["task"],
        "split": split,
        "condition": condition,
        "question": item["base_prompt"],
        "input_prompt": input_prompt,
        "output": generated["output"],
        "truncated": rule_facts["truncated"],
        "n_new_tokens": generated["n_new_tokens"],
        "verifier": item["verifier"],
        "rule_pass": rule_facts["rule_pass"],
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _expected_paths(model_slug: str, split: str,
                    root: Path | None) -> list[Path]:
    paths = [run_path(model_slug, split, root)]
    for task in TASKS:
        for condition in CONDITIONS:
            paths.extend([
                outputs_path(model_slug, split, task, condition, root),
                summary_path(model_slug, split, task, condition, root),
            ])
    return paths


def generate_reference_conditions(
    generator,
    model_name: str,
    model_slug: str,
    split: str,
    *,
    limit: int | None = None,
    force: bool = False,
    root: Path | None = None,
    progress: bool = True,
) -> dict:
    """Generate all task references with exactly two generation calls.

    ``run.json`` is written last and therefore acts as the completion marker.
    Existing artifacts are rejected before model inference unless ``force`` is
    explicit.
    """
    _validate_coordinates(split)
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be positive, got {limit}")
    if generator.cfg.model != model_name:
        raise ValueError(
            f"Generator model {generator.cfg.model!r} does not match "
            f"recorded model {model_name!r}"
        )

    existing = [
        path for path in _expected_paths(model_slug, split, root)
        if path.exists()
    ]
    if existing and not force:
        raise FileExistsError(
            f"Reference artifacts already exist (for example {existing[0]}); "
            "pass force=True or --force to replace them"
        )

    items_by_task = {
        task: eval_items(task, split)[:limit]
        for task in TASKS
    }
    validate_alignment(items_by_task)
    first_items = items_by_task[next(iter(TASKS))]
    if not first_items:
        raise ValueError(f"No prompts found for split {split!r}")

    stripped_before = generator.stripped_think
    unsteered_results = generator.generate(
        [item["base_prompt"] for item in first_items], progress=progress
    )
    prompted_inputs = [
        item["prompted_prompt"]
        for task in TASKS
        for item in items_by_task[task]
    ]
    prompted_results = generator.generate(prompted_inputs, progress=progress)

    n_prompts = len(first_items)
    if len(unsteered_results) != n_prompts:
        raise RuntimeError(
            f"Expected {n_prompts} unsteered results, "
            f"got {len(unsteered_results)}"
        )
    expected_prompted = n_prompts * len(TASKS)
    if len(prompted_results) != expected_prompted:
        raise RuntimeError(
            f"Expected {expected_prompted} prompted results, "
            f"got {len(prompted_results)}"
        )

    summaries: dict[str, dict[str, dict]] = {}
    offset = 0
    for task, items in items_by_task.items():
        task_prompted = prompted_results[offset:offset + n_prompts]
        offset += n_prompts
        condition_results = {
            "unsteered": unsteered_results,
            "prompted": task_prompted,
        }
        summaries[task] = {}
        for condition, generated_rows in condition_results.items():
            records = [
                make_record(item, condition, generated, split)
                for item, generated in zip(items, generated_rows, strict=True)
            ]
            summary = summarize_records(records)
            directory = condition_dir(
                model_slug, split, task, condition, root
            )
            _write_jsonl(directory / "outputs.jsonl", records)
            _write_json(directory / "summary.json", summary)
            summaries[task][condition] = summary

    cfg = generator.cfg
    run = {
        "model": model_name,
        "model_slug": model_slug,
        "split": split,
        "max_new_tokens": cfg.max_new_tokens,
        "batch_size": cfg.batch_size,
        "dtype": str(cfg.dtype),
        "do_sample": False,
        "n_prompts": n_prompts,
        "n_unique_generations": n_prompts * (1 + len(TASKS)),
        "generation_calls": 2,
        "tasks": list(TASKS),
        "conditions": list(CONDITIONS),
        "stripped_think_outputs": generator.stripped_think - stripped_before,
        "summaries": summaries,
    }
    _write_json(run_path(model_slug, split, root), run)
    return run


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> None:
    from kernel.generate import MAX_NEW_TOKENS, GenConfig, Generator

    parser = argparse.ArgumentParser(
        description="Generate unsteered and prompted reference conditions"
    )
    parser.add_argument("--split", required=True, choices=EVAL_SPLITS)
    parser.add_argument("--model", default=TARGET_MODEL)
    parser.add_argument("--batch-size", type=_positive_int, default=16)
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument(
        "--max-new-tokens", type=_positive_int, default=MAX_NEW_TOKENS,
        help="development override; a run for publication leaves this alone, "
             "and the value used is recorded so a mismatch is visible"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    generator = Generator(GenConfig(
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    ))
    run = generate_reference_conditions(
        generator,
        args.model,
        slug(args.model),
        args.split,
        limit=args.limit,
        force=args.force,
    )
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
