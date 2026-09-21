"""Record settings and bind checkpoints to an explicit run identifier."""

import importlib.metadata
import platform
import os
import time
import uuid
from contextlib import contextmanager

from .config import MODELS, PINNED
from .data import audit, load, read, write_new

_MODEL_CACHE = {}


@contextmanager
def invocation(a):
    """Separate execution records from scientific checkpoint identities."""
    identifier = uuid.uuid4().hex
    folder = a.out / "execution"
    started = time.time()
    job = {
        k: os.environ.get(k)
        for k in (
            "SLURM_JOB_ID",
            "SLURM_ARRAY_JOB_ID",
            "SLURM_ARRAY_TASK_ID",
            "SLURM_JOB_NODELIST",
        )
    }
    write_new(
        folder / f"{identifier}.started.json",
        dict(
            started_unix=started,
            stage=a.stage,
            model=a.model,
            rank=a.rank,
            shards=a.shards,
            config=read(a.config),
            slurm=job,
        ),
    )
    status = "failed"
    try:
        yield
        status = "completed"
    finally:
        write_new(
            folder / f"{identifier}.finished.json",
            dict(status=status, elapsed_seconds=time.time() - started, slurm=job),
        )


def publish(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Inputs changed; use a new output directory: {path}")
    else:
        write_new(path, value)


def inventory(a):
    packages = {k: importlib.metadata.version(k) for k in PINNED}
    for k, expected in PINNED.items():
        if packages[k].split("+")[0] != expected:
            raise ValueError(f"{k} must be {expected}; got {packages[k]}")
    location = a.model_path.resolve()
    config = read(location / "config.json")
    repo, kind, depth = MODELS[a.model]
    if (config["model_type"], config["num_hidden_layers"]) != (kind, depth):
        raise ValueError("Checkpoint architecture differs from selected model")
    shards = location / "model.safetensors.index.json"
    if shards.exists():
        if any(
            not (location / f).is_file()
            for f in set(read(shards)["weight_map"].values())
        ):
            raise ValueError("Missing model weight shard")
    elif not (location / "model.safetensors").is_file():
        raise ValueError("A complete safetensors checkpoint is required")
    override = a.spec["context_overrides"].get(a.model)
    limit = (
        override["max_position_embeddings"]
        if override
        else config["max_position_embeddings"]
    )
    dictionaries = {}
    if "sae" in a.spec["methods"]:
        from .sae import dictionary_record

        for layer in a.spec["sae_layers"]:
            dictionaries[str(layer)] = dictionary_record(a.sae_dir, layer)
    return dict(
        schema=1,
        model_key=a.model,
        spec=a.spec,
        python=platform.python_version(),
        packages=packages,
        model=dict(
            repo_id=repo,
            config=config,
        ),
        data=audit(a.root),
        dictionaries=dictionaries,
        effective_max_context=limit,
    )


def preflight(a):
    from transformers import AutoTokenizer
    from .runtime import prompt_ids, completion_ids

    result = inventory(a)
    tok = AutoTokenizer.from_pretrained(a.model_path, local_files_only=True)
    splits = {}
    for split in ("extraction", "eval"):
        rows, path = load(a.root, a.model, split)
        lengths = [
            max(len(prompt_ids(tok, row, a.model, p)) for p in (False, True))
            for row in rows
        ]
        if max(lengths) + 8 > result["effective_max_context"]:
            raise ValueError(f"{split}: context overflow; no truncation is allowed")
        splits[split] = dict(
            n=len(rows),
            sample_ids=[r["sample_id"] for r in rows],
            gold=[r["label"] for r in rows],
            max_prompt_tokens=max(lengths),
        )
    rows, _ = load(a.root, a.model, "extraction")
    boundaries = [completion_ids(tok, rows[0], a.model, label)[1] for label in (0, 1)]
    publish(
        a.out / "preflight.json",
        dict(inputs=result, splits=splits, boundaries=boundaries),
    )


def verified_preflight(a):
    saved = read(a.out / "preflight.json")
    if inventory(a) != saved["inputs"]:
        raise ValueError(
            "Preflight is stale: data summary, model settings, configuration or environment changed"
        )
    return saved


def model_for_run(a):
    from .runtime import load_model

    info = read(a.out / "preflight.json")["inputs"]
    key = (str(a.model_path.resolve()), info["effective_max_context"])
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_model(*key)
    return _MODEL_CACHE[key]


def protocol(a):
    from .runtime import SYSTEM, PROMPTED

    saved = verified_preflight(a)
    receipt = read(a.out / "validation.json")
    if not receipt["passed"] or receipt["preflight"] != saved:
        raise ValueError(
            "Run Extraction/Eval acceptance checks with these exact inputs first"
        )
    result = dict(
        kind="safety_portable",
        inputs=saved["inputs"],
        preflight=saved,
        validation=receipt,
        system=SYSTEM,
        prompted=PROMPTED,
        max_new_tokens=8,
        parser="outer_whitespace_then_exact_case_sensitive_safe_or_unsafe",
        completion_recipe="native_prefix_plus_literal_answer_no_end_marker",
        capture_site="last_lexical_answer_token",
        intervention_positions="last_prefill_and_each_decode",
        eval_batch_max_records=4,
        eval_batch_max_padded_tokens=12288,
        prediction_checkpoint_batches=8,
        batch_order="source_order",
        prefix_cache="unsteered_all_but_last_prompt_token",
        evaluation="complete_grid_reporting_without_operating_point_selection",
        gold_weights="equal_total_mass_each_class",
        truncation="none_fail_on_overflow",
        annotation="Model annotations; not human gold",
    )
    path = a.out / "protocol.json"
    result["run_id"] = read(path)["run_id"] if path.exists() else uuid.uuid4().hex
    publish(path, result)
    return result["run_id"]
