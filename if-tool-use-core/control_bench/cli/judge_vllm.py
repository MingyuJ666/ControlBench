"""Run the blind judge with vLLM over judging directories.

Runs in the judge environment (JUDGE_PYTHON, default <repo>/.venv-judge/bin/python),
which has vLLM but not the rest of the benchmark. It therefore imports nothing
from control_bench: the domain's judge_core.py is loaded by file path, and the
prompt, the blind hashing, the parsing and the output format all come from it.
Run it as a file, never with -m.

  judge_vllm.py <judging_dir> ... [--domain instruction_following] [--model ...] [--force]

Each directory holds judge_in_*.jsonl shards written by the shard writer; the
script writes judge_out_*.jsonl and judge_unparsed_*.jsonl per shard and a
judge_run.json manifest. Completed verdicts are reused; missing verdicts are retried.
"""

import argparse
import importlib.util
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]


def load_core(domain: str):
    """The domain's judge_core.py, imported by path so the package is not needed."""
    path = PACKAGE / "domains" / domain / "judge_core.py"
    if not path.exists():
        # The standalone judge interpreter does not have control_bench installed.
        project = {"instruction_following": "instruction-following",
                   "tool_use": "tool-use", "tool_use_multi_turn": "tool-use"}.get(domain)
        if project is not None:
            path = PACKAGE.parents[1] / project / "src" / domain / "judge_core.py"
    if not path.exists():
        raise SystemExit(f"domain {domain!r} has no judge_core.py at {path}")
    spec = importlib.util.spec_from_file_location("judge_core", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


class UnparsedVerdictsError(RuntimeError):
    """The judge returned no valid decision for one or more inputs."""


def make_retry_sampling(core):
    """Use the structured-output API supported by the installed vLLM."""
    if not hasattr(core, "retry_pattern"):
        return None
    from vllm import SamplingParams
    from vllm import sampling_params

    pattern = core.retry_pattern()
    if hasattr(sampling_params, "StructuredOutputsParams"):
        guided = sampling_params.StructuredOutputsParams(regex=pattern)
        options = {"structured_outputs": guided}
    else:
        guided = sampling_params.GuidedDecodingParams(regex=pattern)
        options = {"guided_decoding": guided}
    return SamplingParams(temperature=0.0, max_tokens=core.RETRY_MAX_NEW, **options)


def judge_shard(llm, tokenizer, core, sampling, in_path: Path, *, max_attempts: int = 3,
                retry_sampling=None, max_model_len: int | None = None) -> None:
    """Resume valid verdicts; switch malformed IF replies to bounded evidence output."""
    items = [json.loads(line) for line in in_path.read_text().splitlines()]
    ids = [item["uid"] for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate input UID in {in_path}")
    out_path = in_path.with_name(in_path.name.replace("judge_in_", "judge_out_"))
    unparsed_path = in_path.with_name(in_path.name.replace("judge_in_", "judge_unparsed_"))
    verdicts = {}
    if out_path.exists():
        for row in (json.loads(line) for line in out_path.read_text().splitlines()):
            uid = row.get("uid")
            if uid in ids and core._fields(row) is not None:
                if (hasattr(core, "retry_spec") and row.get("judge_mode")
                        and row.get("retry_spec") != core.retry_spec()):
                    continue
                if uid in verdicts:
                    raise ValueError(f"duplicate verdict UID {uid!r} in {out_path}")
                verdicts[uid] = row
    failures = {}
    if unparsed_path.exists():
        failures = {row["uid"]: row for row in
                    (json.loads(line) for line in unparsed_path.read_text().splitlines())}
    pending = [item for item in items if item["uid"] not in verdicts]
    prompts = {}
    oversized = {}
    output_budget = max(core.MAX_NEW, getattr(core, "RETRY_MAX_NEW", core.MAX_NEW))
    for item in pending:
        uid = item["uid"]
        prompt = core.render(tokenizer, core.build_prompt(item))
        if max_model_len is None:
            prompts[uid] = prompt
            continue
        # Match vLLM's text-prompt encoding, then submit these exact IDs so the
        # capacity check and inference use the same complete input.
        token_ids = tokenizer.encode(prompt)
        required = len(token_ids) + output_budget
        if required > max_model_len:
            oversized[uid] = required
            failures[uid] = {"uid": uid, "raw_response": "",
                             "error": "context_length_exceeded",
                             "input_tokens": len(token_ids), "max_new_tokens": output_budget,
                             "max_model_len": max_model_len, "required_model_len": required}
        else:
            prompts[uid] = {"prompt_token_ids": token_ids}
            if failures.get(uid, {}).get("error") == "context_length_exceeded":
                failures.pop(uid)

    def save():
        atomic_write(out_path, "".join(json.dumps(verdicts[uid], ensure_ascii=False) + "\n"
                                      for uid in ids if uid in verdicts))
        atomic_write(unparsed_path, "".join(json.dumps(failures[uid], ensure_ascii=False) + "\n"
                                           for uid in ids if uid not in verdicts and uid in failures))

    if oversized:
        save()
        print(f"  {in_path}: {len(oversized)} inputs exceed context length {max_model_len}; "
              f"up to {max(oversized.values())} tokens needed including output. "
              "Recorded as unjudged; continuing other inputs.", flush=True)
    pending = [item for item in pending if item["uid"] not in oversized]
    attempts = 0
    for attempt in range(max_attempts):
        if not pending:
            break
        bounded = retry_sampling is not None and (attempt > 0 or all(item["uid"] in failures for item in pending))
        parse = core.parse_retry if bounded else core.parse
        outputs = llm.generate([prompts[item["uid"]] for item in pending],
                               retry_sampling if bounded else sampling, use_tqdm=False)
        attempts += 1
        if len(outputs) != len(pending):
            raise RuntimeError(f"judge returned {len(outputs)} outputs for {len(pending)} inputs")
        for item, output in zip(pending, outputs, strict=True):
            uid = item["uid"]
            text = output.outputs[0].text
            verdict = parse(text, uid)
            if verdict is not None:
                if bounded and uid in failures:
                    verdict["previous_raw_response"] = failures[uid]["raw_response"]
                if max_model_len is not None:
                    verdict.update(input_tokens=len(prompts[uid]["prompt_token_ids"]),
                                   max_model_len=max_model_len)
                verdicts[uid] = verdict
                failures.pop(uid, None)
            else:
                failures[uid] = {"uid": uid, "raw_response": text,
                                 "judge_mode": core.RETRY_MODE if bounded else "evidence",
                                 "finish_reason": getattr(output.outputs[0], "finish_reason", None)}
        pending = [item for item in pending if item["uid"] not in verdicts]
        save()
        if bounded:
            break  # A second identical constrained greedy request cannot repair a backend failure.
    pending = [item for item in items if item["uid"] not in verdicts]
    save()
    print(f"  {in_path.parent.parent.name}/{in_path.name}: {len(verdicts)} verdicts, {len(pending)} unparsed", flush=True)
    if pending:
        context_error = (f"; {len(oversized)} inputs exceed --max-model-len={max_model_len}, "
                         f"need up to {max(oversized.values())} tokens" if oversized else "")
        raise UnparsedVerdictsError(f"{in_path}: {len(pending)} judge verdicts missing after {attempts} attempts"
                                   + context_error)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--domain", default="instruction_following")
    ap.add_argument("--model", default=None, help="default: the domain's own judge model")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="total input + output capacity (default: the domain context window, or 8192 if unspecified)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                    help="lower it when the card is shared with another job")
    args = ap.parse_args()
    core = load_core(args.domain)
    if args.max_model_len is None:
        args.max_model_len = getattr(core, "MAX_MODEL_LEN", 8192)
    if args.max_model_len <= max(core.MAX_NEW, getattr(core, "RETRY_MAX_NEW", core.MAX_NEW)):
        ap.error("--max-model-len must leave room for the prompt and the judge's output tokens")
    if args.model is None:
        args.model = core.JUDGE_MODEL

    import torch
    import transformers
    import vllm
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, enable_prefix_caching=True, seed=0)
    tokenizer = llm.get_tokenizer()
    revision = getattr(getattr(llm.llm_engine, "model_config", None), "revision", None)
    sampling = SamplingParams(temperature=0.0, max_tokens=core.MAX_NEW)
    retry_sampling = make_retry_sampling(core)
    versions = {"vllm_version": vllm.__version__, "transformers_version": transformers.__version__,
                "torch_version": torch.__version__}
    print(f"judge={args.model} via vLLM {vllm.__version__}", flush=True)

    errors = []
    for directory in args.dirs:
        in_paths = sorted(directory.glob("judge_in_*.jsonl"))
        if not in_paths:
            raise SystemExit(f"no judge_in_*.jsonl in {directory}")
        manifest_path = directory / "judge_run.json"
        expected = {**core.run_spec([p.name for p in in_paths], args.model, "vllm", versions),
                    "model_revision": revision}
        if args.force:
            for p in list(directory.glob("judge_out_*.jsonl")) + list(directory.glob("judge_unparsed_*.jsonl")):
                p.unlink()
            manifest_path.unlink(missing_ok=True)
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            semantic = ("schema_version", "model", "backend", "prompt_sha256", "max_new_tokens", "do_sample", "dtype", "input_shards")
            mismatched = [k for k in semantic if previous.get(k) != expected.get(k)]
            if mismatched:
                raise SystemExit(f"{directory}: existing verdicts came from a different judge run ({', '.join(mismatched)}); use --force")
        else:
            previous = expected
        previous = {**previous, "max_model_len": args.max_model_len,
                    "input_overflow_policy": "leave_unjudged"}
        if retry_sampling is not None:
            previous = {**previous, "unparsed_retry": core.retry_spec()}
        atomic_write(manifest_path, json.dumps(previous, ensure_ascii=False, indent=2))
        for in_path in in_paths:
            try:
                judge_shard(llm, tokenizer, core, sampling, in_path, retry_sampling=retry_sampling,
                            max_model_len=args.max_model_len)
            except UnparsedVerdictsError as exc:
                errors.append(str(exc))
                print(f"  incomplete: {exc}", flush=True)
    if errors:
        raise UnparsedVerdictsError("\n".join(errors))


if __name__ == "__main__":
    main()
