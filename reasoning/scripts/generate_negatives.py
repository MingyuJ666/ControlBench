"""Background, resumable full-MATH500 negative generation for three models."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from reasoning.common import ROOT, SYSTEM, SUFFIX, Journal, digest, dispatch, read_jsonl, state, write_json
from reasoning.datasets import load

MODELS = {
    'llama': ('meta-llama/Llama-3.1-8B-Instruct', '0e9e39f249a16976918f6564b8830bc894c89659'),
    'gemma': ('google/gemma-2-9b-it', '11c9b309abf73637e4b6f9a3fa1e92e615547819'),
    'qwen': ('Qwen/Qwen3-8B', 'b968826d9c46dd6066d109eabc6255188de91218'),
}


def messages(family, question):
    if family == 'gemma':
        return [{'role': 'user', 'content': SYSTEM + '\n\n' + question + SUFFIX}]
    return [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': question + SUFFIX}]


def configuration(family):
    name, revision = MODELS[family]
    return dict(model=name, revision=revision, data_sha256=digest(ROOT/'data/original.jsonl'),
                script_sha256=digest(__file__), max_new_tokens=8192,
                output_budget_policy='min(8192, model_context - padded_prompt_tokens)', do_sample=False,
                batch_size=2, seed=42, system=SYSTEM, suffix=SUFFIX,
                system_in_user=family == 'gemma', enable_thinking=False if family == 'qwen' else None,
                label='negative', label_basis='model source, not answer correctness',
                scope='all 500 MATH500; downstream extraction must filter by fixed split')


def worker(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from reasoning.evaluation import score
    torch.set_num_threads(4)
    torch.manual_seed(42)
    family = args.family
    rows = load()
    if len(rows) != 500 or len({r['id'] for r in rows}) != 500:
        raise ValueError('Expected 500 unique MATH500 questions')
    target = ROOT/f'data/negative_{family}.jsonl'
    if target.exists():
        existing = read_jsonl(target)
        if ({r['id'] for r in existing} != {r['id'] for r in rows} or len(existing) != 500 or
                any(r.get('model') != MODELS[family][0] for r in existing)):
            raise ValueError(f'Existing {target} is incomplete or from another model; move it away to regenerate')
        state(args.output, 'completed', model=family, path=str(target), sha256=digest(target))
        return
    config = configuration(family)
    # Checkpoint and metadata stay in the run directory; data/ receives only the finished file.
    journal = Journal(args.output/f'negative_{family}.jsonl', config)
    if not journal.ids <= {r['id'] for r in rows}:
        raise ValueError('Output contains unexpected IDs')
    pending = [r for r in rows if r['id'] not in journal.ids]
    if pending:
        state(args.output, 'loading_model', model=family, device=args.device, completed=len(journal.rows), total=500)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True,
                    dtype=torch.bfloat16, attn_implementation='sdpa').to(args.device).eval()
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        stop = model.generation_config.eos_token_id
        stop = stop if isinstance(stop, list) else [stop if stop is not None else tokenizer.eos_token_id]
        state(args.output, 'generating', model=family, device=args.device, completed=len(journal.rows), total=500)
        for offset in range(0, len(pending), 2):
            batch = pending[offset:offset + 2]
            options = {'enable_thinking': False} if family == 'qwen' else {}
            prompts = [tokenizer.apply_chat_template(messages(family, r['problem']),
                tokenize=True, add_generation_prompt=True, return_dict=False, **options) for r in batch]
            width = max(map(len, prompts))
            budget = min(8192, model.config.max_position_embeddings - width)
            if budget <= 0:
                raise ValueError('Prompt exhausts model context')
            ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long, device=args.device)
            mask = torch.zeros_like(ids)
            for i, prompt in enumerate(prompts):
                ids[i, -len(prompt):] = torch.tensor(prompt, device=args.device)
                mask[i, -len(prompt):] = 1
            start = time.monotonic()
            with torch.inference_mode():
                outputs = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=budget,
                    do_sample=False, use_cache=True, eos_token_id=stop, pad_token_id=tokenizer.pad_token_id)
            elapsed = time.monotonic() - start
            for row, output in zip(batch, outputs[:, width:].tolist()):
                end = next((i+1 for i, t in enumerate(output) if t in stop), None)
                tokens = output[:end] if end is not None else output
                truncated = end is None and len(tokens) >= budget
                completion = tokenizer.decode(tokens, skip_special_tokens=True)
                journal.append(dict(row, completion=completion, tokens=len(tokens), truncated=truncated,
                    max_new_tokens=budget, padded_prompt_tokens=width,
                    **score(completion, row['answer'], truncated), model=MODELS[family][0],
                    label='negative', label_basis=config['label_basis'], seconds=elapsed/len(batch)))
            state(args.output, 'generating', model=family, device=args.device, completed=len(journal.rows), total=500)
    if journal.ids != {r['id'] for r in rows}:
        raise ValueError('Incomplete output')
    shutil.copyfile(journal.path, target)
    summary = dict(model=family, completed=len(journal.rows), total=500,
                   correct=sum(r['correct'] for r in journal.rows),
                   truncated=sum(r['truncated'] for r in journal.rows),
                   path=str(target), sha256=digest(target))
    write_json(args.output/'summary.json', summary)
    state(args.output, 'completed', **summary)


def available_gpu():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu',
                                      '--format=csv,noheader,nounits'], text=True)
    candidates = []
    for line in output.splitlines():
        index, free, util = map(int, line.split(','))
        if free >= 32000 and util <= 10:
            candidates.append((free, index))
    return f'cuda:{max(candidates)[1]}' if candidates else None


def run(args):
    from huggingface_hub import snapshot_download
    paths, failures = {}, {}
    for family, (name, revision) in MODELS.items():
        state(args.output, 'preparing_model', model=family)
        local = Path.home()/'.cache/huggingface/hub'/('models--'+name.replace('/', '--'))/'snapshots'/revision
        try:
            if family == 'llama' and local.exists():
                paths[family] = str(local)
            else:
                paths[family] = snapshot_download(name, revision=revision,
                    local_dir=ROOT/'outputs/v2/negative_models'/name.replace('/', '--'),
                    allow_patterns=['*.json', '*.safetensors', '*.model', '*.jinja', '*.txt'], max_workers=4)
            # Validate all prompt templates on CPU before waiting for a GPU.
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(paths[family], local_files_only=True)
            options = {'enable_thinking': False} if family == 'qwen' else {}
            tokenizer.apply_chat_template(messages(family, 'Compute 1+1.'), tokenize=True,
                                          add_generation_prompt=True, **options)
        except Exception as exc:
            failures[family] = f'{type(exc).__name__}: {exc}'
            state(args.output, 'model_preparation_failed', model=family, error=failures[family])
    for family, path in paths.items():
        if family in failures:
            continue
        for attempt in range(1, 4):
            device = available_gpu()
            state(args.output, 'waiting_for_gpu', model=family, required_free_mib=32000, attempt=attempt)
            while device is None:
                time.sleep(30)
                device = available_gpu()
            state(args.output, 'running_model', model=family, device=device, attempt=attempt)
            directory = args.output/family
            directory.mkdir(exist_ok=True)
            with (directory/'run.log').open('a') as log:
                result = subprocess.run([sys.executable, '-u', '-m', 'reasoning.scripts.generate_negatives',
                    'worker', '--family', family, '--model-path', path, '--device', device,
                    '--output', str(directory)], cwd=ROOT.parent, stdout=log, stderr=log)
            if result.returncode == 0:
                break
            if attempt == 3:
                failures[family] = f'Worker failed three times; see {directory}/run.log'
            else:
                time.sleep(30)
    state(args.output, 'failed' if failures else 'completed', failures=failures,
          outputs={f: str(ROOT/f'data/negative_{f}.jsonl') for f in MODELS})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'run', 'status', 'worker'])
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/v2/negative_full_20260914')
    parser.add_argument('--family', choices=list(MODELS))
    parser.add_argument('--model-path')
    parser.add_argument('--device')
    args = parser.parse_args()
    if args.action == 'worker':
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output/'generation.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                worker(args)
            except Exception as exc:
                state(args.output, 'failed', error=f'{type(exc).__name__}: {exc}')
                raise
    else:
        dispatch(args, run, 'reasoning.scripts.generate_negatives')
