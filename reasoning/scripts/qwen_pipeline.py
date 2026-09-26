"""Frozen Qwen negatives -> 160 extraction pairs / 60 dev questions -> five steering baselines."""
from reasoning.datasets import add_extraction_argument
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from reasoning.common import ROOT, MAX_TOKENS, digest, dispatch, open_weight_model, read_jsonl, state, write_json
from reasoning.datasets import load
from reasoning.scripts.generate_negatives import messages

SPLIT = ROOT/'data/qwen/split.json'
PAIRS = SPLIT.with_name('pairs.jsonl')


def rebalance_split(original):
    if tuple(len(original[k]) for k in ('extraction', 'dev', 'eval')) != (200, 20, 280):
        raise ValueError('Expected original 200/20/280 split')
    groups = [set(original[k]) for k in ('extraction', 'dev', 'eval')]
    if len(set.union(*groups)) != 500:
        raise ValueError('Duplicate or overlapping original split')
    return dict(extraction=original['extraction'][:160],
                dev=original['dev'] + original['extraction'][160:],
                eval=list(original['eval']))


def prepare_split():
    original = json.loads((ROOT/'outputs/archive/data/math500/steering_run/split.json').read_text())
    split = rebalance_split(original)
    if SPLIT.exists() and json.loads(SPLIT.read_text()) != split:
        raise ValueError('Qwen split changed; use a new frozen source directory')
    write_json(SPLIT, split)
    return split


def build_pairs(positive, negative):
    original = load()
    ids = {r['id'] for r in original}
    pos = {r['id']: r for r in positive}
    neg = {r['id']: r for r in negative}
    if len(positive) != 500 or len(negative) != 500 or set(pos) != ids or set(neg) != ids:
        raise ValueError('Positive/negative data must each cover 500 unique questions')
    model = open_weight_model()['model']
    if any(r.get('model') != model or not r.get('completion', '').strip() for r in negative):
        raise ValueError('Negative model mismatch or empty completion')
    if any(not r.get('solution', '').strip() for r in positive):
        raise ValueError('Empty teacher solution')
    split = json.loads(SPLIT.read_text())
    lookup = {r['id']: r for r in original}
    pairs = [dict(id=i, problem=lookup[i]['problem'], teacher=pos[i]['solution'],
                  negative=neg[i]['completion']) for i in split['extraction']]
    if len(pairs) != 160 or {r['id'] for r in pairs} & set(split['dev'] + split['eval']):
        raise ValueError('Extraction split overlap')
    return pairs


def preflight():
    from transformers import AutoTokenizer
    from reasoning.models import TransformersLLM
    from reasoning.common import SUFFIX, COT_SUFFIX
    model = open_weight_model()
    path = Path(model['local_path'])
    index = json.loads((path/'model.safetensors.index.json').read_text())
    if not all((path/f).is_file() for f in set(index['weight_map'].values())):
        raise ValueError('Missing model weights')
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    # Exercise both generation code paths without loading weights on a GPU.
    ids = tokenizer.apply_chat_template(messages('qwen', 'Compute 1+1.'),
        tokenize=True, add_generation_prompt=True, return_dict=False, enable_thinking=False)
    if not ids or not all(isinstance(i, int) for i in ids):
        raise ValueError('Negative prompt must return integer token IDs')
    base = TransformersLLM.__new__(TransformersLLM)
    base.tokenizer, base.enable_thinking = tokenizer, False
    for suffix in [SUFFIX, COT_SUFFIX]:
        ids = base.prompt('Compute 1+1.', suffix)
        if not ids or not all(isinstance(i, int) for i in ids):
            raise ValueError('Evaluation prompt must return integer token IDs')
    return model


def run(args):
    split = prepare_split()
    model = preflight()
    manifest = {str(p.relative_to(ROOT)): digest(p) for p in ROOT.rglob('*.py')}
    write_json(args.output/'pipeline_config.json', dict(model=model, source_hashes=manifest,
        extraction_position=args.extraction_position,
        wait_pid=args.wait_pid, device=args.device, methods=['caa','pca','lat','probe','iti'],
        max_tokens=MAX_TOKENS, batch_size=4, enable_thinking=False,
        negative_count=500, extraction_count=len(split['extraction']),
        dev_count=len(split['dev']), eval_count=len(split['eval']),
        split_sha256=digest(SPLIT), split_path=str(SPLIT),
        positive_sha256=digest(ROOT/'data/positive.jsonl'),
        original_sha256=digest(ROOT/'data/original.jsonl')))
    state(args.output, 'waiting_for_previous_run', wait_pid=args.wait_pid, device=args.device)
    while Path(f'/proc/{args.wait_pid}').exists():
        time.sleep(30)
    state(args.output, 'waiting_for_gpu', device=args.device, required_free_mib=32000)
    while True:
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu',
                                       '--format=csv,noheader,nounits'], text=True)
        index = int(args.device.split(':')[1])
        if any(int(i) == index and int(free) >= 32000 and int(util) <= 10
               for i, free, util in (line.split(',') for line in rows.splitlines())):
            break
        time.sleep(30)
    if any(digest(ROOT/name) != value for name, value in manifest.items()):
        raise ValueError('Frozen source changed while queued')
    directory = args.output/'negatives'
    directory.mkdir(exist_ok=True)
    state(args.output, 'generating_negatives', progress=str(directory/'status.json'))
    with (directory/'run.log').open('a') as log:
        subprocess.run([sys.executable, '-u', '-m', 'reasoning.scripts.generate_negatives',
            'worker', '--family', 'qwen', '--model-path', model['local_path'], '--device', args.device,
            '--output', str(directory)], check=True, stdout=log, stderr=log, cwd=ROOT.parent)
    pairs = build_pairs(read_jsonl(ROOT/'data/positive.jsonl'), read_jsonl(ROOT/'data/negative_qwen.jsonl'))
    PAIRS.parent.mkdir(parents=True, exist_ok=True)
    PAIRS.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in pairs))
    from reasoning.datasets.benchmarks import load_suite
    _, audit = load_suite(split_path=SPLIT, pairs_path=PAIRS)
    write_json(args.output/'pairs_audit.json', dict(**audit, pairs_sha256=digest(PAIRS),
        negative_sha256=digest(ROOT/'data/negative_qwen.jsonl')))
    directory = args.output/'suite'
    directory.mkdir(exist_ok=True)
    state(args.output, 'running_baselines', progress=str(directory/'status.json'))
    with (directory/'run.log').open('a') as log:
        subprocess.run([sys.executable, '-u', '-m', 'reasoning.scripts.paper_baselines', 'run',
            '--methods', 'caa', 'pca', 'lat', 'probe', 'iti', '--device', args.device,
            '--extraction-position', args.extraction_position,
            '--max-tokens', str(MAX_TOKENS), '--batch-size', '4',
            '--output', str(directory)], check=True, stdout=log, stderr=log, cwd=ROOT.parent,
            env=dict(os.environ, REASONING_SPLIT=str(SPLIT)))
    state(args.output, 'completed', report=str(directory/'report.md'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start','run','status'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:2')
    parser.add_argument('--wait-pid', type=int, default=304734)
    add_extraction_argument(parser)
    dispatch(parser.parse_args(), run, 'reasoning.scripts.qwen_pipeline')
