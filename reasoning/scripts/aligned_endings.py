"""Isolated, fixed-configuration ablation of contrastive extraction endings."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import time

from reasoning.common import (ROOT, PAIRS, SYSTEM, SUFFIX, code_hashes, digest,
                              dispatch, open_weight_model, prepare_run,
                              read_jsonl, state, write_json)
from reasoning.datasets import reasoning_prefix, problem_id

ANCHOR = '\n\nFinal answer:'
DELIMITER = re.compile(r'^(?:\\\[|\\\]|\\\(|\\\)|\$\$|\$)$')


def aligned_prefix(text):
    """Keep the historical reasoning prefix, strip orphan display delimiters.

    No answer is inserted. Existing reasoning may already mention an answer,
    as in the original extraction; this is not an answer-redaction procedure.
    """
    prefix = reasoning_prefix(text)
    lines = prefix.splitlines()
    while lines and (not lines[-1].strip() or DELIMITER.fullmatch(lines[-1].strip())):
        lines.pop()
    body = '\n'.join(lines).rstrip()
    if not body:
        raise ValueError('No reasoning body before final answer')
    return body + ANCHOR


def build_records(tokenizer, pairs):
    records, details = [], []
    for row in pairs:
        prompt_ids = tokenizer.apply_chat_template(
            [{'role': 'system', 'content': SYSTEM},
             {'role': 'user', 'content': row['problem'] + SUFFIX}],
            tokenize=True, add_generation_prompt=True, return_dict=False)
        prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        if tokenizer.encode(prompt, add_special_tokens=False) != prompt_ids:
            raise ValueError('Prompt token roundtrip changed')
        record = {'id': row['id']}
        detail = {'id': row['id']}
        for key, source in [('positive', 'teacher'), ('negative', 'llama')]:
            old = reasoning_prefix(row[source])
            aligned = aligned_prefix(row[source])
            full = prompt + aligned
            old_ids = tokenizer.encode(prompt + old, add_special_tokens=False)
            ids = tokenizer.encode(full, add_special_tokens=False)
            if not old or max(len(old_ids), len(ids)) > 16384:
                raise ValueError('A pair would be excluded; ablation requires identical pairs')
            record[key] = full
            detail[key] = dict(old_last_token=tokenizer.convert_ids_to_tokens(old_ids[-1]),
                               new_last_token=tokenizer.convert_ids_to_tokens(ids[-1]),
                               new_last_id=ids[-1], suffix_ids=ids[-3:],
                               old_tokens=len(old_ids), new_tokens=len(ids),
                               old_tail=old[-120:], new_tail=aligned[-120:])
        records.append(record)
        details.append(detail)
    if len(records) < 8 or len({r['id'] for r in records}) != len(records):
        raise ValueError('Too few pairs or duplicate IDs')
    suffixes = {tuple(r[k]['suffix_ids']) for r in details for k in ('positive', 'negative')}
    if len(suffixes) != 1:
        raise ValueError('Aligned suffix tokens differ between examples')
    diagnostics = {'pairs': len(records), 'anchor': ANCHOR,
                   'capture': 'last_non_padding_token (colon in identical suffix)',
                   'common_suffix_ids': list(next(iter(suffixes))), 'examples': details}
    for key in ('positive', 'negative'):
        diagnostics[key] = {field: dict(Counter(r[key][field] for r in details))
                            for field in ('old_last_token', 'new_last_token')}
    return records, diagnostics


def inputs(args):
    source = args.source.resolve()
    old = json.loads((source / 'config.json').read_text())
    selected = json.loads((source / 'selected_steering.json').read_text())['candidate']
    local = open_weight_model()
    if old['method'] != 'diffmean' or selected not in old['grid']:
        raise ValueError('Expected frozen DiffMean selection')
    if any(old[k] != local[k] for k in ('model', 'revision')):
        raise ValueError('Current model differs from source')
    for name, expected in old['files'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Source input changed: {name}')
    pairs = read_jsonl(PAIRS)
    ids = old['eval_ids']
    if not ids or len(set(ids)) != len(ids) or set(ids) & {r['id'] for r in pairs}:
        raise ValueError('Invalid or overlapping evaluation IDs')
    return source, old, selected, local, pairs


def prepare(args):
    from transformers import AutoTokenizer
    _, _, _, local, pairs = inputs(args)
    tokenizer = AutoTokenizer.from_pretrained(local['local_path'], local_files_only=True)
    records, diagnostics = build_records(tokenizer, pairs)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'alignment_preview.json', diagnostics)
    print(json.dumps({k: v for k, v in diagnostics.items() if k != 'examples'}, indent=2))
    return records


def paired_result(baseline, treatment):
    other = {r['id']: r for r in treatment}
    wins = sum(not r['correct'] and other[r['id']]['correct'] for r in baseline)
    losses = sum(r['correct'] and not other[r['id']]['correct'] for r in baseline)
    return {'wins': wins, 'losses': losses, 'delta_correct': wins - losses}


def run(args):
    import torch
    from reasoning.steering import RepSteerBackend, rs, vector
    from reasoning.evaluation import evaluate
    source, old, candidate, local, pairs = inputs(args)
    artifact_path = source / 'artifacts' / f'L{candidate["layer"]}'
    config = dict(experiment='aligned_endings_ablation', source=str(source),
                  method='diffmean', candidate=candidate, anchor=ANCHOR,
                  eval_ids=old['eval_ids'], budget=old['eval_budget'],
                  batch_size=old['batch_size'], seed=old['seed'],
                  model=local['model'], revision=local['revision'], device=args.device,
                  code=code_hashes(), script_sha256=digest(__file__),
                  pairs_sha256=digest(PAIRS), source_config_sha256=digest(source/'config.json'),
                  old_artifact_sha256=digest(artifact_path/'tensors.safetensors'),
                  old_manifest_sha256=digest(artifact_path/'manifest.json'),
                  selection_sha256=digest(source/'selected_steering.json'))
    config_path = args.output/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Configuration changed; choose a new output directory')
    write_json(config_path, config)
    if args.wait_pid:
        state(args.output, 'waiting_for_previous_run', wait_pid=args.wait_pid)
        while Path(f'/proc/{args.wait_pid}').exists():
            time.sleep(30)
    # Reject changes made while queued before loading a model or learning.
    if code_hashes() != config['code'] or digest(__file__) != config['script_sha256']:
        raise ValueError('Code changed while queued')
    inputs(args)
    torch.set_num_threads(4)
    torch.manual_seed(old['seed'])
    prepare_run(args.output, config, args.device)
    backend = RepSteerBackend(args.device, layer=candidate['layer'])
    records, diagnostics = build_records(backend.base.tokenizer, pairs)
    write_json(args.output/'alignment.json', diagnostics)
    write_json(args.output/'aligned_records.json', records)
    old_artifact = rs.load_artifact(artifact_path)
    old_artifact.bind(backend.model)
    if old_artifact.metadata.config.get('positive_samples') != len(records):
        raise ValueError('Historical artifact uses different pair count')
    state(args.output, 'learning_aligned_vector', pairs=len(records))
    aligned = backend.fit('diffmean', rs.data.ContrastivePairs.from_records(records),
                          args.output/'artifacts'/'aligned')
    write_json(args.output/'vector_comparison.json', {
        'cosine_old_aligned': torch.nn.functional.cosine_similarity(
            vector(old_artifact).float().reshape(1, -1),
            vector(aligned).float().reshape(1, -1)).item(),
        'old_norm': vector(old_artifact).norm().item(),
        'aligned_norm': vector(aligned).norm().item()})
    lookup = {problem_id(r['problem']): dict(r, id=problem_id(r['problem']))
              for r in read_jsonl(ROOT/'data/original.jsonl')}
    rows = [lookup[i] for i in old['eval_ids']]
    results, generated = [], {}
    for name, artifact, setting in [('baseline', None, None),
                                    ('original', old_artifact, candidate),
                                    ('aligned', aligned, candidate)]:
        result = evaluate(backend, artifact, setting, rows, old['eval_budget'],
                          name, config, args.output, old['batch_size'])
        result['condition'] = name
        results.append(result)
        generated[name] = read_jsonl(args.output/f'{name}_{result["label"]}.jsonl')
        write_json(args.output/'results.json', results)
    report = {'results': results,
              'aligned_vs_original': paired_result(generated['original'], generated['aligned']),
              'aligned_vs_baseline': paired_result(generated['baseline'], generated['aligned']),
              'note': 'Frozen previously selected configuration; reused historical evaluation set. '
                      'Only extraction ending changes; no evaluation-based tuning. '
                      'Does not isolate all teacher style or generation-position differences.'}
    write_json(args.output/'report.json', report)
    state(args.output, 'completed', **report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'start', 'run', 'status'])
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:2')
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args)
    else:
        dispatch(args, run, 'reasoning.scripts.aligned_endings')
