"""Reproducible reasoning-prefix steering baseline suite; select only on dev."""
from reasoning.datasets import (add_extraction_argument, EXTRACTION_POSITIONS,
    extraction_positions, expand_extraction_grid)
from reasoning.common import steering_artifact_path
from reasoning.datasets.contrastive import prepare_pairs
import argparse
import json
from pathlib import Path
import time

from reasoning.common import (ROOT, SPLIT, MAX_TOKENS, SYSTEM, SUFFIX, COT_SUFFIX, Journal,
    code_hashes, digest, dispatch, open_weight_model, prepare_run, read_jsonl,
    state, steering_grid, write_json)
from reasoning.datasets import load
from reasoning.datasets.benchmarks import load_suite
from reasoning.evaluation import score

DEFAULT_METHODS = ('caa', 'pca', 'lat', 'probe', 'iti')
METHODS = (*DEFAULT_METHODS, 'diffmean', 'sae')
STRENGTH_GRID_FOR_ITI = [1, 5, 10, 15, 20]


def grid(method, layers, strengths):
    sites = [-1] if method == 'iti' else layers
    if method == 'iti':
        strengths = STRENGTH_GRID_FOR_ITI
    return [dict(layer=l, alpha=a, phase='decode') for l in sites for a in strengths]


def condition_label(candidate):
    if candidate is None:
        return 'baseline'
    prefix = candidate.get('extraction_position', '')
    return (prefix + '_' if prefix else '') + f'L{candidate["layer"]}_a{candidate["alpha"]:g}'


def control_plan(rs, artifact, candidate):
    if artifact.metadata.method == 'iti':
        return rs.recipes.iti(artifact, strength=candidate['alpha'], phase='decode')
    return rs.SteeringPlan([rs.Intervention(artifact=artifact,
        operator=rs.operators.Add(), positions=rs.positions.GeneratedTokens(),
        strength=rs.schedules.Constant(candidate['alpha']), phase='decode')])


def evaluate(backend, rows, artifact, candidate, budget, output, config, batch_size,
             prompt_suffix=None):
    from contextlib import nullcontext
    from reasoning.steering import rs
    if prompt_suffix is not None:
        if artifact is not None or candidate is not None:
            raise ValueError('Prompt baseline must not use steering')
        config = dict(config, prompt_suffix=prompt_suffix, budget=budget)
    journal = Journal(output, config)
    if candidate is None and artifact is None:
        from reasoning.baseline_cache import reuse
        reuse(journal, rows, config, budget, prompt_suffix)
    for offset in range(0, len(rows), batch_size):
        batch = [r for r in rows[offset:offset + batch_size] if r['id'] not in journal.ids]
        if not batch:
            continue
        start = time.monotonic()
        context = (backend.model.steer(control_plan(rs, artifact, candidate))
                   if candidate is not None else nullcontext())
        with context:
            kwargs = {} if prompt_suffix is None else dict(prompt_suffix=prompt_suffix)
            answers = backend.generate([r['problem'] for r in batch], max_tokens=budget, **kwargs)
        seconds = (time.monotonic() - start) / len(batch)
        if len(answers) != len(batch):
            raise ValueError('Generation batch size mismatch')
        for row, answer in zip(batch, answers):
            if row['id'] not in journal.ids:
                journal.append({**row, **answer, **score(answer['completion'], row['answer']),
                                'seconds': seconds})
        state(output.parent, 'evaluating', condition=output.stem,
              completed=len(journal.rows), total=len(rows))
    result = dict(candidate=candidate, completed=len(journal.rows), total=len(rows),
                  correct=sum(r['correct'] for r in journal.rows),
                  missing_final_answer=sum(r['parsed_answer'] is None for r in journal.rows),
                  truncated=sum(r['truncated'] for r in journal.rows),
                  tokens=sum(r['tokens'] for r in journal.rows))
    result['reused'] = sum('baseline_source' in r for r in journal.rows)
    result['accuracy'] = result['correct'] / len(rows)
    write_json(output.with_suffix('.summary.json'), result)
    return result


def fit(backend, method, layer, data, path, sae_weights=None):
    import torch
    from reasoning.steering import rs, learner
    if (path / 'manifest.json').exists():
        artifact = rs.load_artifact(path)
        backend.check_extraction_position(artifact)
        artifact.bind(backend.model)
        return artifact
    if method == 'sae':
        from reasoning.experiments.sae import fit_sae
        artifact = fit_sae(backend, data, layer, sae_weights)
        artifact.save(path)
        return artifact
    common = dict(positions=rs.positions.LastNonPaddingToken(), batch_size=1,
                  apply_chat_template=False, special_tokens=False,
                  capture_dtype=torch.float32, seed=42)
    if method == 'pca':
        model = rs.learners.PCA(site=rs.sites.resid_post(layer), n_components=1, **common)
    elif method == 'iti':
        model = rs.learners.ITI(top_k=48, validation_fraction=0.2, **common)
    else:
        model = learner('diffmean' if method == 'caa' else method, layer)
    artifact = model.fit(backend.model, data, store=backend.store)
    tensors = list(artifact.tensors().values())
    if not all(torch.isfinite(t).all() for t in tensors):
        raise ValueError('Nonfinite artifact')
    artifact = backend.tag_extraction_position(artifact)
    artifact.save(path)
    return artifact


def run(args):
    import torch
    from reasoning.steering import RepSteerBackend
    local = open_weight_model(args.model_index)
    if local['model'] not in ('meta-llama/Llama-3.1-8B-Instruct', 'Qwen/Qwen3-8B', 'google/gemma-2-9b-it'):
        raise ValueError('Unsupported model for this suite')
    split = json.loads(SPLIT.read_text())
    datasets, audit = load_suite()
    lookup = {r['id']: r for r in load()}
    dev = [lookup[i] for i in split['dev']]
    layers, strengths = steering_grid(local['model'])
    grids = search_grids(args, layers, strengths)
    positions = extraction_positions(args.extraction_position)
    pair_path = prepare_pairs(local['model'], args.output)
    if args.smoke:
        dev = dev[:2]
        datasets = {k: v[:1] for k, v in datasets.items()}
    config = dict(model=local['model'], revision=local['revision'],
        methods=args.methods, grid=grids, system=SYSTEM, suffix=SUFFIX,
        baselines=['baseline', 'cot'], prompts=dict(system=SYSTEM, baseline=SUFFIX, cot=COT_SUFFIX),
        system_in_user=local['model'] == 'google/gemma-2-9b-it',
        output_budget_policy=('remaining_context' if local['model'] == 'google/gemma-2-9b-it' else 'fixed'),
        dev_ids=[r['id'] for r in dev], tests={k:[r['id'] for r in v] for k,v in datasets.items()},
        max_tokens=args.max_tokens, dev_tokens=args.max_tokens, batch_size=args.batch_size,
        seed=42, do_sample=False, extraction={p: EXTRACTION_POSITIONS[p] for p in positions}, extraction_position=args.extraction_position,
        labels=f"positive=teacher source; negative={local['model']} source; not correctness labels",
        enable_thinking=False if local['model'] == 'Qwen/Qwen3-8B' else None,
        selection='best dev correct; ties smaller alpha then lower layer; always report steering candidate',
        iti='repsteer ITI, 48 heads, internal stratified 80/20 probe split; source-label adaptation',
        sae=(dict(weights=str(args.sae_weights.resolve()), sha256=digest(args.sae_weights), layer=args.sae_layer) if 'sae' in args.methods else None),
        audit=audit, files={str(p):digest(p) for p in [pair_path,SPLIT,ROOT/'data/original.jsonl',
            ROOT/'data/benchmarks/manifest.json', ROOT/'data/benchmarks/amc2023.jsonl',
            ROOT/'data/benchmarks/aime2024.jsonl', ROOT/'outputs/repsteer_tuning/config.json']},
        repsteer_code={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'vendor/repsteer/src').rglob('*.py'))},
        torch=torch.__version__, transformers=__import__('transformers').__version__,
        code=code_hashes(), script=digest(__file__),
        device=args.device, smoke=args.smoke)
    existing = args.output / 'config.json'
    if existing.exists() and json.loads(existing.read_text()) != config:
        raise ValueError('Configuration changed; use a new output directory')
    write_json(existing, config)
    if args.wait_pid:
        state(args.output, 'waiting_for_previous_run', wait_pid=args.wait_pid)
        while Path(f'/proc/{args.wait_pid}').exists():
            time.sleep(30)
    if code_hashes() != config['code'] or digest(__file__) != config['script']:
        raise ValueError('Source changed while queued; use frozen source checkout')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    prepare_run(args.output, config, args.device)
    backend = RepSteerBackend(args.device, model_config=local,
        extraction_position=positions[0],
        enable_thinking=False if local['model'] == 'Qwen/Qwen3-8B' else None)
    if args.smoke:
        records = read_jsonl(pair_path)[:8]
        pair_path = args.output/'smoke_pairs.jsonl'
        pair_path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    data_by_position = {}
    for position in positions:
        backend.extraction_position = position
        data, valid = backend.contrastive_data(pair_path)
        if not set(valid) <= set(split['extraction']):
            raise ValueError('Extraction split overlap')
        data_by_position[position] = (data, valid)
        audit_name = f'extraction_{position}.json' if len(positions) > 1 else 'extraction.json'
        write_json(args.output/audit_name, dict(valid_ids=valid, pairs=len(valid),
            extraction_position=position,
            excluded_ids=[r['id'] for r in read_jsonl(pair_path) if r['id'] not in valid]))
    results = {}
    for dataset, rows in datasets.items():
        results[dataset] = {}
        for name, suffix in [('baseline', None), ('cot', COT_SUFFIX)]:
            path = args.output/f'{dataset}_{name}_baseline.jsonl'
            # Keep the historical ordinary-baseline filename.
            if name == 'baseline':
                path = args.output/f'{dataset}_baseline.jsonl'
            results[dataset][name] = evaluate(backend, rows, None, None,
                args.max_tokens, path, dict(run=config, stage=dataset, candidate=None),
                args.batch_size, prompt_suffix=suffix)
        write_report(args.output, results, args.extraction_position)
    for name, suffix in [('baseline', None), ('cot', COT_SUFFIX)]:
        filename = 'dev_baseline.jsonl' if name == 'baseline' else 'dev_cot_baseline.jsonl'
        evaluate(backend, dev, None, None, args.max_tokens, args.output/filename,
                 dict(run=config, stage='dev', candidate=None), args.batch_size,
                 prompt_suffix=suffix)
    for method in args.methods:
        scores, artifacts = [], {}
        for position in positions:
            backend.extraction_position = position
            data, valid = data_by_position[position]
            for layer in sorted({c['layer'] for c in grids[method]}):
                state(args.output, 'learning', method=method, layer=layer,
                      pairs=len(valid), extraction_position=position)
                candidate = dict(layer=layer)
                if len(positions) > 1:
                    candidate['extraction_position'] = position
                path = steering_artifact_path(args.output/'artifacts'/method, candidate)
                artifacts[path] = fit(backend, method, layer, data, path, args.sae_weights)
        for candidate in grids[method]:
            path = args.output/f'dev_{method}_{condition_label(candidate)}.jsonl'
            scores.append(evaluate(backend, dev, artifacts[steering_artifact_path(args.output/'artifacts'/method, candidate)], candidate,
                args.max_tokens, path, dict(run=config, stage='dev', method=method,
                candidate=candidate), args.batch_size))
            write_json(args.output/f'{method}_dev.json', scores)
        selected = min(scores, key=lambda r:(-r['correct'], r['candidate']['alpha'], r['candidate']['layer']))
        write_json(args.output/f'{method}_selected.json', selected)
        candidate = selected['candidate']
        for dataset, rows in datasets.items():
            path = args.output/f'{dataset}_{method}.jsonl'
            result = evaluate(backend, rows, artifacts[steering_artifact_path(args.output/'artifacts'/method, candidate)], candidate,
                args.max_tokens, path, dict(run=config, stage=dataset, method=method,
                candidate=candidate), args.batch_size)
            baseline = {r['id']:r for r in read_jsonl(args.output/f'{dataset}_baseline.jsonl')}
            answers = read_jsonl(path)
            result['gains'] = sum(not baseline[r['id']]['correct'] and r['correct'] for r in answers)
            result['losses'] = sum(baseline[r['id']]['correct'] and not r['correct'] for r in answers)
            results[dataset][method] = result
            write_report(args.output, results, args.extraction_position)
    state(args.output, 'completed', report=str(args.output/'report.md'))


def write_report(output, results, extraction_position):
    write_json(output/'results.json', results)
    lines = ['| Dataset | Method | Position | Correct | Accuracy | Missing boxed | Truncated |',
             '|---|---|---|---:|---:|---:|---:|']
    for dataset, conditions in results.items():
        for method, result in conditions.items():
            position = ((result.get('candidate') or {}).get('extraction_position', extraction_position)
                        if method not in ('baseline', 'cot') else '—')
            lines.append(f'| {dataset} | {method} | {position} | {result["correct"]}/{result["total"]} '
                         f'| {result["accuracy"]:.2%} | {result["missing_final_answer"]} | {result["truncated"]} |')
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def search_grids(args, layers, strengths):
    result = {}
    for method in args.methods:
        candidates = grid(method, [args.sae_layer] if method == 'sae' else layers,
                          args.sae_strengths if method == 'sae' else strengths)
        result[method] = expand_extraction_grid(
            candidates[:1] if args.smoke else candidates, args.extraction_position)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run','start','status','dry-run'])
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument('--model-index', type=int, default=0)
    parser.add_argument('--sae-weights', type=Path, help='Gemma Scope JumpReLU params.npz')
    parser.add_argument('--sae-layer', type=int, default=20)
    parser.add_argument('--sae-strengths', nargs='+', type=float,
                        default=[0.1,0.25,0.5,0.75,1,1.5,2,3,5,10,20,40])
    parser.add_argument('--device', default='cuda:2')
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--max-tokens', type=int, default=MAX_TOKENS, help='Budget for dev selection and tests')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/v2/steering_suite')
    add_extraction_argument(parser, allow_both=True)
    args = parser.parse_args()
    if min(args.max_tokens,args.batch_size) <= 0 or len(set(args.methods)) != len(args.methods):
        parser.error('Positive budgets and unique methods required')
    if args.action != 'status':
        import math
        local = open_weight_model(args.model_index)
        if 'caa' in args.methods and 'diffmean' in args.methods:
            parser.error('caa and diffmean are aliases; choose one')
        if 'sae' in args.methods:
            if local['model'] != 'google/gemma-2-9b-it':
                parser.error('SAE currently requires google/gemma-2-9b-it; select --model-index')
            if args.sae_weights is None or not args.sae_weights.is_file():
                parser.error('SAE requires a local --sae-weights params.npz')
            if not 0 <= args.sae_layer < 42 or any(not math.isfinite(a) or a <= 0 for a in args.sae_strengths) or len(set(args.sae_strengths)) != len(args.sae_strengths):
                parser.error('Invalid SAE layer or strengths')
            from reasoning.experiments.sae import validate_weights
            validate_weights(args.sae_weights, args.sae_layer)
        if args.action == 'dry-run':
            layers, strengths = steering_grid(local['model'])
            grids = search_grids(args, layers, strengths)
            print(json.dumps(dict(model=local['model'], extraction_position=args.extraction_position,
                                  baselines=['baseline', 'cot'],
                                  candidates={m: len(g) for m,g in grids.items()}, grid=grids), indent=2))
            return
    dispatch(args, run, 'reasoning.scripts.paper_baselines')


if __name__ == '__main__':
    main()
