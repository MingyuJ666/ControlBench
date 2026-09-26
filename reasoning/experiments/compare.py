"""Checkpointed development-set comparison using repsteer_backend."""
from reasoning.datasets import add_extraction_argument, EXTRACTION_POSITIONS
import argparse
import json
import subprocess
import time
from reasoning.common import SPLIT, PAIRS, MAX_TOKENS, open_weight_model, prepare_run, state, ROOT, Journal, digest, read_jsonl, write_json, code_hashes
from pathlib import Path
from reasoning.evaluation import summarize_brief as summarize, score
from reasoning.datasets import problem_id
from reasoning.datasets.contrastive import prepare_pairs


def choose_winner(scores):
    complete = [s for s in scores if s['completed'] == s['total']]
    if not complete:
        return None
    return max(complete, key=lambda s: (s['correct'], s['method'] == 'baseline', -abs(s['alpha'])))


def run(args):
    import torch
    from reasoning.steering import RepSteerBackend, vector
    import transformers
    torch.set_num_threads(4)
    torch.manual_seed(42)
    output = args.output
    pair_path = prepare_pairs(open_weight_model()['model'], output)
    split_path = SPLIT
    split = json.loads(split_path.read_text())
    all_rows = {problem_id(r['problem']): dict(r, id=problem_id(r['problem']))
                for r in read_jsonl(ROOT / 'data/original.jsonl')}
    rows = [all_rows[i] for i in split['dev'][:args.limit]]
    pair_ids = [r['id'] for r in read_jsonl(pair_path)]
    if not set(pair_ids) <= set(split['extraction']) or set(pair_ids) & {r['id'] for r in rows}:
        raise ValueError('Extraction/development split overlap')
    local = open_weight_model()
    config = dict(methods=args.methods, alphas=args.alphas, layer=args.layer,
        max_tokens=args.max_tokens, batch_size=args.batch_size, dev_ids=[r['id'] for r in rows],
        pair_ids=pair_ids, model=local['model'], revision=local['revision'], device=args.device,
        normalization='l2', pooling=EXTRACTION_POSITIONS[args.extraction_position], extraction_position=args.extraction_position,
        injection='Add on GeneratedTokens, decode only', probe_l2=0.1, probe_max_iter=25,
        lat_center=True, seed=42, torch=torch.__version__, transformers=transformers.__version__,
        repsteer_commit=subprocess.check_output(['git','-C',str(ROOT/'vendor/repsteer'),
                                                'rev-parse','HEAD'],text=True).strip(),
        files={str(p.relative_to(ROOT)): digest(p) for p in
               [pair_path, split_path, ROOT/'data/original.jsonl']},
        code=code_hashes())
    prepare_run(output, config, args.device)
    backend = RepSteerBackend(args.device, args.layer, extraction_position=args.extraction_position)
    data, valid_ids = backend.contrastive_data(pair_path)
    artifacts = {}
    for method in args.methods:
        state(output, 'learning', method=method, pairs=len(valid_ids))
        artifacts[method] = backend.fit(method, data, output / 'artifacts' / method)
    diagnostics = {a: {b: float(torch.dot(vector(artifacts[a]), vector(artifacts[b])))
                       for b in artifacts} for a in artifacts}
    write_json(output / 'vector_diagnostics.json', dict(valid_pairs=len(valid_ids), cosine=diagnostics))
    conditions = [('baseline', 0)] + [(m, a) for m in args.methods for a in args.alphas]
    scores = []
    baseline = {}
    for method, alpha in conditions:
        label = f'{method}_{alpha:g}'
        journal = Journal(output / f'{label}.jsonl', dict(run=config, method=method, alpha=alpha))
        # Fixed batches on resume preserve padding/computation semantics.
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset:offset+args.batch_size]
            if all(r['id'] in journal.ids for r in batch):
                continue
            state(output, 'evaluating', condition=label, completed=len(journal.rows), total=len(rows),
                  completed_conditions=len(scores), total_conditions=len(conditions))
            start = time.monotonic()
            results = backend.generate([r['problem'] for r in batch], artifacts.get(method),
                                       alpha, args.max_tokens)
            seconds = time.monotonic() - start
            for row, result in zip(batch, results):
                if row['id'] in journal.ids:
                    continue
                journal.append({**row, **result, **score(result['completion'], row['answer']),
                                'seconds': seconds/len(batch), 'timing':'amortized_batch'})
            write_json(output / f'{label}.summary.json', summarize(journal.rows,len(rows)))
        summary = dict(method=method, alpha=alpha, **summarize(journal.rows,len(rows)))
        if method == 'baseline':
            baseline = {r['id']:r for r in journal.rows}
        summary.update(gains=sum(not baseline[r['id']]['correct'] and r['correct'] for r in journal.rows),
                       losses=sum(baseline[r['id']]['correct'] and not r['correct'] for r in journal.rows))
        scores.append(summary)
        write_json(output / 'leaderboard.json', scores)
        state(output, 'condition_completed', **summary)
    selected = choose_winner(scores)
    write_json(output / 'selected.json', selected)
    report = dict(scope='20-question development screening; not held-out evaluation',
                  max_tokens=args.max_tokens, selected=selected, conditions=scores,
                  note='All methods share samples, layer, unit norm, decoding and budget. '
                       'Small reused dev set; selection does not establish general improvement.')
    write_json(output / 'report.json', report)
    text = '# repsteer development comparison\n\n'
    text += f'Token budget: {args.max_tokens}; questions: {len(rows)}. Development results only.\n\n'
    text += '| Method | Alpha | Correct | Accuracy | Truncated | Gains | Losses |\n'
    text += '|---|---:|---:|---:|---:|---:|---:|\n'
    for s in scores:
        text += f'| {s["method"]} | {s["alpha"]} | {s["correct"]}/{s["total"]} | {s["accuracy"]:.1%} | {s["truncated"]} | {s["gains"]} | {s["losses"]} |\n'
    text += '\nSelected: ' + json.dumps(selected) + '\n\n' + report['note'] + '\n'
    (output / 'report.md').write_text(text)
    state(output, 'completed', selected=selected, report=str(output/'report.md'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start','run','status'])
    parser.add_argument('--methods', nargs='+', choices=['diffmean','lat','probe'], default=['diffmean','lat','probe'])
    parser.add_argument('--alphas', default='-1,0.5,1.5')
    parser.add_argument('--max-tokens', type=int, default=MAX_TOKENS)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--layer', type=int, default=16)
    parser.add_argument('--device', default='cuda:2')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/v2/repsteer_compare')
    add_extraction_argument(parser)
    args = parser.parse_args()
    args.alphas = [float(a) for a in args.alphas.split(',')]
    import math
    if (not 1 <= args.limit <= 20 or args.max_tokens < 1 or args.batch_size < 1 or
            not args.alphas or any(not math.isfinite(a) or a == 0 for a in args.alphas) or
            len(set(args.alphas)) != len(args.alphas) or len(set(args.methods)) != len(args.methods)):
        parser.error('Invalid limit, budget, batch size, or duplicate/zero/nonfinite candidates')
    from reasoning.common import dispatch
    dispatch(args, run, 'reasoning.experiments.compare')


if __name__ == '__main__':
    main()
