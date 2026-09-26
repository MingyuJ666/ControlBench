"""Reuse compatible ordinary and CoT completions across run directories."""
import json
from pathlib import Path
from reasoning.common import ROOT, SYSTEM, SUFFIX, digest, read_jsonl
from reasoning.datasets import problem_id


def signature(meta, budget=None, suffix=None):
    run = meta.get('run', meta)
    if meta.get('candidate') is not None or ('run' in meta and 'candidate' not in meta):
        raise ValueError('Not an identified unsteered condition')
    if 'cot' in meta.get('stage', '') and 'prompt_suffix' not in meta:
        raise ValueError('Missing CoT prompt')
    prompts = run.get('prompts', {})
    system = prompts.get('system', run.get('system'))
    ordinary = prompts.get('baseline', run.get('suffix'))
    if system is None or ordinary is None:
        if run.get('code', {}).get('models.py') != digest(ROOT/'models.py'):
            raise ValueError('Legacy prompt cannot be verified')
        system, ordinary = SYSTEM, SUFFIX
    if run.get('do_sample', False) or run.get('temperature', 0) != 0:
        raise ValueError('Only greedy reuse is supported')
    stage_budget = run.get('dev_tokens') if meta.get('stage') == 'dev' else run.get('max_tokens')
    values = dict(model=run.get('model'), revision=run.get('revision'),
        budget=budget if budget is not None else meta.get('budget', stage_budget),
        system=system, suffix=suffix if suffix is not None else meta.get('prompt_suffix', ordinary),
        backend=run.get('backend', 'transformers' if 'run' in meta else None),
        batch_size=run.get('batch_size'), enable_thinking=run.get('enable_thinking'), seed=run.get('seed'),
        system_in_user=run.get('system_in_user', False),
        output_budget_policy=run.get('output_budget_policy', 'fixed'))
    if any(values[k] is None for k in ('model','revision','budget','backend','batch_size','seed')):
        raise ValueError('Incomplete generation settings')
    return values


def sources():
    yield from sorted((ROOT/'data').glob('negative_*.jsonl'))
    for base in (ROOT/'outputs', ROOT/'outputs/v2'):
        yield from sorted(base.glob('*baseline*.jsonl'))
        for directory in sorted(base.glob('*')):
            if directory.is_dir():
                yield from sorted(directory.glob('*baseline*.jsonl'))


def source_metadata(path, rows):
    meta = path.with_suffix(path.suffix+'.meta.json')
    if meta.exists():
        return json.loads(meta.read_text()), meta
    if path.resolve() == (ROOT/'data/negative_llama.jsonl').resolve():
        original = ROOT/'outputs/llama31_8b_vllm_math500_8192.jsonl'
        originals = read_jsonl(original)
        indexed = {r['id']:r for r in rows}
        if len(indexed) != len(originals) or any(
                any(indexed.get(r['id'], {}).get(k) != v for k,v in r.items()) for r in originals):
            raise ValueError('Legacy export differs from original baseline')
        meta = original.with_suffix('.jsonl.meta.json')
        return json.loads(meta.read_text()), meta
    raise ValueError('Missing source metadata')


def reuse(journal, rows, metadata, budget, suffix=None):
    pending = {r['id']:r for r in rows if r['id'] not in journal.ids}
    if not pending:
        return 0
    try:
        expected = signature(metadata, budget, suffix)
    except ValueError:
        return 0
    count = 0
    for path in sources():
        if not pending:
            break
        if path.resolve() == journal.path.resolve() or not path.is_file():
            continue
        try:
            cached = read_jsonl(path)
            meta, meta_path = source_metadata(path, cached)
            actual = signature(meta)
            if actual != expected:
                continue
            ids = [r['id'] for r in cached]
            if len(ids) != len(set(ids)):
                continue
            selected = []
            for row in cached:
                target = pending.get(row['id'])
                if target is None:
                    continue
                if (problem_id(row['problem']) != row['id'] or row['problem'] != target['problem'] or
                        row['answer'] != target['answer'] or not isinstance(row['completion'], str) or
                        type(row['tokens']) is not int or not 0 <= row['tokens'] <= budget or
                        type(row['truncated']) is not bool):
                    raise ValueError('Invalid source question or generation fields')
                selected.append((row, target))
        except (ValueError, KeyError, OSError, TypeError):
            continue
        if not selected:
            continue
        from reasoning.evaluation import score
        provenance = dict(path=str(path.resolve()), sha256=digest(path),
                          metadata_sha256=digest(meta_path), settings=actual)
        for row, target in selected:
            journal.append({**target, **{k:row[k] for k in ('completion','tokens','truncated')},
                **score(row['completion'], target['answer']),
                'seconds':0.0, 'timing':'reused', 'baseline_source':provenance})
            del pending[row['id']]
            count += 1
    if count:
        print(f'Reused {count} baseline responses for {journal.path.name}; missing {len(pending)}', flush=True)
    return count
