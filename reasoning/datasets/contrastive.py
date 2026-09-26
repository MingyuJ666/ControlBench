"""Model-specific contrastive pairs from the complete fixed extraction split."""
import json
from pathlib import Path
from reasoning.common import DATA, SPLIT, digest, read_jsonl, write_json
from reasoning.datasets import problem_id


def negative_path(model, directory=DATA):
    families = {'meta-llama/Llama-3.1-8B-Instruct': 'llama',
                'google/gemma-2-9b-it': 'gemma', 'Qwen/Qwen3-8B': 'qwen'}
    if model not in families:
        raise ValueError(f'No negative dataset configured for model: {model}')
    return Path(directory)/f'negative_{families[model]}.jsonl'


def pair_records(positive, negative, originals, split, model):
    lookup = {problem_id(r['problem']): r for r in originals}
    if len(lookup) != len(originals):
        raise ValueError('Duplicate original problems')
    names = ('extraction', 'dev', 'eval')
    groups = [set(split[k]) for k in names]
    if any(len(g) != len(split[k]) for g, k in zip(groups, names)):
        raise ValueError('Duplicate split IDs')
    if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('Extraction/development/evaluation overlap')
    if set.union(*groups) != set(lookup):
        raise ValueError('Split does not cover original data')
    sources = []
    for name, rows, field in [('positive', positive, 'solution'), ('negative', negative, 'completion')]:
        indexed = {r['id']: r for r in rows}
        if len(indexed) != len(rows) or set(indexed) != set(lookup):
            raise ValueError(f'{name} dataset incomplete or duplicate; wait for all 500 questions')
        for ident, row in indexed.items():
            if problem_id(row['problem']) != ident or not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f'Invalid {name} text or problem ID: {ident}')
            if name == 'negative' and row.get('model') != model:
                raise ValueError(f'Negative model mismatch: expected {model}')
        sources.append(indexed)
    pos, neg = sources
    return [dict(id=i, problem=lookup[i]['problem'], teacher=pos[i]['solution'],
                 negative=neg[i]['completion'], negative_model=model) for i in split['extraction']]


def prepare_pairs(model, output):
    positive = DATA/'positive.jsonl'
    negative = negative_path(model)
    if not negative.exists():
        raise ValueError(f'Negative dataset not ready: {negative}; wait for generation to finish')
    originals = read_jsonl(DATA/'original.jsonl')
    split = json.loads(SPLIT.read_text())
    if len(originals) != 500 or tuple(len(split[k]) for k in ('extraction', 'dev', 'eval')) not in ((200, 20, 280), (160, 60, 280)):
        raise ValueError('Expected MATH500 200/20/280 or 160/60/280 split')
    records = pair_records(read_jsonl(positive), read_jsonl(negative), originals, split, model)
    config = dict(model=model, selection=f'all {len(records)} fixed extraction IDs',
                  labels='teacher/model source; no correctness filtering',
                  ids=split['extraction'], sources={str(p): digest(p) for p in
                      [positive, negative, DATA/'original.jsonl', SPLIT]})
    path = Path(output)/'pairs.jsonl'
    meta = path.with_suffix('.jsonl.meta.json')
    content = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records)
    if path.exists() or meta.exists():
        if not path.exists() or not meta.exists() or json.loads(meta.read_text()) != config or path.read_text() != content:
            raise ValueError('Pair sources changed or snapshot incomplete; use a new output directory')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.jsonl.tmp')
        temporary.write_text(content)
        temporary.replace(path)
        write_json(meta, config)
    return path
