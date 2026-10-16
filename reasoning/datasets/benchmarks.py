"""Pinned competition datasets and a strict MATH500 split audit (no model imports)."""
import argparse
import io
import json
import re
import urllib.request
from pathlib import Path
from reasoning.common import SPLIT, digest, read_jsonl, write_json, ROOT
from reasoning.datasets import load, problem_id


SOURCES = {
    'amc2023': dict(repo='AI-MO/aimo-validation-amc',
                    revision='69d78a4a2c840e82d69af6bc742bda09005f6316',
                    pattern=r'/2023_AMC_12[AB]_Problems/Problem_\d+$', count=40),
    'aime2024': dict(repo='AI-MO/aimo-validation-aime',
                     revision='13f9e12f613e720c2a2b2f345dd04b998a29494d',
                     pattern=r'/2024_AIME_I{1,2}_Problems/Problem_\d+$', count=30),
}
DATA = ROOT / 'data/benchmarks'


def normalize(name, raw):
    spec = SOURCES[name]
    rows = []
    for row in raw:
        if not re.search(spec['pattern'], row.get('url', '')):
            continue
        raw_answer = row['answer']
        if isinstance(raw_answer, float) and raw_answer.is_integer():
            raw_answer = int(raw_answer)
        answer = str(raw_answer).strip().replace(',', '')
        if not re.fullmatch(r'-?\d+', answer):
            raise ValueError(f'Noninteger reference answer in {name}')
        problem = row['problem'].strip()
        if not problem:
            raise ValueError('Empty problem')
        rows.append(dict(id=problem_id(problem), problem=problem,
                         answer=str(int(answer)), source_id=str(row['id']),
                         url=row['url'], dataset=name))
    rows.sort(key=lambda r: (r['url'].rsplit('/', 1)[0],
                             int(r['url'].rsplit('_', 1)[1])))
    if len(rows) != spec['count']:
        raise ValueError(f'{name}: expected {spec["count"]} questions, got {len(rows)}')
    if len({r['id'] for r in rows}) != len(rows) or len({r['url'] for r in rows}) != len(rows):
        raise ValueError(f'Duplicate questions in {name}')
    return rows


def prepare(directory=DATA):
    import pyarrow.parquet as pq
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / 'manifest.json'
    if manifest_path.exists():
        datasets = load_external(directory)
        return {name: len(rows) for name, rows in datasets.items()}
    manifest = {}
    for name, spec in SOURCES.items():
        filename = 'data/train-00000-of-00001.parquet'
        url = f'https://huggingface.co/datasets/{spec["repo"]}/resolve/{spec["revision"]}/{filename}'
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        rows = normalize(name, pq.read_table(io.BytesIO(content)).to_pylist())
        raw_path = directory / f'{name}.source.parquet'
        path = directory / f'{name}.jsonl'
        encoded = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
        # Never silently replace an existing benchmark, including interrupted preparation.
        for target, payload in [(raw_path, content), (path, encoded.encode())]:
            if target.exists() and target.read_bytes() != payload:
                raise ValueError(f'Existing data differs: {target}; choose a new directory')
            if not target.exists():
                target.write_bytes(payload)
        manifest[name] = dict(**spec, source_url=url, source_sha256=digest(raw_path),
                              sha256=digest(path), rows=len(rows),
                              role='test_only', upstream_split='train')
    write_json(manifest_path, manifest)
    return {name: item['rows'] for name, item in manifest.items()}


def load_external(directory=DATA):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    result = {}
    for name, spec in SOURCES.items():
        meta = manifest[name]
        if any(meta.get(key) != value for key, value in spec.items()):
            raise ValueError(f'Source version changed: {name}')
        path = directory / f'{name}.jsonl'
        if digest(path) != meta['sha256']:
            raise ValueError(f'Benchmark hash mismatch: {name}')
        rows = read_jsonl(path)
        if len(rows) != spec['count'] or len({r['id'] for r in rows}) != len(rows):
            raise ValueError(f'Incomplete or duplicate benchmark: {name}')
        if any(r['id'] != problem_id(r['problem']) or
               not re.search(spec['pattern'], r['url']) for r in rows):
            raise ValueError(f'Invalid benchmark IDs/year: {name}')
        result[name] = rows
    return result


def audit_split(split, math_rows, pairs):
    expected = {k: len(split[k]) for k in ('extraction', 'dev', 'eval')}
    if tuple(expected.values()) not in ((200, 20, 280), (160, 60, 280)):
        raise ValueError('Unsupported MATH500 split sizes')
    groups = {}
    for name, count in expected.items():
        ids = split[name]
        if len(ids) != count or len(set(ids)) != count:
            raise ValueError(f'Invalid {name} split')
        groups[name] = set(ids)
    if any(groups[a] & groups[b] for a, b in
           [('extraction', 'dev'), ('extraction', 'eval'), ('dev', 'eval')]):
        raise ValueError('MATH500 split overlap')
    if len(math_rows) != 500 or set.union(*groups.values()) != {r['id'] for r in math_rows}:
        raise ValueError('MATH500 split does not cover exactly 500 unique questions')
    pair_ids = [r['id'] for r in pairs]
    if len(set(pair_ids)) != len(pair_ids) or not set(pair_ids) <= groups['extraction']:
        raise ValueError('Extraction pair leakage or duplicates')
    if any(r['id'] != problem_id(r['problem']) for r in pairs):
        raise ValueError('Pair problem/ID mismatch')
    if set(pair_ids) != groups['extraction']:
        raise ValueError('Pairs must cover the complete extraction split')
    lookup = {r['id']: r for r in math_rows}
    if any(r['problem'] != lookup[r['id']]['problem'] for r in pairs):
        raise ValueError('Pair problem differs from original data')
    return dict(split_sizes=expected, extraction_pairs=len(pairs),
                extraction_eval_overlap=len(groups['extraction'] & groups['eval']),
                dev_eval_overlap=len(groups['dev'] & groups['eval']),
                math500_test_note='Repeated evaluation: these 280 questions have been evaluated before.',
                overlap_check='Whitespace-normalized exact problem hashes; not semantic deduplication.')


def load_suite(directory=DATA, *, split_path=SPLIT, pairs_path):
    split = json.loads(split_path.read_text())
    math_rows = load()
    audit = audit_split(split, math_rows, read_jsonl(pairs_path))
    lookup = {r['id']: r for r in math_rows}
    datasets = load_external(directory)
    seen = set(lookup)
    for name, rows in datasets.items():
        overlap = seen & {r['id'] for r in rows}
        if overlap:
            raise ValueError(f'Cross-dataset exact problem overlap: {name}: {len(overlap)}')
        seen.update(r['id'] for r in rows)
    datasets['math500'] = [dict(lookup[i], dataset='math500') for i in split['eval']]
    return datasets, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DATA)
    args = parser.parse_args()
    print(json.dumps(prepare(args.directory), indent=2))


if __name__ == '__main__':
    main()
