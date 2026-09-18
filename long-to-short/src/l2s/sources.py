"""Pinned source pool, leakage removal, and deterministic stratification."""
import random
import re
from collections import Counter, defaultdict, deque
from .matching import boxed, normalize_problem

MATH = 'EleutherAI/hendrycks_math'
MATH500 = 'HuggingFaceH4/MATH-500'
AMC = 'edev2000/amc12-full'
TOKENIZER = 'meta-llama/Llama-3.1-8B-Instruct'
SUBJECTS = ('algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra',
            'number_theory', 'prealgebra', 'precalculus')


def stratify(items):
    rng = random.Random(0)
    buckets = defaultdict(list)
    for item in items:
        if item['level'] not in (2, 3, 4):
            raise ValueError('Generation queue supports levels 2, 3, 4 only')
        buckets[item['subject'], item['level']].append(item)
    for key in sorted(buckets):
        bucket = buckets[key]
        bucket.sort(key=lambda item: item['math_id'])
        rng.shuffle(bucket)
    buckets = {key: deque(value) for key, value in buckets.items()}
    subjects = sorted({subject for subject, _ in buckets})
    queue = []
    # One item per subject each round; cycle levels 2, 3, 4 within subject.
    round_index = 0
    while any(buckets.values()):
        for subject_index, subject in enumerate(subjects):
            levels = [2 + (round_index + subject_index + offset) % 3 for offset in range(3)]
            for level in levels:
                if buckets.get((subject, level)):
                    queue.append(buckets[subject, level].popleft())
                    break
        round_index += 1
    return queue


def load_sources(revisions):
    from datasets import load_dataset
    math500 = load_dataset(MATH500, split='test', revision=revisions[MATH500])
    amc = load_dataset(AMC, split='train', revision=revisions[AMC])
    if len(math500) != 500 or len(amc) != 1199:
        raise RuntimeError('Evaluation corpus sizes differ from the frozen experiment plan')
    eval_ids = {row['unique_id'] for row in math500}
    if not all(value.startswith('test/') for value in eval_ids):
        raise RuntimeError('Unexpected MATH-500 unique_id namespace')
    forbidden = {normalize_problem(row['problem']) for row in math500}
    forbidden.update(normalize_problem(row['question']) for row in amc)
    counts, items, seen = Counter(), [], set()
    for subject in SUBJECTS:
        train = load_dataset(MATH, subject, split='train', revision=revisions[MATH])
        for index, row in enumerate(train):
            counts['source_rows'] += 1
            gold, _ = boxed(row['solution'])
            problem_id = row.get('id') or row.get('unique_id') or f'train/{subject}/row-{index:06d}'
            if not problem_id.startswith('train/') or problem_id in eval_ids:
                raise RuntimeError('Non-training problem ID in source pool')
            if not gold:
                counts['empty_gold'] += 1
                continue
            normalized = normalize_problem(row['problem'])
            if normalized in forbidden:
                counts['leakage_removed'] += 1
                continue
            if normalized in seen:
                counts['duplicates_removed'] += 1
                continue
            seen.add(normalized)
            level_match = re.fullmatch(r'(?:Level\s+)?([1-5])', str(row['level']))
            if not level_match:
                counts['missing_or_invalid_level'] += 1
                continue
            level = int(level_match[1])
            if level not in (2, 3, 4):
                counts['outside_levels_2_4'] += 1
                continue
            # Only the boxed answer survives: never serialize the gold writeup.
            items.append(dict(source='hendrycks_math_train', math_id=problem_id,
                              subject=row['type'], level=level, problem=row['problem'], gold_answer=gold))
    counts['eligible'] = len(items)
    return stratify(items), dict(counts)


def resolve_revisions():
    from huggingface_hub import HfApi
    api = HfApi()
    revisions = {repo: api.dataset_info(repo).sha for repo in (MATH, MATH500, AMC)}
    revisions[TOKENIZER] = api.model_info(TOKENIZER).sha
    return revisions
