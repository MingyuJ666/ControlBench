"""G1/F1 -> G2/F2 -> G3/F3; dependency-injected for offline checks."""
import json
from . import prompts
from .matching import answers_match, repetition_collapsed, split_trace, step_matches

TARGET_PAIRS = 64
MIN_LONG_TOKENS = 80
MIN_SHORT_TOKENS = 32
MAX_SKELETON_STEPS = 12


def parse_skeleton(text, gold):
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict) or not isinstance(value.get('final_answer'), str):
        return None
    if not answers_match(value['final_answer'], gold):
        return None
    steps = value.get('key_steps')
    if not isinstance(steps, list) or not 1 <= len(steps) <= MAX_SKELETON_STEPS:
        return None
    for index, step in enumerate(steps, 1):
        if not isinstance(step, dict) or type(step.get('id')) is not int or step['id'] != index:
            return None
        if not isinstance(step.get('claim'), str) or not step['claim'].strip():
            return None
        if any(k not in step or (step[k] is not None and not isinstance(step[k], (str, int, float)))
               for k in ('equation', 'value')):
            return None
    return value


def inspect_trace(reply, gold, encode, minimum):
    reasoning, answer = split_trace(reply['text'])
    tokens = encode(reasoning)
    reasons = []
    if reply['finish_reason'] != 'stop':
        reasons.append('unfinished')
    if not answers_match(answer, gold):
        reasons.append('answer')
    if len(tokens) < minimum:
        reasons.append('direct')
    if repetition_collapsed(tokens):
        reasons.append('repetition')
    return reasoning, len(tokens), reasons


def craft_pair(item, teacher, encode):
    """Return (accepted record or None, filter audit); never sees gold solution."""
    context = {k: item[k] for k in ('problem', 'gold_answer')}
    audit = []
    for attempt, temperature in enumerate((0.2, 0.4)):
        reply = teacher('g1', attempt, prompts.G1.format(**context), temperature)
        _, long_n, errors = inspect_trace(reply, item['gold_answer'], encode, MIN_LONG_TOKENS)
        audit.append(dict(stage='f1', attempt=attempt, errors=errors))
        if not errors:
            context['long_trace'] = reply['text']
            break
    else:
        return None, audit
    for attempt in range(2):
        reply = teacher('g2', attempt, prompts.G2.format(**context), 0.0)
        skeleton = parse_skeleton(reply['text'], item['gold_answer']) if reply['finish_reason'] == 'stop' else None
        audit.append(dict(stage='f2', attempt=attempt, valid=skeleton is not None))
        if skeleton is not None:
            context['skeleton'] = json.dumps(skeleton, ensure_ascii=False)
            break
    else:
        return None, audit
    for attempt in range(2):
        reply = teacher('g3', attempt, prompts.G3.format(**context), 0.0)
        reasoning, short_n, errors = inspect_trace(reply, item['gold_answer'], encode, MIN_SHORT_TOKENS)
        ratio = long_n / short_n if short_n else 0
        if not 2 <= ratio <= 5:
            errors.append('length_band')
        coverage = 0.0
        matched = []
        if not errors:
            for step in skeleton['key_steps']:
                covered = step_matches(step, reasoning)
                if not covered:
                    answer = teacher(f'coverage-{attempt}-{step["id"]}', 0,
                                     prompts.COVERAGE.format(claim=step['claim'], reasoning=reasoning), 0.0)
                    covered = answer['finish_reason'] == 'stop' and answer['text'].strip().lower() == 'yes'
                matched.append(bool(covered))
            coverage = sum(matched) / len(matched)
            if coverage < 0.8:
                errors.append('coverage')
        audit.append(dict(stage='f3', attempt=attempt, errors=errors, coverage=coverage, matched=matched))
        if not errors:
            return dict(item, long_trace=context['long_trace'], short_trace=reply['text'],
                        long_n_tokens=long_n, short_n_tokens=short_n, length_ratio=ratio,
                        skeleton=skeleton['key_steps'], coverage=coverage, split='extraction',
                        teacher=dict(model='gpt-5.1')), audit
    return None, audit
