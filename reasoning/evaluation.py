"""Answer grading, summary metrics and checkpointed evaluation."""
import time
from reasoning.common import Journal, state, write_json

def boxed(text):
    """Return the final balanced \\boxed{...} content; never guess a last number."""
    start = text.rfind('\\boxed{')
    if start < 0:
        return None
    begin = start + len('\\boxed{')
    depth = 1
    for i in range(begin, len(text)):
        if text[i] == '{' and (i == 0 or text[i - 1] != '\\'):
            depth += 1
        elif text[i] == '}' and (i == 0 or text[i - 1] != '\\'):
            depth -= 1
            if depth == 0:
                return text[begin:i].strip() or None
    return None

def score(text, gold, truncated=False):
    """Grade the final boxed answer independently of the generation stop reason.

    ``truncated`` remains accepted for existing callers; callers persist it as
    generation metadata. Missing/incomplete final answers still earn no credit.
    """
    from math_verify import LatexExtractionConfig, parse, verify
    answer = boxed(text)
    if answer is None:
        return {'correct': False, 'parsed_answer': None, 'scorable': False, 'score_status': 'missing_final_answer'}
    config = [LatexExtractionConfig()]
    target = parse('\\boxed{' + gold + '}', extraction_config=config)
    prediction = parse('\\boxed{' + answer + '}', extraction_config=config)
    scorable = bool(target and prediction)
    correct = bool(scorable and verify(target, prediction))
    return {'correct': correct, 'parsed_answer': answer, 'scorable': scorable, 'score_status': ('correct' if correct else 'incorrect') if scorable else 'unparseable_answer'}


def summarize_full(rows, total):
    correct = sum((r['correct'] for r in rows))
    groups = {}
    for field in ('subject', 'level'):
        groups[field] = {}
        for value in sorted({str(r[field]) for r in rows}):
            members = [r for r in rows if str(r[field]) == value]
            hits = sum((r['correct'] for r in members))
            groups[field][value] = {'count': len(members), 'correct': hits, 'accuracy': hits / len(members)}
    return {'completed': len(rows), 'total': total, 'finished': len(rows) == total, 'correct': correct, 'accuracy': correct / len(rows) if rows else None, 'unscorable': sum((not r['scorable'] for r in rows)), 'truncated': sum((r['truncated'] for r in rows)), 'truncated_correct': sum((r['truncated'] and r['correct'] for r in rows)), 'missing_final_answer': sum((r['parsed_answer'] is None for r in rows)), 'generated_tokens': sum((r['tokens'] for r in rows)), 'generation_seconds': sum((r['seconds'] for r in rows)), 'by': groups}

def summarize_brief(rows, total):
    return dict(completed=len(rows), total=total, correct=sum(r['correct'] for r in rows),
                accuracy=sum(r['correct'] for r in rows)/len(rows) if rows else None,
                truncated=sum(r['truncated'] for r in rows))


def label(candidate):
    if candidate is None:
        return 'baseline'
    prefix = candidate.get('extraction_position', '')
    return (prefix + '_' if prefix else '') + f'L{candidate["layer"]}_a{candidate["alpha"]:g}_{candidate["phase"]}' 


def evaluate(backend, artifact, candidate, rows, budget, stage, config, output, batch_size=4,
             prompt_suffix=None):
    from reasoning.steering import generate
    if prompt_suffix is not None and (candidate is not None or artifact is not None):
        raise ValueError('Prompt baseline must not use a steering artifact or candidate')
    name = label(candidate)
    metadata = dict(run=config,stage=stage,candidate=candidate,budget=budget,
                    ids=[r['id'] for r in rows])
    if prompt_suffix is not None:
        metadata['prompt_suffix'] = prompt_suffix
    journal = Journal(output/f'{stage}_{name}.jsonl',metadata)
    if candidate is None and artifact is None:
        from reasoning.baseline_cache import reuse
        reuse(journal, rows, metadata, budget, prompt_suffix)
    for offset in range(0,len(rows),batch_size):
        batch = [r for r in rows[offset:offset+batch_size] if r['id'] not in journal.ids]
        if not batch:
            continue
        state(output,stage,condition=name,completed=len(journal.rows),total=len(rows))
        start = time.monotonic()
        questions = [r['problem'] for r in batch]
        results = (generate(backend,questions,artifact,candidate,budget)
                   if prompt_suffix is None else
                   backend.generate(questions,max_tokens=budget,prompt_suffix=prompt_suffix))
        if len(results) != len(batch):
            raise ValueError('Generation batch size mismatch')
        elapsed = time.monotonic()-start
        for row,result in zip(batch,results):
            if row['id'] not in journal.ids:
                journal.append({**row,**result,**score(result['completion'],row['answer']),
                                'seconds':elapsed/len(batch),'timing':'amortized_batch'})
        write_json(output/f'{stage}_{name}.summary.json',summarize_brief(journal.rows,len(rows)))
    return dict(label=name,candidate=candidate,**summarize_brief(journal.rows,len(rows)))


