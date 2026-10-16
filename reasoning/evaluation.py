"""Extract and grade final boxed mathematical answers."""

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
