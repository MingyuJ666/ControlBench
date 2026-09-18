"""Conservative MATH exact normalization shared by construction and future eval.

No model-generated code is evaluated. Nonmatching answers are rejected during
construction; the optional evaluation equivalence judge belongs to a later stage.
"""
import re
import unicodedata
from collections import Counter
from fractions import Fraction

MATCHER_VERSION = 'l2s-math-exact-1'
FINAL = re.compile(r'^\s*(?:\*\*)?Final answer:(?:\*\*)?\s*(.+?)\s*$', re.M | re.I)


def boxed(text):
    """Return the last balanced box and its start, including nested braces."""
    matches = list(re.finditer(r'\\(?:boxed|fbox)\s*\{', text))
    if not matches:
        return '', len(text)
    match = matches[-1]
    depth = 1
    for end in range(match.end(), len(text)):
        if text[end] == '{':
            depth += 1
        elif text[end] == '}':
            depth -= 1
            if depth == 0:
                return text[match.end():end].strip(), match.start()
    return '', len(text)


def split_trace(text):
    matches = list(FINAL.finditer(text))
    if matches:
        match = matches[-1]
        # Output after the final line is not part of the required trace form.
        if text[match.end():].strip():
            return text, ''
        return text[:match.start()].strip(), match.group(1).strip()
    answer, start = boxed(text)
    return text[:start].strip(), answer


def normalize_answer(text):
    text = unicodedata.normalize('NFKC', str(text)).strip()
    if not text:
        return ''
    inner, _ = boxed(text)
    if inner:
        text = inner
    text = text.replace('−', '-').replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    for old in ('\\left', '\\right', '\\!', '\\,', '\\;', '\\:', '\\$', '$', '\\(', '\\)', '\\[', '\\]'):
        text = text.replace(old, '')
    text = text.replace('^{\\circ}', '').replace('^\\circ', '').replace('\\%', '%')
    text = re.sub(r'\\(?:text|mathrm)\{([^{}]*)\}', r'\1', text)
    text = re.sub(r'\\sqrt(\w)(?![\w{])', r'\\sqrt{\1}', text)
    text = re.sub(r'\\frac\s*([0-9])\s*([0-9])', r'\\frac{\1}{\2}', text)
    text = re.sub(r'\\frac\s*([0-9])\s*\{([^{}]+)\}', r'\\frac{\1}{\2}', text)
    text = re.sub(r'\s+', '', text).rstrip('.')
    # The usual MATH convention permits a single variable assignment.
    if re.fullmatch(r'[a-zA-Z]=[^=]+', text):
        text = text.split('=', 1)[1]
    if re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?', text):
        text = text.replace(',', '')
    rational = re.fullmatch(r'([+-]?)\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}', text)
    try:
        if rational:
            return str(Fraction(int(rational[2]), int(rational[3])) * (-1 if rational[1] == '-' else 1))
        if re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?', text):
            return str(Fraction(text))
    except (ValueError, ZeroDivisionError):
        pass
    return text


def answers_match(prediction, gold):
    prediction, gold = normalize_answer(prediction), normalize_answer(gold)
    return bool(prediction and gold and prediction == gold)


def normalize_problem(text):
    """Strip common TeX/plain AMC choice blocks before exact leakage matching."""
    text = unicodedata.normalize('NFKC', text)
    patterns = [r'\\(?:text|mathrm|textbf)\s*\{\s*\(?A\)?[.:]?\s*\}',
                r'\(\s*A\s*\)', r'(?:^|\n)\s*A[.)]\s']
    starts = [m.start() for pattern in patterns if (m := re.search(pattern, text))]
    if starts:
        text = text[:min(starts)]
    text = text.replace('\\left', '').replace('\\right', '').replace('$', '')
    return re.sub(r'\s+', '', text).casefold()


def repetition_collapsed(tokens):
    return any(n >= 3 for n in Counter(tuple(tokens[i:i+50]) for i in range(len(tokens)-49)).values())


def normalize_equation(text):
    text = unicodedata.normalize('NFKC', str(text)).replace('−', '-')
    text = text.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    text = re.sub(r'\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}',
                  lambda m: normalize_answer(m[0]), text)
    text = re.sub(r'(?<![\w.])[+-]?(?:\d+\.\d+|\.\d+|\d+/[1-9]\d*)(?![\w.])',
                  lambda m: normalize_answer(m[0]), text)
    for old in ('\\left', '\\right', '\\,', '\\!', '$', '\\(', '\\)', '\\[', '\\]'):
        text = text.replace(old, '')
    return re.sub(r'\s+', ' ', text).strip()


def step_matches(step, reasoning):
    haystack = normalize_equation(reasoning)
    for field in ('equation', 'value'):
        if step.get(field) is None:
            continue
        needle = normalize_equation(step[field])
        pattern = r'\s*'.join(re.escape(char) for char in needle if not char.isspace())
        if pattern and re.search(r'(?<![\w.])' + pattern + r'(?!\w|\.\d)', haystack):
            return True
    return False
