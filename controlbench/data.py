"""Datasets + answer extraction for ControlBench.

Currently: GSM8K (grade-school math). Each example becomes a dict:
    {"question": str, "gold": str}   where gold is the normalized final number.

Answer extraction is deliberately lenient (last number in the text) because
small models rarely follow an exact output format.
"""

import re
from datasets import load_dataset


# --- final-answer parsing ------------------------------------------------

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _normalize_number(s):
    """'1,024.0' -> '1024', '18.00' -> '18'. Returns None if not parseable."""
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        f = float(s)
    except ValueError:
        return None
    # integers print without a trailing .0 so gold/pred compare as strings too
    return str(int(f)) if f == int(f) else str(f)


def extract_gold_answer(answer_field):
    """GSM8K gold answers end with '#### <number>'."""
    if "####" in answer_field:
        answer_field = answer_field.split("####")[-1]
    nums = _NUM.findall(answer_field)
    return _normalize_number(nums[-1]) if nums else None


def extract_pred_answer(text):
    """Pull the model's final numeric answer.

    Prefers a number right after 'answer is'/'answer:'; otherwise the last
    number anywhere in the completion.
    """
    tail = re.split(r"answer\s*(?:is|:)", text, flags=re.IGNORECASE)
    candidate = tail[-1] if len(tail) > 1 else text
    nums = _NUM.findall(candidate)
    if not nums and candidate is not text:
        nums = _NUM.findall(text)
    return _normalize_number(nums[-1]) if nums else None


def is_correct(pred_text, gold):
    return extract_pred_answer(pred_text) == gold and gold is not None


# --- loading -------------------------------------------------------------

def load_gsm8k(split="test", limit=None, seed=0):
    """Return a list of {'question', 'gold'} dicts.

    split: 'train' | 'test'. limit: keep only the first N (after a fixed shuffle
    so subsets are reproducible but not just the dataset's natural order).
    """
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if limit is not None:
        ds = ds.shuffle(seed=seed).select(range(min(limit, len(ds))))
    return [
        {"question": ex["question"], "gold": extract_gold_answer(ex["answer"])}
        for ex in ds
    ]
