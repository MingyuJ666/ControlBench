"""Datasets + answer extraction for ControlBench.

Currently: GSM8K (grade-school math). Each example becomes a dict:
    {"question": str, "gold": str}   where gold is the normalized final number.

Answer extraction is deliberately lenient (last number in the text) because
small models rarely follow an exact output format.
"""

import math
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

# --- long2short cot -------------------------------------------------------------

_MIXCHAIN_LONG_SOLUTION_IDS = (1, 2, 3, 4)


def _count_tokens(text):
    return len(text.split())


def _true_mixchain_solution_candidates(ex):
    candidates = []
    for solution_id in _MIXCHAIN_LONG_SOLUTION_IDS:
        if not ex.get(f"correct_{solution_id}", True):
            continue
        response = ex[f"solution_{solution_id}"]
        token_count = ex.get(f"solution_{solution_id}_token")
        if token_count is None:
            token_count = _count_tokens(response)
        candidates.append((token_count, response))

    return candidates


def _longest_true_mixchain_solution(ex):
    candidates = _true_mixchain_solution_candidates(ex)
    return (
        max(candidates, key=lambda item: item[0])[1]
        if candidates
        else ex["solution_0"]
    )


def _longest_true_mixchain_solution_token_count(ex):
    candidates = _true_mixchain_solution_candidates(ex)
    return (
        max(candidates, key=lambda item: item[0])[0]
        if candidates
        else _count_tokens(ex["solution_0"])
    )


def _long_short_token_ratio(ex):
    short_token_count = _count_tokens(ex["solution_0"])
    if short_token_count == 0:
        return math.inf
    return _longest_true_mixchain_solution_token_count(ex) / short_token_count


def _select_long_short_token_ratio_percentile(ds, percentile):
    if percentile is None:
        return ds
    if not 0 <= percentile <= 100:
        raise ValueError("long_short_token_ratio_percentile must be between 0 and 100")
    if percentile == 0 or len(ds) == 0:
        return ds

    keep_count = max(1, math.ceil(len(ds) * (100 - percentile) / 100))
    ranked = sorted(
        ((idx, _long_short_token_ratio(ex)) for idx, ex in enumerate(ds)),
        key=lambda item: (-item[1], item[0]),
    )
    keep_indices = sorted(idx for idx, _ in ranked[:keep_count])
    return ds.select(keep_indices)


def _format_mixchain_long2short_cot(ex):
    return {
        "question": ex["question"],
        "response_true_short": ex["solution_0"],
        "response_true_long": _longest_true_mixchain_solution(ex),
    }


def _load_mixchain_long2short_cot(
    dataset_name,
    split="train",
    limit=None,
    seed=0,
    long_short_token_ratio_percentile=50,
):
    """Return a Dataset with question, response_true_short, response_true_long.

    long_short_token_ratio_percentile=50 keeps the top half by long/short
    token ratio. Use None to keep all rows.
    """
    ds = load_dataset(dataset_name, split=split)
    ds = _select_long_short_token_ratio_percentile(
        ds, long_short_token_ratio_percentile
    )
    if limit is not None:
        ds = ds.shuffle(seed=seed).select(range(min(limit, len(ds))))
    return ds.map(_format_mixchain_long2short_cot, remove_columns=ds.column_names)


def load_mixchain_z_gsm8k_long2short_cot(
    split="train", limit=None, seed=0, long_short_token_ratio_percentile=50
):
    return _load_mixchain_long2short_cot(
        "horseee/MixChain-Z-GSM8K",
        split=split,
        limit=limit,
        seed=seed,
        long_short_token_ratio_percentile=long_short_token_ratio_percentile,
    )


def load_mixchain_z_prm12k_long2short_cot(
    split="train", limit=None, seed=0, long_short_token_ratio_percentile=50
):
    return _load_mixchain_long2short_cot(
        "horseee/MixChain-Z-PRM12K",
        split=split,
        limit=limit,
        seed=seed,
        long_short_token_ratio_percentile=long_short_token_ratio_percentile,
    )
