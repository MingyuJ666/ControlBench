"""Evaluation scope and the hard subject-generation budget."""

EVAL_SIZES = {"math500": 500, "amc12": 1199}
DEFAULT_EVAL_SETS = ("math500",)
MAX_NEW_TOKENS = 3072
DEFAULT_BATCH_SIZE = 32


def batches(values, size=DEFAULT_BATCH_SIZE):
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError("batch_size must be a positive integer")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def eval_sets_for(values=None):
    values = list(DEFAULT_EVAL_SETS if values is None else values)
    if not values or set(values) - EVAL_SIZES.keys() or len(set(values)) != len(values):
        raise ValueError("Choose distinct eval sets from math500 and amc12")
    if "math500" not in values:
        raise ValueError("math500 is required; amc12 is an optional additional eval set")
    return [name for name in EVAL_SIZES if name in values]


def token_limit(value=MAX_NEW_TOKENS):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    return min(value, MAX_NEW_TOKENS)
