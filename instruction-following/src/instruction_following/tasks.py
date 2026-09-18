"""The five instruction-following tasks: instruction wording and rule check.

Each task is one instruction appended to a carrier prompt and one rule that
decides whether an answer has the required surface form. The judge decides
the rest (coherent, answer_status); ``score`` records the rule facts.
"""

import pysbd

TASKS = {
    "lowercase": {
        "instruction": "Your entire response must be in lowercase letters only.",
        "verifier": "verify_lowercase",
    },
    "fixed_ending": {
        "instruction": 'End your response with the exact sentence: "Hope this helps!"',
        "verifier": "verify_fixed_ending",
    },
    "question_ending": {
        "instruction": "The last sentence of your response must be a question.",
        "verifier": "verify_ends_with_question",
    },
    "one_sentence": {
        "instruction": "Answer in exactly one sentence.",
        "verifier": "verify_single_sentence",
    },
    "no_comma": {
        "instruction": "Do not use any commas in your response.",
        "verifier": "verify_no_comma",
    },
}


def make_prompted_prompt(base_prompt: str, task: str) -> str:
    """Append a task's instruction to a stripped base prompt."""
    return f"{base_prompt.strip()} {TASKS[task]['instruction']}"


def make_pair(base_prompt: str, task: str) -> tuple[str, str]:
    """(prompted, unprompted): the two differ by exactly the appended instruction."""
    base = base_prompt.strip()
    return make_prompted_prompt(base, task), base


# ---- rule verifiers -----------------------------------------------------------

# Rendering noise the ending tasks tolerate: markdown emphasis, quotes, brackets.
# Punctuation is not stripped: each ending task carries its own required mark.
WRAPPER_CHARS = " \t\n\r*_`~#" + "\"'“”‘’" + ")]}）】"
TRAILING_MARKS = "!.！。"
_SEGMENTER = pysbd.Segmenter(language="en", clean=False)
_FIXED_SENTENCE = "Hope this helps!"


def _tail(output: str) -> str:
    return output.strip().rstrip(WRAPPER_CHARS)


def verify_lowercase(output: str) -> bool:
    """Nonempty text with no uppercase or titlecase characters."""
    return bool(output.strip()) and not any(c.isupper() or c.istitle() for c in output)


def verify_fixed_ending(output: str) -> bool:
    """Ends with the exact sentence "Hope this helps!" (markdown/quotes around it tolerated)."""
    sentences = [s.strip(WRAPPER_CHARS) for s in _SEGMENTER.segment(_tail(output)) if s.strip(WRAPPER_CHARS)]
    return bool(sentences) and sentences[-1] == _FIXED_SENTENCE


def verify_ends_with_question(output: str) -> bool:
    return _tail(output).rstrip(TRAILING_MARKS).endswith(("?", "？"))


def verify_single_sentence(output: str) -> bool:
    s = output.strip()
    if not s:
        return False
    return len([x for x in _SEGMENTER.segment(s) if x.strip()]) == 1


def verify_no_comma(output: str) -> bool:
    return "," not in output and "，" not in output and "、" not in output


VERIFIERS = {
    "verify_lowercase": verify_lowercase,
    "verify_fixed_ending": verify_fixed_ending,
    "verify_ends_with_question": verify_ends_with_question,
    "verify_single_sentence": verify_single_sentence,
    "verify_no_comma": verify_no_comma,
}


def score(output: str, verifier: str, truncated: bool = False) -> dict:
    """Check the visible output; record truncation independently of the rule."""
    if not output.strip():
        return {"rule_pass": False, "truncated": truncated}
    return {"rule_pass": VERIFIERS[verifier](output), "truncated": truncated}
