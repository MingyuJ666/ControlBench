"""Rule-based checks for surface-form instruction compliance.

Each verifier answers only whether an output has the required surface
property. ``score`` records that rule result and truncation status; the judge
pipeline later combines these facts with its separate quality verdict.
"""

import pysbd

# Rendering noise the ending tasks tolerate: markdown emphasis, straight and
# curly quotes, brackets. Models routinely close with "**Hope this helps!**"
# or `code`, and that is a formatting choice rather than a content failure.
# Punctuation is NOT here: each ending task carries its own required mark, so
# stripping punctuation globally would erase the very thing being checked.
WRAPPER_CHARS = " \t\n\r*_`~#" + "\"'“”‘’" + ")]}）】"

# Marks Task 3 accepts after its question mark: "Is that right?!" still ends
# on a question. Task 2 has no equivalent, since its sentence ends in '!'.
TRAILING_MARKS = "!.！。"

_SEGMENTER = pysbd.Segmenter(language="en", clean=False)
_FIXED_SENTENCE = "Hope this helps!"


def _tail(output: str) -> str:
    return output.strip().rstrip(WRAPPER_CHARS)


def verify_lowercase(output: str) -> bool:
    """Task 1 — no uppercase letter appears.

    Only CASED characters count. A Han character passes isalpha() but is
    neither upper nor lower, so testing `all(c.islower())` would fail an
    otherwise-perfect answer the moment it quotes one — the same
    compliant-answer-scored-as-failure bug class as v0.4's missing '!'.
    Requiring at least one cased letter keeps a digits-only string from
    passing vacuously."""
    cased = [c for c in output if c.isupper() or c.islower()]
    return len(cased) > 0 and not any(c.isupper() for c in cased)


def verify_fixed_ending(output: str) -> bool:
    """Task 2 — ends with the exact sentence "Hope this helps!".

    The exclamation mark is part of the required sentence, not punctuation to
    be normalised away: the instruction names a sentence to reproduce, and an
    answer closing on "Hope this helps." reproduced a different one. Markdown
    and quotes around it are still tolerated."""
    return _tail(output).endswith(_FIXED_SENTENCE)


def verify_ends_with_question(output: str) -> bool:
    """Task 3 — last sentence is a question."""
    return _tail(output).rstrip(TRAILING_MARKS).endswith(("?", "？"))


def verify_single_sentence(output: str) -> bool:
    """Task 4 — exactly one sentence according to the English segmenter."""
    s = output.strip()
    if not s:
        return False
    sentences = [x for x in _SEGMENTER.segment(s) if x.strip()]
    return len(sentences) == 1


def verify_no_comma(output: str) -> bool:
    """Task 5 — no comma of any width."""
    return "," not in output and "，" not in output and "、" not in output


VERIFIERS = {
    "verify_lowercase": verify_lowercase,
    "verify_fixed_ending": verify_fixed_ending,
    "verify_ends_with_question": verify_ends_with_question,
    "verify_single_sentence": verify_single_sentence,
    "verify_no_comma": verify_no_comma,
}


def is_degenerate(output: str) -> bool:
    """Flag obvious repeated-trigram collapse.

    Part of the success definition: a collapsed output is not a valid answer
    even when the rule and the judge both pass it. Two failure modes made this
    necessary — the judge rates a collapse coherent when a sound opening
    precedes it, and under Task 5 collapse actively helps the rule, since
    repeated filler contains no commas.
    """
    toks = output.split()
    if not toks:
        return True
    if len(toks) < 6:
        return False
    trigrams = [tuple(toks[i:i + 3]) for i in range(len(toks) - 2)]
    return len(set(trigrams)) / len(trigrams) < 0.5


def score(output: str, verifier: str, truncated: bool = False) -> dict:
    """Return rule-level facts about one output, not a final verdict.

    An output that ran into the generation cap fails, whatever the task. The
    instruction constrains the answer, and an answer cut off mid-sentence is
    not one; scoring it on the visible fragment, or excluding it, both let a
    configuration be judged on the subset it happened not to break. Excluding
    was worse: the excluded set is chosen by the thing being measured, so
    steering that destroyed the model's ability to stop was graded only on
    its survivors — which are the easy items — and scored highest of all.

    The budget is set so this is a real failure rather than an artifact: the
    unsteered reference finishes every dev prompt inside it.

    An empty output fails too. Most verifiers reject one already, but "no
    comma" is satisfied by a string with nothing in it, and empty output is
    exactly what a think block that never closed leaves behind — so the worst
    generations were passing Task 5 on a technicality.
    """
    if not output.strip():
        return {"rule_pass": False, "truncated": truncated}
    return {
        "rule_pass": False if truncated else VERIFIERS[verifier](output),
        "truncated": truncated,
    }
