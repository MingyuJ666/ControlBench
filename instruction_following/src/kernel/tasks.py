"""Canonical instruction-following tasks and prompt construction.

Each task has one instruction and the name of its rule-based verifier. The
same instruction is appended for prompted reference generation and for the
positive half of each activation-extraction pair.
"""

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

    base = base_prompt.strip()
    return f"{base} {TASKS[task]['instruction']}"


def make_pair(base_prompt: str, task: str) -> tuple[str, str]:
    """Return ``(prompted, unprompted)`` for activation extraction.

    The two strings differ only by the instruction appended to the prompted
    string, so their activation difference isolates that intervention.
    """
    base = base_prompt.strip()
    return make_prompted_prompt(base, task), base
