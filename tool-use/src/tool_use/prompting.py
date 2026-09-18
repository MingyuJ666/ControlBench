"""Target instructions shared by the single- and multi-turn prompted baselines."""

TARGET_INSTRUCTIONS = {
    "call": (
        "Use the available tools to fulfill the user's request whenever an applicable "
        "tool and its required arguments are available. Return tool calls in the "
        "specified format instead of only describing what you would do. Do not "
        "invent tools or argument values."
    ),
    "clarify": (
        "When an applicable tool requires information that the user has not provided "
        "and the conversation does not establish, ask a focused clarification "
        "question in natural language before making that call. Do not guess missing "
        "arguments. Once the required information is available, continue with the "
        "appropriate tool calls in the specified format."
    ),
    "abstain": (
        "When none of the available tools can fulfill the user's request, explain "
        "that limitation in natural language without making an unsuitable or "
        "invented tool call. If a suitable tool becomes available later, use it "
        "with the required arguments in the specified format to continue the task."
    ),
}


def prompted_items(items_: list[dict]) -> list[dict]:
    """Add generation-only instructions without changing the task or its history.

    The instruction depends only on the target, never on reference calls,
    missing argument values, future tools, or judge-turn annotations. Its text
    is part of the generation input fingerprint, so edits invalidate the cache.
    """
    return [{**item, "generation_instruction": TARGET_INSTRUCTIONS[item["target"]]}
            for item in items_]
