"""The prompting baseline: can a sentence of instruction do the steering?

One behaviour-matched sentence per axis, prepended as a system message on top
of the untouched chat template. This is the control every activation-level
method has to beat to justify itself -- if a sentence moves the decision as
far with as little damage, vectors are engineering, not findings. It is also
the direct measure of how much of the template's call-leaning framing plain
text can undo.

Dose is ignored: an instruction is present or absent. A dose curve for this
method is flat by construction, and that is the point of plotting it next to
the others.

Config: {"axis": "clarify" | "abstain"}, or {"system": "..."} to supply a
custom instruction.
"""

from __future__ import annotations

from .base import Method

INSTRUCTIONS = {
    # Verbatim from the previous codebase's prompting arms.
    "clarify": ("If any required parameter for a tool call is missing or "
                "unclear from the user's request, ask the user a clarifying "
                "question first instead of guessing or calling a tool."),
    "abstain": ("If none of the available tools can actually accomplish the "
                "user's request, clearly say you cannot help with the "
                "available tools. Do not call a tool that does not fit, and "
                "do not guess."),
    # Behaviour-matched counterpart for the call axis, same register as the
    # other two (new in this codebase; the previous one had no call arm).
    "call": ("If an available tool fits the user's request and the required "
             "information is present, call it directly. Do not ask for "
             "confirmation or additional preferences, and do not refuse."),
}


class Prompting(Method):
    name = "prompting"

    def system(self) -> str | None:
        if "system" in self.config:
            return self.config["system"]
        return INSTRUCTIONS[self.config["axis"]]
