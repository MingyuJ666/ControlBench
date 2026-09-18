"""The vocabulary and the paths both tracks share.

Given a user request and a tool list, an agent must call the right tool, ask
for the argument it is missing, or decline when no tool fits. Those three
actions are the targets of both tracks; `direct` -- answering from the model's
own knowledge, or announcing an outcome as though a tool had run -- is a
negative-side action only, since inventing an answer is never correct.
"""

from ...paths import input_root

NAME = "tool_use"
INPUTS = input_root(NAME)

#: What a response can be. Only the first three are ever the correct answer,
#: so only they are targets.
ACTIONS = ("call", "clarify", "abstain", "direct")
TARGETS = ACTIONS[:3]

#: One evaluation set, nothing held out. The shared layer asks for it by name.
EVAL_SPLIT = "full"
