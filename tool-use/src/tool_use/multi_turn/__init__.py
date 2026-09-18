"""The multi-turn track: BFCL v4 multi_turn against executable backends.

Here a model's calls are executed -- a cd() moves a real directory object, a
trade moves a real balance -- and the results feed back into the conversation.
Three categories of 200 conversations mirror the three targets: base is call,
miss_param is clarify, and miss_func is abstain, with the withheld tool
restored a turn later.

Directions are not learned here. The track reuses what the single-turn
contrastive pairs produced, which is the question it exists to answer: does a
direction learned on one decision still hold where the decision has
consequences?

This package holds the track's own pieces. The registered domain that exposes
them to the command line is control_bench.domains.tool_use_multi_turn.
"""

from .driver import next_turn
from .items import MULTI_TURN, eval_items, load_items
from .scoring import build_records, group_turns, score_item, summarize

__all__ = ["MULTI_TURN", "next_turn", "eval_items", "load_items",
           "build_records", "group_turns", "score_item", "summarize"]
