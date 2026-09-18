"""The single-turn track: one request, one decision, scored.

The model sees a request and a tool list. Each response receives one blind
judgment of action, coherence and appropriateness. The program additionally
checks submitted calls against the tool signature and the expected tool name.

An item's `gold.followup` scripts a second turn -- for a clarify item the user
supplies the withheld argument, for an abstain item the missing tool comes
back; call items have none, being complete at the decision. The next_turn
hook plays this follow-up, and scoring reports first-decision accuracy and
whether the model then calls the expected tool.
"""

from .followup import check_second, next_turn
from .items import EVAL, eval_items, load_items
from .scoring import (build_records, judging_complete, judge_rows, load_verdicts,
                      merge, prepare_shards, summarize)

__all__ = ["EVAL", "check_second", "next_turn", "eval_items", "load_items", "build_records", "judging_complete",
           "judge_rows", "load_verdicts", "merge", "prepare_shards", "summarize"]
