"""Compatibility entry for the tool_use_multi_turn domain.

Implementation lives in tool_use.multi_turn. Alias the module itself so legacy
imports and patches refer to the same functions and globals, without copies.
"""
import sys as _sys

from .tool_use.multi_turn import domain as _domain, judge_core as _judge_core

_domain.judge_core = _judge_core
_sys.modules[__name__ + ".judge_core"] = _judge_core
_sys.modules[__name__] = _domain
