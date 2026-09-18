"""Playing a BFCL conversation: execute what the model calls, feed the result back.

A user turn is not one exchange. The model may call a tool, read the result,
call another, and only then speak; BFCL allows up to MAXIMUM_STEP_LIMIT such
steps before the turn is cut off. So after every reply this asks: did the model
call something? Then execute it and hand back the result, still in this turn.
Did it not? Then the turn is over and the next user line arrives.

Calls are executed against BFCL's own stateful backends, so a cd() moves a real
directory object and a trade moves a real balance. Each conversation gets its
own worlds: the executor caches backend instances by (model_name, entry id), so
a run name unique to this playthrough is what keeps one conversation's state out
of the next one's.

Restoring a withheld function follows BFCL's prompting convention rather than
its function-calling one, because that is the transport every model here is
given: the restored documentation arrives inside the user's line
(DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING), and the tool block at
the top of the conversation stays the list the conversation started with.

The state this keeps is only what generation needs. Scoring does not read it:
the grouping of calls into turns is a deterministic function of the replies,
which scoring.py replays from what was stored.
"""

import itertools
import json

from .. import calls as calls_mod
from ..render import conversation, render_chat

#: One counter per process; a playthrough's run name must not collide with a
#: previous one, or it would inherit its backend state.
_RUNS = itertools.count()
_STATE: dict[str, dict] = {}


def step_limit() -> int:
    from bfcl_eval.constants.default_prompts import MAXIMUM_STEP_LIMIT
    return MAXIMUM_STEP_LIMIT


def call_strings(parsed: list[dict]) -> list[str]:
    """Parsed calls -> the Python call strings the executor takes."""
    out = []
    for call in parsed:
        arguments = [repr(value) for value in call.get("positional") or []]
        arguments += [f"{key}={value!r}" for key, value in call["args"].items()]
        out.append(f"{call['name']}(" + ", ".join(arguments) + ")")
    return out


def restored_line(documentation: list[dict]) -> str:
    from bfcl_eval.constants.default_prompts import (
        DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING)
    return DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING.format(
        functions=json.dumps(documentation, indent=2))


def _start(item: dict) -> dict:
    state = {"turn": 0, "step": 0, "run": f"cb_{next(_RUNS)}",
             "history": list(item["entry"]["question"][0]),
             "turn_calls": [[]], "turn_texts": [[]], "force_quit": False,
             "tool_executions": []}
    _STATE[item["item_id"]] = state
    return state


def state_of(item: dict) -> dict:
    return _STATE[item["item_id"]]


def next_turn(tokenizer, item: dict, turns: list[str]) -> str | None:
    """The prompt for the next exchange, or None when the conversation is over."""
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call)

    state = _start(item) if len(turns) == 1 else state_of(item)
    entry, text = item["entry"], turns[-1]
    state["turn_texts"][-1].append(text)
    state["history"] = state["history"] + [{"role": "assistant", "content": text}]

    attempt, parsed, _, _ = calls_mod.parse(text)
    if attempt and parsed:
        if state["step"] >= step_limit():
            state["force_quit"] = True          # still calling when the turn was cut off
        else:
            strings = call_strings(parsed)
            state["turn_calls"][-1].append(strings)
            results, _ = execute_multi_turn_func_call(
                strings, entry["initial_config"], entry["involved_classes"],
                state["run"], item["item_id"], long_context=False)
            state["tool_executions"].append({"reply_index": len(turns) - 1,
                                              "calls": strings,
                                              "results": [str(result) for result in results]})
            state["step"] += 1
            state["history"] = state["history"] + [{"role": "tool", "content": str(result)}
                                                   for result in results]
            return render_chat(tokenizer, conversation(
                state["history"], item["tools"],
                instruction=item.get("generation_instruction")))

    state["turn"] += 1
    state["step"] = 0
    if state["turn"] >= len(entry["question"]):
        return None
    state["turn_calls"].append([])
    state["turn_texts"].append([])
    spoken = list(entry["question"][state["turn"]])
    restored = item["injections"].get(str(state["turn"]))
    if restored:
        spoken = spoken or [{"role": "user", "content": restored_line(restored)}]
        if spoken[0]["content"] != restored_line(restored):
            spoken = [{"role": "user", "content": restored_line(restored)}] + spoken
    state["history"] = state["history"] + spoken
    return render_chat(tokenizer, conversation(
        state["history"], item["tools"],
        instruction=item.get("generation_instruction")))


def generation_record(item: dict, replies: list[str]) -> dict:
    """Save actual executor output; judging must not recreate historical results."""
    state = state_of(item)
    if sum(map(len, state["turn_texts"])) != len(replies):
        raise ValueError(f"{item['item_id']}: generation trace and replies differ")
    return {"tool_trace_version": 1, "tool_executions": state["tool_executions"]}
