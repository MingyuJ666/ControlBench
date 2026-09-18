"""Observed prefix and the selected user turn, from saved replies and tool results.

Never execute model or reference calls here. Tools can depend on time or external
state, so replay cannot stand in for what the assistant actually saw.
"""
from .. import calls
from .driver import call_strings, restored_line, step_limit


def user_turn(item, index):
    messages = list(item['entry']['question'][index])
    restored = item['injections'].get(str(index))
    if restored:
        line = restored_line(restored)
        if not messages or messages[0]['content'] != line:
            messages = [{'role': 'user', 'content': line}] + messages
    return messages


def selected_turn(item: dict, generated: dict, turn_index: int) -> dict:
    replies = [turn['output'] for turn in generated.get('turns') or [generated]]
    events = generated.get('tool_executions', [])
    observed = {}
    for event in events:
        index = event['reply_index']
        if (type(index) is not int or not 0 <= index < len(replies) or index in observed
                or not isinstance(event.get('results'), list)
                or not all(isinstance(result, str) for result in event['results'])):
            raise ValueError(f"{item['item_id']}: invalid saved tool execution trace")
        observed[index] = event
    if events and generated.get('tool_trace_version') != 1:
        raise ValueError(f"{item['item_id']}: unsupported tool execution trace version")
    tools = list(item['tools'])
    for index in range(turn_index + 1):
        tools.extend(item['injections'].get(str(index), []))
    history, trace, before = [], [], None
    current, step = 0, 0
    if turn_index == 0:
        before = []
    history.extend(user_turn(item, 0))
    for index, text in enumerate(replies):
        if current > turn_index:
            break
        event = {'role': 'assistant', 'content': text}
        history.append(event)
        if current == turn_index:
            trace.append(event)
        attempt, parsed, _, _ = calls.parse(text)
        if attempt and parsed and step < step_limit():
            strings = call_strings(parsed)
            saved = observed.get(index)
            if (generated.get('tool_trace_version') != 1 or saved is None
                    or saved.get('calls') != strings or len(saved['results']) != len(strings)):
                raise ValueError(f"{item['item_id']}: missing or mismatched tool results at reply {index}; "
                                 "regenerate this conversation with tool trace recording")
            results = saved['results']
            if current == turn_index:
                trace.append({'role': 'tool_execution', 'calls': strings, 'results': results})
            history.extend({'role': 'tool', 'content': result} for result in results)
            step += 1
            continue
        if index in observed:
            raise ValueError(f"{item['item_id']}: unexpected tool execution at reply {index}")
        current += 1
        step = 0
        if current > turn_index:
            break
        if current == turn_index:
            before = list(history)
        history.extend(user_turn(item, current))
    # Empty/unreached turn: no fabricated assistant behavior; it cannot succeed.
    if before is None:
        before = list(history)
    return {'kind': 'decision_turn', 'history': before,
            'request': user_turn(item, turn_index), 'tools': tools, 'trace': trace}
