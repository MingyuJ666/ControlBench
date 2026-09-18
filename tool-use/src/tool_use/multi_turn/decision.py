"""Predeclared evaluation turns; these annotations never enter generation or judge prompts."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path


class DecisionTurnError(ValueError):
    pass


@lru_cache(maxsize=1)
def annotations():
    return json.loads(Path(__file__).with_name('judge_turns.json').read_text())['items']


def decision_turn(item: dict) -> int | None:
    expected = {'base': 'call', 'miss_param': 'clarify', 'miss_func': 'abstain'}
    if item.get('target') != expected.get(item['category']) or item['category'] not in expected:
        raise DecisionTurnError(f"{item['item_id']}: category and target disagree")
    if item['category'] == 'base':
        return None
    # Explicit annotations support independent fixtures/custom datasets without changing the frozen corpus.
    if 'judge_turn_index' in item:
        turn = item['judge_turn_index']
    else:
        entry = annotations().get(item['item_id'])
        if entry is None:
            raise DecisionTurnError(f"{item['item_id']}: no reviewed judge-turn annotation")
        digest = hashlib.sha256(json.dumps(item, sort_keys=True, ensure_ascii=False,
                                          separators=(',', ':')).encode()).hexdigest()
        if digest != entry['item_sha256']:
            raise DecisionTurnError(f"{item['item_id']}: input changed; review its judge-turn annotation")
        turn = entry['turn_index']
        if turn is None:
            raise DecisionTurnError(f"{item['item_id']}: {entry['reason']}")
    if type(turn) is not int or not 0 <= turn < len(item['answer']) or item['answer'][turn] or turn >= len(item['entry']['question']):
        raise DecisionTurnError(f"{item['item_id']}: judge turn must be a valid no-call reference turn")
    return turn
