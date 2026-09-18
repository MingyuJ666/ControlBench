"""One blind row per annotated missing turn; call/base needs no LLM process."""
import hashlib
import json
import random
from pathlib import Path

from .decision import decision_turn
from .trace import selected_turn

PREPARATION_VERSION = 1


def core():
    # Lazy import avoids a cycle through the registered domain's package.
    from ...tool_use_multi_turn import judge_core
    return judge_core


def judge_rows(target, split, items, outputs, config):
    rows = []
    for item, generated in zip(items, outputs, strict=True):
        if item['target'] != target or generated['item_id'] != item['item_id']:
            raise ValueError('multi-turn outputs and items are misaligned')
        index = decision_turn(item)
        if index is not None:
            rows.append({'uid': f"{target}|{split}|{config}|{item['item_id']}|turn_{index}",
                         **selected_turn(item, generated, index)})
    return rows


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _write(path, text):
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)


def prepare_shards(rows, out_dir: Path, *, mechanical=False, n_batches=4, seed=11):
    if mechanical and rows:
        raise ValueError('mechanical scoring cannot have judge questions')
    if not mechanical and not rows:
        raise ValueError('missing-turn judging requires questions')
    if len({r['uid'] for r in rows}) != len(rows):
        raise ValueError('duplicate multi-turn judge input UID')
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    shards = ({} if mechanical else {
        f'judge_in_{i}.jsonl': ''.join(json.dumps(r, ensure_ascii=False) + '\n'
                                     for r in rows[i::n_batches]) for i in range(n_batches)})
    expected = {'schema_version': PREPARATION_VERSION, 'mechanical_only': mechanical,
                'n_questions': len(rows), 'prompt_sha256': core()._prompt_sha256(),
                'shards': {name: _digest(text) for name, text in shards.items()}}
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / 'preparation.json'
    old = {p.name: p.read_text() for p in out_dir.glob('judge_in_*.jsonl')}
    try:
        unchanged = json.loads(marker.read_text()) == expected and old == shards
        if unchanged and (mechanical or (not (out_dir / 'judge_run.json').exists()
                                      and not list(out_dir.glob('judge_out_*.jsonl')))):
            return len(rows)
        if unchanged:
            load_verdicts(out_dir)
            return len(rows)
    except (OSError, ValueError, TypeError):
        pass
    for pattern in ('judge_in_*.jsonl', 'judge_out_*.jsonl', 'judge_unparsed_*.jsonl'):
        for path in out_dir.glob(pattern):
            path.unlink()
    (out_dir / 'judge_run.json').unlink(missing_ok=True)
    for name, text in shards.items():
        _write(out_dir / name, text)
    _write(out_dir / 'judge_instructions.txt', core().INSTRUCTIONS)
    _write(marker, json.dumps(expected, indent=2) + '\n')
    return len(rows)


def preparation(out_dir):
    marker = json.loads((out_dir / 'preparation.json').read_text())
    actual = {p.name: _digest(p.read_text()) for p in out_dir.glob('judge_in_*.jsonl')}
    if (marker.get('schema_version') != PREPARATION_VERSION
            or type(marker.get('mechanical_only')) is not bool
            or marker.get('prompt_sha256') != core()._prompt_sha256()
            or marker.get('shards') != actual):
        raise ValueError(f'{out_dir}: stale or changed multi-turn judge preparation')
    if marker['mechanical_only'] and (actual or marker.get('n_questions') != 0):
        raise ValueError('mechanical-only scoring cannot contain judge questions')
    return marker


def validate_run(out_dir):
    marker = preparation(out_dir)
    if marker['mechanical_only']:
        return
    shards = sorted(marker['shards'])
    if not shards:
        raise ValueError(f'{out_dir}: no multi-turn judge inputs')
    actual = json.loads((out_dir / 'judge_run.json').read_text())
    expected = core().run_spec(shards, core().JUDGE_MODEL, 'vllm', {})
    if not isinstance(actual, dict) or any(actual.get(k) != v for k, v in expected.items()):
        raise ValueError(f'{out_dir}: multi-turn judge settings changed; rerun judge')


def load_verdicts(out_dir):
    validate_run(out_dir)
    if preparation(out_dir)['mechanical_only']:
        return {}
    verdicts = {}
    expected_all = []
    for path in sorted(out_dir.glob('judge_in_*.jsonl')):
        ids = [json.loads(line)['uid'] for line in path.read_text().splitlines()]
        expected_all.extend(ids)
        output = path.with_name(path.name.replace('judge_in_', 'judge_out_'))
        if not output.exists():
            continue
        for line in output.read_text().splitlines():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'{output}: invalid verdict')
            uid = row.get('uid')
            if not isinstance(uid, str) or uid not in ids or uid in verdicts:
                raise ValueError(f'{output}: unexpected or duplicate verdict UID {uid!r}')
            fields = core()._fields(row)
            raw = row.get('raw_response')
            parsed = core().parse(raw, uid) if isinstance(raw, str) else None
            if fields is None or parsed is None or core()._fields(parsed) != fields:
                raise ValueError(f'{output}: malformed or mismatched raw verdict for {uid}')
            if row.get('judge_mode') and (row.get('judge_mode') != core().RETRY_MODE
                                        or row.get('retry_spec') != core().retry_spec()):
                raise ValueError(f'{output}: stale retry verdict for {uid}')
            verdicts[uid] = fields
    if len(expected_all) != len(set(expected_all)):
        raise ValueError('duplicate multi-turn input UID across shards')
    if len(expected_all) != preparation(out_dir)['n_questions']:
        raise ValueError('multi-turn input count differs from preparation')
    return verdicts


def judging_complete(out_dir):
    try:
        marker = preparation(out_dir)
        return len(load_verdicts(out_dir)) == marker['n_questions']
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False
