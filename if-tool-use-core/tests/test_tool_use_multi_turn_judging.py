"""Regression checks for selected-turn context, BFCL gates and data repair."""
import copy
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from control_bench import paths
from control_bench.cli import judge_vllm
from control_bench.domains import tool_use_multi_turn as domain
from control_bench.domains.tool_use_multi_turn import judge_core as core
from control_bench.domains.tool_use.multi_turn import decision, driver, judging, scoring, trace
from control_bench.domains.tool_use.multi_turn.items import load_items, MULTI_TURN
from control_bench.runtime.search import (candidate, generate_candidate, read_jsonl, write_json, write_jsonl)
from test_native_methods import tiny_wrapper


def message(text):
    return [{'role': 'user', 'content': text}]


def fixture(category='miss_param'):
    return {'item_id': f'multi_turn_{category}_fixture',
            'category': category, 'target': {'base': 'call', 'miss_param': 'clarify', 'miss_func': 'abstain'}[category],
            'judge_turn_index': 1, 'tools': [{'name': 'add', 'parameters': {
                'type': 'object', 'properties': {'a': {'type': 'number'}, 'b': {'type': 'number'}},
                'required': ['a', 'b']}}], 'injections': {},
            'answer': [['add(a=1, b=2)'], [], ['add(a=3, b=4)']],
            'entry': {'id': f'multi_turn_{category}_fixture', 'involved_classes': ['MathAPI'],
                      'initial_config': {}, 'question': [message('Add 1 and 2.'),
                         message('Add the result to another number.'), message('Use 4. FUTURE_SECRET')]}}


def generated(item, texts=None):
    if texts is None:
        texts = ['[add(a=1, b=2)]', 'The result is 3.', 'Which number?', '[add(a=3, b=4)]', '7.']
    turns = [{'output': t, 'truncated': False, 'n_new_tokens': 5} for t in texts]
    return {'item_id': item['item_id'], **turns[0], 'turns': turns, 'tool_trace_version': 1,
            'tool_executions': [{'reply_index': 0, 'calls': ['add(a=1, b=2)'],
                                 'results': ['{"result": 3}']},
                                {'reply_index': 3, 'calls': ['add(a=3, b=4)'],
                                 'results': ['{"result": 7}']}]}


def verdict(action='clarify', coherent=True, appropriate=True):
    return {'note': 'The required second operand is missing.', 'action': action,
            'coherent': coherent, 'appropriate': appropriate}


def scored(item=None, output=None, judgment=None):
    item = item or fixture()
    output = output or generated(item)
    records = scoring.build_records(item['target'], 'full', [item], [output], 'private_config')
    if judgment is not None:
        uid = f"{records[0]['uid']}|turn_{records[0]['judge_turn_index']}"
        scoring.merge(records, {uid: judgment})
    return records


def save_verdicts(folder, value=None):
    shards = sorted(folder.glob('judge_in_*.jsonl'))
    value = value or verdict()
    for shard in shards:
        write_jsonl(shard.with_name(shard.name.replace('judge_in_', 'judge_out_')),
                    [core.parse(json.dumps(value), r['uid']) for r in read_jsonl(shard)])
    write_json(folder / 'judge_run.json', core.run_spec([p.name for p in shards], core.JUDGE_MODEL, 'vllm', {}))


class FrozenDataTest(unittest.TestCase):
    def test_all_400_decision_turns_reviewed_without_changing_counts(self):
        rows = load_items()
        self.assertEqual(Counter(r['target'] for r in rows), {'call': 200, 'clarify': 200, 'abstain': 200})
        for item in rows:
            index = decision.decision_turn(item)
            if item['category'] == 'base':
                self.assertIsNone(index)
            else:
                self.assertEqual(item['answer'][index], [])
                if item['category'] == 'miss_func':
                    self.assertIn(str(index + 1), item['injections'])
        by_id = {r['item_id']: r for r in rows}
        self.assertEqual(decision.decision_turn(by_id['multi_turn_miss_param_167']), 1)
        self.assertEqual(decision.decision_turn(by_id['multi_turn_miss_param_180']), 5)

    def test_correction_is_one_row_with_reproducible_original_and_required_missing_title(self):
        folder = MULTI_TURN.parent
        repair = json.loads((folder / 'multi_turn_corrections.json').read_text())
        change, = repair['corrections']
        rows = load_items()
        fixed = next(r for r in rows if r['item_id'] == change['item_id'])
        self.assertEqual(fixed, change['corrected'])
        base = next(r for r in rows if r['item_id'] == change['paired_base_item'])
        self.assertEqual(fixed['entry']['question'][:5], base['entry']['question'][:5])
        self.assertEqual(fixed['answer'], base['answer'][:5] + [[]] + base['answer'][5:])
        title_tool = next(t for t in fixed['tools'] if t['name'] == 'create_ticket')
        self.assertIn('title', title_tool['parameters']['required'])
        self.assertNotIn('description', title_tool['parameters']['required'])
        self.assertNotIn('Invoice Discrepancy', json.dumps(fixed['entry']['question'][:6]))
        self.assertIn('Invoice Discrepancy', json.dumps(fixed['entry']['question'][6]))
        manifest = json.loads((folder / 'MANIFEST.json').read_text())
        self.assertEqual(hashlib.sha256(MULTI_TURN.read_bytes()).hexdigest(),
                         manifest['files']['multi_turn.jsonl']['sha256'])
        # Original line formatting and every unaffected row must be retained.
        raw_lines = MULTI_TURN.read_text().splitlines(keepends=True)
        pos = next(i for i, r in enumerate(rows) if r['item_id'] == change['item_id'])
        raw_lines[pos] = json.dumps(change['original'], ensure_ascii=False) + '\n'
        self.assertEqual(hashlib.sha256(''.join(raw_lines).encode()).hexdigest(), repair['upstream_sha256'])

    def test_changed_unreviewed_data_is_rejected(self):
        item = next(r for r in load_items() if r['category'] == 'miss_param')
        item['entry']['question'][0][0]['content'] += ' changed'
        with self.assertRaisesRegex(decision.DecisionTurnError, 'input changed'):
            decision.decision_turn(item)

    def test_corrected_reference_executes_and_passes_bfcl(self):
        item = next(r for r in load_items() if r['item_id'] == 'multi_turn_miss_param_180')
        replies = []
        for calls in item['answer']:
            if calls:
                replies.append('[' + ', '.join(calls) + ']')
            replies.append('What exact title should I use?' if not calls else 'Done.')
        result = scoring.score_item(item, replies)
        self.assertTrue(result['bfcl_passed'], result)
        self.assertIsNone(result['passed'])


class SelectedTurnTest(unittest.TestCase):
    def test_one_row_with_actual_past_results_and_no_future_or_hidden_labels(self):
        item = fixture()
        row, = judging.judge_rows('clarify', 'full', [item], [generated(item)], 'private_config')
        self.assertEqual(row['trace'], [{'role': 'assistant', 'content': 'Which number?'}])
        self.assertEqual(row['history'][-2], {'role': 'tool', 'content': '{"result": 3}'})
        prompt = core.build_prompt({**row, 'gold': 'HIDDEN_GOLD', 'target': 'HIDDEN_TARGET'})
        for hidden in ('FUTURE_SECRET', 'private_config', 'HIDDEN_GOLD', 'HIDDEN_TARGET', 'initial_config', 'Use 4.'):
            self.assertNotIn(hidden, prompt)
        self.assertIn('The result is 3.', prompt)
        self.assertNotIn('add(a=3, b=4)', prompt)

    def test_restored_tools_only_visible_once_restored(self):
        item = fixture('miss_func')
        restored = {'name': 'FUTURE_TOOL_SECRET', 'parameters': {}}
        item['injections'] = {'2': [restored]}
        first = trace.selected_turn(item, generated(item), 1)
        self.assertNotIn('FUTURE_TOOL_SECRET', core.build_prompt(first))
        later = trace.selected_turn(item, generated(item), 2)
        self.assertIn(restored, later['tools'])
        self.assertIn('FUTURE_TOOL_SECRET', later['request'][0]['content'])

    def test_earlier_call_then_clarification_both_reach_judge(self):
        item = fixture()
        output = generated(item, ['[add(a=1, b=2)]', '3.', '[add(a=3, b=100)]', 'Which number?',
                                  '[add(a=3, b=4)]', '7.'])
        output['tool_executions'][1]['reply_index'] = 4
        output['tool_executions'].append({'reply_index': 2, 'calls': ['add(a=3, b=100)'], 'results': ['103']})
        row = trace.selected_turn(item, output, 1)
        self.assertEqual([e['role'] for e in row['trace']], ['assistant', 'tool_execution', 'assistant'])
        records = scored(item, output, verdict())  # Even an overgenerous judge cannot override the actual call.
        self.assertFalse(records[0]['decision'])
        self.assertFalse(records[0]['irrelevance_passed'])
        self.assertFalse(records[0]['passed'])

    def test_missing_tool_results_rejected_instead_of_replayed(self):
        item = fixture()
        output = generated(item)
        output.pop('tool_executions')
        with self.assertRaisesRegex(ValueError, 'regenerate'):
            trace.selected_turn(item, output, 1)

    def test_mismatched_saved_call_rejected(self):
        item = fixture()
        output = generated(item)
        output['tool_executions'][0]['calls'] = ['add(a=999, b=999)']
        with self.assertRaisesRegex(ValueError, 'mismatched'):
            trace.selected_turn(item, output, 1)

    def test_empty_unreached_turn_not_fabricated(self):
        item = fixture()
        output = generated(item, ['I cannot do the first task.'])
        output['tool_executions'] = []
        selected = trace.selected_turn(item, output, 1)
        self.assertEqual(selected['trace'], [])
        records = scored(item, output, verdict('direct_or_other', False, False))
        self.assertFalse(records[0]['passed'])
        self.assertEqual(records[0]['error'], 'multi_turn:turn_count_mismatch')
        self.assertFalse(scored(item, output, verdict())[0]['decision'])

    def test_step_limit_call_is_present_but_not_falsely_executed(self):
        item = fixture()
        output = generated(item, ['[add(a=1, b=2)]', '[add(a=1, b=2)]', 'Which number?', 'Done.'])
        output['tool_executions'] = output['tool_executions'][:1]
        with patch.object(trace, 'step_limit', return_value=1), patch.object(scoring, 'step_limit', return_value=1):
            selected = trace.selected_turn(item, output, 0)
            self.assertEqual([r['role'] for r in selected['trace']], ['assistant', 'tool_execution', 'assistant'])
            calls, texts = scoring.group_turns([t['output'] for t in output['turns']])
            self.assertEqual(len(texts[0]), 2)
            self.assertEqual(len(calls[0]), 1)


class ScoringTest(unittest.TestCase):
    def test_execution_and_decision_are_reported_separately(self):
        records = scored(judgment=verdict())
        summary = scoring.summarize(records)
        self.assertEqual((summary['accuracy'], summary['bfcl_accuracy'], summary['decision_accuracy']), (1, 1, 1))
        for wrong in (verdict('direct_or_other'), verdict(coherent=False), verdict(appropriate=False)):
            records = scored(judgment=wrong)
            summary = scoring.summarize(records)
            self.assertEqual((summary['accuracy'], summary['bfcl_accuracy'], summary['decision_accuracy']), (0, 1, 0))

    def test_no_call_gate_closes_native_execution_check_gap(self):
        item = fixture('base')
        output = generated(item, ['[add(a=1, b=2)]', '3.', '[add(a=3, b=100)]', 'Done.',
                                  '[add(a=3, b=4)]', '7.'])
        result = scored(item, output)[0]
        self.assertTrue(result['execution_passed'])  # stateless extra call would otherwise be accepted
        self.assertFalse(result['irrelevance_passed'])
        self.assertFalse(result['passed'])

    def test_wrong_prior_execution_does_not_change_current_decision(self):
        item = fixture()
        output = generated(item)
        output['turns'][0]['output'] = '[add(a=100, b=2)]'
        records = scored(item, output, verdict())
        self.assertTrue(records[0]['decision'])
        self.assertFalse(records[0]['bfcl_passed'])
        self.assertFalse(records[0]['passed'])

    def test_missing_judgment_stays_unknown(self):
        records = scored()
        self.assertIsNone(records[0]['passed'])
        with self.assertRaisesRegex(RuntimeError, 'missing'):
            scoring.summarize(records)

    def test_bfcl_checker_releases_only_its_own_instances(self):
        from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils as backend
        marker = 'unrelated_fixture_instance'
        setattr(backend, marker, object())
        before = set(vars(backend))
        try:
            scored(judgment=verdict())
            self.assertEqual(set(vars(backend)), before)
        finally:
            delattr(backend, marker)


class CacheAndRunnerTest(unittest.TestCase):
    def rows(self):
        item = fixture()
        return judging.judge_rows('clarify', 'full', [item], [generated(item)], 'config')

    def test_call_has_no_judge_rows_or_model_process(self):
        item = fixture('base')
        self.assertEqual(judging.judge_rows('call', 'full', [item], [generated(item)], 'config'), [])
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.assertFalse(judging.judging_complete(folder))
            judging.prepare_shards([], folder, mechanical=True)
            with patch.object(domain.subprocess, 'run', side_effect=AssertionError('must not start Qwen')):
                domain.judge([folder])
            self.assertTrue(judging.judging_complete(folder))
            summary = scoring.summarize(scored(item, generated(item)))
            self.assertIsNone(summary['decision_accuracy'])
            self.assertEqual(summary['n_judged'], 0)

    def test_cache_reuses_exact_inputs_but_invalidates_changed_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            rows = self.rows()
            judging.prepare_shards(rows, folder)
            save_verdicts(folder)
            self.assertTrue(judging.judging_complete(folder))
            before = (folder / 'judge_out_0.jsonl').read_bytes()
            judging.prepare_shards(rows, folder)
            self.assertEqual((folder / 'judge_out_0.jsonl').read_bytes(), before)
            rows[0]['trace'][0]['content'] = 'I already did it.'
            judging.prepare_shards(rows, folder)
            self.assertFalse(list(folder.glob('judge_out_*.jsonl')))
            self.assertFalse(judging.judging_complete(folder))

    def test_stale_prompt_or_malformed_raw_cache_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            judging.prepare_shards(self.rows(), folder)
            save_verdicts(folder)
            out = folder / 'judge_out_0.jsonl'
            row = read_jsonl(out)[0]
            row['raw_response'] = '{"asks_for":true}'
            write_jsonl(out, [row])
            self.assertFalse(judging.judging_complete(folder))
            judging.prepare_shards(self.rows(), folder)
            self.assertFalse(list(folder.glob('judge_out_*.jsonl')))
            save_verdicts(folder)
            with patch.object(core, 'INSTRUCTIONS', core.INSTRUCTIONS + '\nChanged rubric.'):
                self.assertFalse(judging.judging_complete(folder))

    def test_parser_is_strict_and_core_loads_standalone(self):
        standalone = judge_vllm.load_core('tool_use_multi_turn')
        self.assertEqual(standalone.parse(json.dumps(verdict()), 'id')['action'], 'clarify')
        for text in ('{"asks_for":true}', json.dumps({**verdict(), 'appropriate': 'true'}),
                     '{"action":"clarify","action":"call","coherent":true,"appropriate":true,"note":"x"}',
                     json.dumps({**verdict(), 'extra': 1})):
            self.assertIsNone(standalone.parse(text, 'id'))
        self.assertIsNotNone(standalone.parse_retry(json.dumps(verdict()), 'id'))

    def test_unparsed_retry_is_one_combined_verdict_with_same_prompt(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return json.dumps(messages)
        requests = []
        def generate(prompts, sampling, **kwargs):
            requests.append(prompts)
            text = '{"asks_for":true}' if len(requests) == 1 else json.dumps(verdict())
            return [SimpleNamespace(outputs=[SimpleNamespace(text=text)])]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'judge_in_0.jsonl'
            write_jsonl(source, self.rows())
            judge_vllm.judge_shard(SimpleNamespace(generate=generate), Tokenizer(), core,
                                   object(), source, retry_sampling=object())
            self.assertEqual(len(requests), 2)
            self.assertEqual(requests[0], requests[1])
            row, = read_jsonl(source.with_name('judge_out_0.jsonl'))
            self.assertEqual(row['action'], 'clarify')
            self.assertEqual(row['previous_raw_response'], '{"asks_for":true}')

    def test_generation_saves_actual_results_for_selected_turn_history(self):
        item = fixture()
        replies = iter([t['output'] for t in generated(item)['turns']])
        def output(*args, **kwargs):
            return [{'output': next(replies), 'truncated': False, 'n_new_tokens': 5}]
        wrapper = tiny_wrapper()
        with tempfile.TemporaryDirectory() as tmp, patch.object(paths, 'DATA_ROOT', Path(tmp)), \
             patch('control_bench.runtime.search.generate_texts', side_effect=output):
            folder = generate_candidate(wrapper, domain, model=wrapper.model_id, experiment='test',
                method='unsteered', target='clarify', cand=candidate('resid_post_0', 0), artifact=None,
                items=[item], batch_size=1, max_new_tokens=8)
            saved, = read_jsonl(folder / 'generations.jsonl')
            self.assertEqual(saved['tool_trace_version'], 1)
            self.assertEqual(len(saved['tool_executions']), 2)
            expected = driver.state_of(item)['tool_executions'][0]['results'][0]
            selected = trace.selected_turn(item, saved, 1)
            self.assertIn({'role': 'tool', 'content': expected}, selected['history'])
            self.assertEqual(scoring.summarize(scored(item, saved, verdict()))['accuracy'], 1)


if __name__ == '__main__':
    unittest.main()
