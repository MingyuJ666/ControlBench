import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import core as cb
from core import paths
from core.domains import tool_use, tool_use_multi_turn
from core.domains.tool_use import calls, judge_core
from core.domains.tool_use.multi_turn.driver import call_strings
from core.runtime.experiment import Run
from core.runtime.search import read_jsonl, write_json, write_jsonl
from test_native_methods import tiny_wrapper


TOOLS = [{'name': 'book', 'parameters': {
    'type': 'object', 'properties': {'city': {'type': 'string'}, 'date': {'type': 'string'}},
    'required': ['city', 'date']}}]


class ToolCallValidationTest(unittest.TestCase):
    def test_all_three_formats_accept_a_complete_single_call(self):
        replies = [
            '[book(city="paris", date="tomorrow")]',
            '<tool_call>{"name":"book","arguments":{"city":"paris","date":"tomorrow"}}</tool_call>',
            '{"name":"book","parameters":{"city":"paris","date":"tomorrow"}}',
            'Sure. [book(city="paris", date="tomorrow")]',
            '<tool_call>{"name":"book","arguments":{"city":"paris","date":"tomorrow"}}',
        ]
        for reply in replies:
            with self.subTest(reply=reply):
                result = calls.check(reply, TOOLS, ['book'])
                self.assertTrue(result['success'])
                self.assertEqual(result['n_calls'], 1)

    def test_bare_calls_match_bracketed_calls_without_rewriting_arguments(self):
        examples = [
            'book(city="paris", date="tomorrow")',
            'book(\ncity="paris",\ndate="tomorrow"\n)',
            'book("paris", date="tomorrow")',
            'book(city="[paris]", date="tomorrow")',
        ]
        for text in examples:
            with self.subTest(text=text):
                self.assertEqual(calls.check(text, TOOLS, ['book']),
                                 calls.check('[' + text + ']', TOOLS, ['book']))
                self.assertTrue(calls.check(text, TOOLS, ['book'])['success'])
                self.assertEqual(calls.parse('```python\n' + text + '\n```')[1],
                                 calls.parse(text)[1])
        nested = 'store(payload=[1, 2])'
        self.assertEqual(calls.parse(nested)[1], calls.parse('[' + nested + ']')[1])

    def test_non_serializable_literals_are_invalid_without_breaking_scores(self):
        for value in ("...", "[1, ...]", "{'nested': ...}", "b'paris'", "{1, 2}", "1j", "1e999"):
            for template in ('book(city=%s, date="tomorrow")', '[book(city=%s, date="tomorrow")]', '[book(%s, "tomorrow")]'):
                with self.subTest(value=value, template=template):
                    result = calls.check(template % value, TOOLS, ['book'])
                    self.assertTrue(result['attempt'])
                    self.assertFalse(result['valid'])
                    self.assertEqual(result['invalid_reason'], 'unparseable')
                    json.dumps(result, allow_nan=False)
        self.assertTrue(calls.check('book(city="...", date="tomorrow")', TOOLS, ['book'])['valid'])

    def test_bare_calls_keep_schema_and_multiple_call_checks(self):
        cases = [
            ('book(city="paris")', 'missing_required'),
            ('book(city=5, date="tomorrow")', 'type_mismatch'),
            ('book(city="paris", date="tomorrow", invented=True)', 'invented_param'),
            ('book(city="paris", date="tomorrow"', 'unparseable'),
            ('book(city=paris, date="tomorrow")', 'unparseable'),
            ('book(city="paris", date="tomorrow"), delete_everything()', 'multiple_calls'),
            ('[book(city="paris", date="tomorrow")]', None),
        ]
        for text, reason in cases:
            with self.subTest(text=text):
                self.assertEqual(calls.check(text, TOOLS, ['book'])['invalid_reason'], reason)
        text = 'book(city="paris", date="tomorrow"), book(city="rome", date="today")'
        self.assertEqual(len(calls.parse(text)[1]), 2)
        self.assertEqual(len(call_strings(calls.parse(text)[1])), 2)
        for text in [
            '[book(city="paris", date="tomorrow")]\ndelete_everything()',
            'book(city="paris", date="tomorrow")\n{"name":"delete_everything","arguments":{}}',
        ]:
            with self.subTest(text=text):
                self.assertFalse(calls.check(text, TOOLS, ['book'])['success'])

    def test_missing_duplicate_and_extra_parameters_fail(self):
        cases = [
            ('[book("paris")]', 'missing_required'),
            ('[book(city="paris")]', 'missing_required'),
            ('[book("paris", "tomorrow", "extra")]', 'too_many_positional'),
            ('[book("paris", city="rome", date="tomorrow")]', 'duplicate_param'),
            ('[book(city="paris", date="tomorrow", invented=1)]', 'invented_param'),
            ('[book(city="paris", city="rome", date="tomorrow")]', 'unparseable'),
        ]
        for reply, reason in cases:
            with self.subTest(reply=reply):
                result = calls.check(reply, TOOLS, ['book'])
                self.assertTrue(result['attempt'])
                self.assertFalse(result['valid'])
                self.assertFalse(result['success'])
                self.assertEqual(result['invalid_reason'], reason)
        self.assertEqual(calls.check('[ping(x=1)]', [{'name': 'ping'}], ['ping'])['invalid_reason'],
                         'invented_param')

    def test_positional_binding_checks_all_parameters_and_followup_value(self):
        for reply in ['[book("paris", "tomorrow")]', '[book("paris", date="tomorrow")]']:
            with self.subTest(reply=reply):
                result = calls.check(reply, TOOLS, ['book'])
                self.assertTrue(result['success'])
                self.assertEqual(result['call']['args'], {'city': 'paris', 'date': 'tomorrow'})
                self.assertTrue(calls.used_value(result, 'city', 'paris'))
                self.assertFalse(calls.used_value(result, 'city', 'rome'))
        _, parsed, _, _ = calls.parse('[book("paris", "tomorrow"), book("rome", "today")]')
        self.assertEqual(call_strings(parsed), ["book('paris', 'tomorrow')", "book('rome', 'today')"])

    def test_extra_calls_fail_including_mixed_syntax(self):
        good = '[book(city="paris", date="tomorrow")]'
        for reply in [
            '[book(city="paris",date="tomorrow"), delete_everything()]',
            '[book(city="paris",date="tomorrow"), book(city="rome",date="today")]',
            good + '\n{"name":"delete_everything","parameters":{}}',
            '<tool_call>{"name":"delete_everything","arguments":{}}</tool_call>\n' + good,
            good + '\ndelete_everything(',
        ]:
            with self.subTest(reply=reply):
                result = calls.check(reply, TOOLS, ['book'])
                self.assertFalse(result['success'])
                self.assertEqual(result['invalid_reason'], 'multiple_calls')
                self.assertGreater(result['n_calls'], 1)

    def test_malformed_json_names_and_arguments_do_not_raise(self):
        for name in [[], {}, None, True, 42, '']:
            for envelope in ('llama', 'hermes'):
                obj = {'name': name, 'arguments': {}}
                reply = json.dumps(obj)
                if envelope == 'hermes': reply = '<tool_call>' + reply + '</tool_call>'
                with self.subTest(name=name, envelope=envelope):
                    result = calls.check(reply, TOOLS, ['book'])
                    self.assertTrue(result['attempt'])
                    self.assertFalse(result['success'])
                    self.assertEqual(result['invalid_reason'], 'invalid_name')
        for args in [[], 42, 'paris']:
            reply = json.dumps({'name': 'book', 'parameters': args})
            self.assertEqual(calls.check(reply, TOOLS, ['book'])['invalid_reason'], 'unparseable')

    def test_wrong_gold_tool_is_valid_but_not_successful(self):
        result = calls.check('[book(city="paris",date="tomorrow")]', TOOLS, ['another_tool'])
        self.assertTrue(result['valid'])
        self.assertFalse(result['success'])

    def test_json_payload_is_not_an_additional_call(self):
        tools = [{'name': 'store', 'parameters': {'properties': {'payload': {'type': 'object'}},
                                                'required': ['payload']}}]
        reply = '[store(payload={"name":"book","arguments":{"city":"paris"}})]'
        self.assertTrue(calls.check(reply, tools, ['store'])['success'])

    def test_human_labelled_call_attempt_boundary(self):
        folder = Path(__file__).parent/'fixtures/tool_use'
        count = 0
        for source, field in [('parser_structural.jsonl', 'parser_label'),
                              ('parser_speech_acts.jsonl', 'human')]:
            for row in read_jsonl(folder/source):
                with self.subTest(source=source, id=row.get('sid', row.get('idx'))):
                    self.assertEqual(calls.parse(row['text'])[0], row[field].startswith('tool_call'))
                    count += 1
        self.assertEqual(count, 352)


class ToolUseFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def exercise_flow(self, domain, item, target, expected_replies):
        wrapper = tiny_wrapper()
        with tempfile.TemporaryDirectory() as tmp, patch.object(paths, 'DATA_ROOT', Path(tmp)), \
             patch.object(domain, 'items', side_effect=lambda t, split='full': [item] if t == target else []), \
             patch.object(cb, 'load_model', return_value=(wrapper, wrapper.model, wrapper.tokenizer)):
            run = Run(domain, wrapper.model_id, (), (), batch=1, max_new=1, limit=1)
            run.generate('unsteered', [target])
            out = run.directory('unsteered', target, 'unsteered')
            before = (out/'generations.jsonl').read_bytes()
            row = read_jsonl(out/'generations.jsonl')[0]
            self.assertEqual(len(row.get('turns', [row])), expected_replies)
            self.assertTrue(run.generated('unsteered', target))
            with patch.object(cb, 'load_model', side_effect=AssertionError('must reuse generations')):
                run.generate('unsteered', [target])

            def judge(directories, force=False):
                for directory in directories:
                    shards = sorted(directory.glob('judge_in_*.jsonl'))
                    for shard in shards:
                        verdicts = []
                        for record in read_jsonl(shard):
                            fields = {'action': 'direct_or_other', 'coherent': True, 'appropriate': False}
                            fields['note'] = 'controlled fixture'
                            verdicts.append({'uid': record['uid'], **fields,
                                             'raw_response': json.dumps(fields)})
                        write_jsonl(shard.with_name(shard.name.replace('judge_in_', 'judge_out_')), verdicts)
                    from core.domains.tool_use_multi_turn import judge_core as multi_core
                    selected_core = judge_core if domain is tool_use else multi_core
                    write_json(directory/'judge_run.json', selected_core.run_spec(
                        [p.name for p in shards], selected_core.JUDGE_MODEL, 'vllm', {}))

            with patch.object(domain, 'judge', side_effect=judge):
                run.judge('unsteered', [target])
            run.score('unsteered', [target])
            self.assertTrue(run.scored('unsteered', target))
            judge_before = {p.name: p.read_bytes() for p in run.directory('unsteered', target).glob('judge_out_*.jsonl')}
            with patch.object(domain, 'SCORING_VERSION', domain.SCORING_VERSION + 1), patch.object(
                    domain, 'judge', side_effect=AssertionError('reuse semantic verdicts')):
                self.assertFalse(run.scored('unsteered', target))
                run.judge('unsteered', [target])
                run.score('unsteered', [target])
                self.assertTrue(run.scored('unsteered', target))
                self.assertEqual({p.name: p.read_bytes() for p in run.directory('unsteered', target).glob('judge_out_*.jsonl')}, judge_before)
            if domain is tool_use_multi_turn:
                with patch.object(domain, 'GENERATION_VERSION', domain.GENERATION_VERSION + 1):
                    self.assertFalse(run.generated('unsteered', target))
            self.assertEqual((out/'generations.jsonl').read_bytes(), before)
            meta = json.loads((out/'meta.json').read_text())
            self.assertFalse(meta['rendering']['apply_chat_template'])
            meta['turns'] = False
            write_json(out/'meta.json', meta)
            self.assertFalse(run.generated('unsteered', target))

    def test_single_decision_and_scripted_followup_complete(self):
        for target in ('call', 'clarify'):
            with self.subTest(target=target):
                gold = {'tool_names': ['book']}
                if target == 'clarify':
                    gold['followup'] = {'user_reply': 'paris', 'slot': 'city', 'value': 'paris',
                                        'tool_names': ['book'], 'action': 'call'}
                item = {'item_id': 'fixture', 'target': target, 'tools': TOOLS,
                        'messages': [{'role': 'user', 'content': 'answer red'}], 'gold': gold}
                self.exercise_flow(tool_use, item, target, 1 if target == 'call' else 2)

    @unittest.skipUnless(importlib.util.find_spec('bfcl_eval'), 'requires bfcl-eval')
    def test_multi_turn_generation_and_mechanical_scoring_complete(self):
        item = {'item_id': 'multi_turn_miss_func_fixture', 'target': 'abstain', 'category': 'miss_func',
                'tools': [], 'injections': {}, 'answer': [[], []], 'judge_turn_index': 0,
                'entry': {'id': 'multi_turn_miss_func_fixture', 'involved_classes': [], 'initial_config': {},
                          'question': [[{'role': 'user', 'content': 'answer red'}],
                                       [{'role': 'user', 'content': 'answer blue'}]]}}
        self.exercise_flow(tool_use_multi_turn, item, 'abstain', 2)


if __name__ == '__main__':
    unittest.main()
