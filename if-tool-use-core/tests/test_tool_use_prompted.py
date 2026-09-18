"""Prompted tool-use inputs, conversation continuity and baseline lifecycle."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import Qwen3Config, LlamaConfig, Gemma2Config

import control_bench as cb
from control_bench import paths
from control_bench.cli import campaign, run as cli
from control_bench.domains import tool_use, tool_use_multi_turn
from control_bench.domains.tool_use import judge_core
from control_bench.domains.tool_use.prompting import TARGET_INSTRUCTIONS
from control_bench.domains.tool_use.render import conversation, tool_block
from control_bench.domains.tool_use.multi_turn import driver, judging
from control_bench.domains.tool_use_multi_turn import judge_core as multi_core
from control_bench.runtime.experiment import Run
from control_bench.runtime.search import input_fingerprint, read_jsonl, write_json, write_jsonl
from test_native_methods import tiny_wrapper
import test_sampling
from test_tool_use import TOOLS

HAS_BFCL = importlib.util.find_spec('bfcl_eval') is not None


class MessageTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages)


def single_item(target='clarify'):
    followup = {'user_reply': 'Use paris.', 'slot': 'city', 'value': 'paris',
                'tool_names': ['book'], 'action': 'call'}
    if target == 'abstain':
        followup['restored_tool'] = TOOLS[0]
    return {'item_id': 'fixture', 'target': target,
            'tools': [] if target == 'abstain' else TOOLS,
            'messages': [{'role': 'user', 'content': 'Please book for tomorrow.'}],
            'gold': {'tool_names': ['book'], 'followup': followup}}


def multi_item():
    return {'item_id': 'multi_turn_miss_func_prompted_fixture', 'target': 'abstain',
            'category': 'miss_func', 'tools': [], 'injections': {}, 'answer': [[], []],
            'judge_turn_index': 0,
            'entry': {'id': 'multi_turn_miss_func_prompted_fixture', 'involved_classes': [],
                      'initial_config': {},
                      'question': [[{'role': 'user', 'content': 'answer red'}],
                                   [{'role': 'user', 'content': 'answer blue'}]]}}


class ToolUsePromptedTest(unittest.TestCase):
    dry_run = test_sampling.SamplingTest.dry_run

    def test_cli_baseline_and_mixed_grid_support_both_domains(self):
        for domain in (tool_use, tool_use_multi_turn):
            args = ['cbench-campaign', '--domain', domain.NAME, '--gpus', '0',
                    '--methods', 'prompted', '--sample', '2']
            result = self.dry_run(campaign, args)
            self.assertEqual(result['methods']['prompted'],
                             {'layers': [], 'strengths': [], 'configurations_per_target': 1,
                              'outputs': 6})
            run = Run(domain, 'm', (), (), sample=2)
            with patch.object(domain, 'pairs', side_effect=AssertionError('must not fit')):
                self.assertTrue(run.learned('prompted', 'call'))
                with self.assertRaisesRegex(ValueError, 'no learning step'):
                    run.learn('prompted', ['call'])
                for stage in ('generate', 'judge', 'score'):
                    argv = campaign.child_command(run, stage, 'prompted', ['call'])
                    self.assertNotIn('--layers', argv)
                    self.assertFalse(any(a.startswith('--strengths') for a in argv))
                    self.assertEqual(self.dry_run(cli, ['cbench-run', *argv[3:]]),
                                     run.description(['prompted'], ['call']))
            mixed = self.dry_run(campaign, ['cbench-campaign', '--domain', domain.NAME,
                                 '--gpus', '0', '--sample', '2', '--methods',
                                 'unsteered,prompted,caa,lat', '--layers', '9,18,27',
                                 '--strengths', '.5,1,1.5,2,2.5'])
            self.assertEqual({m: v['configurations_per_target']
                              for m, v in mixed['methods'].items()},
                             {'unsteered': 1, 'prompted': 1, 'caa': 15, 'lat': 15})

    def test_frozen_items_and_other_methods_keep_original_prompts(self):
        tok = MessageTokenizer()
        for domain in (tool_use, tool_use_multi_turn):
            run = Run(domain, 'm', (9,), (1.,), sample=2)
            for target in domain.TARGETS:
                originals = copy.deepcopy(run.items(target))
                prompted = run.generation_items('prompted', target)
                self.assertEqual([r['item_id'] for r in prompted],
                                 [r['item_id'] for r in originals])
                for original, item in zip(originals, prompted):
                    text = domain.render(tok, item)
                    self.assertEqual(text.count(TARGET_INSTRUCTIONS[target]), 1)
                    messages = (original['messages'] if domain is tool_use
                                else original['entry']['question'][0])
                    expected = conversation(messages, original['tools'])
                    self.assertEqual(expected[0]['content'],
                                     tool_block(original['tools']) + '\n' + messages[0]['content'])
                    self.assertEqual(domain.render(tok, original), json.dumps(expected))
                    hidden = {**item, 'gold': {'secret': 'GOLD_SECRET'},
                              'answer': ['REFERENCE_SECRET'], 'judge_turn_index': 999}
                    self.assertEqual(domain.render(tok, hidden), text)
                for method in ('unsteered', 'caa', 'lat', 'linear_probe', 'pca', 'iti'):
                    self.assertEqual(run.generation_items(method, target), originals)
                self.assertEqual(run.items(target), originals)
                self.assertNotEqual(input_fingerprint(prompted), input_fingerprint(originals))

    def test_single_followup_keeps_instruction_once_and_restores_tool(self):
        tok = MessageTokenizer()
        for target in ('clarify', 'abstain'):
            original = single_item(target)
            before = copy.deepcopy(original)
            item, = tool_use.prompted_items([original])
            initial = tool_use.render(tok, item)
            followup = tool_use.next_turn(tok, item, ['Please provide what is missing.'])
            self.assertEqual(followup.count(TARGET_INSTRUCTIONS[target]), 1)
            self.assertIn('Use paris.', followup)
            self.assertIn('book', followup)
            if target == 'abstain':
                self.assertNotIn('book', initial.split('Please book')[0])
                self.assertIn('required', followup)
            self.assertNotIn('generation_instruction', original)
            self.assertEqual(original, before)
            self.assertIsNone(tool_use.next_turn(tok, item, ['Question?', '[book(city="paris")]']))

    @unittest.skipUnless(HAS_BFCL, 'requires bfcl-eval')
    def test_multi_turn_instruction_survives_tool_results_and_restoration(self):
        original = multi_item()
        original['tools'] = TOOLS
        original['answer'] = [['REFERENCE_SECRET'], [], []]
        original['judge_turn_index'] = 1
        original['entry']['question'] = [
            [{'role': 'user', 'content': 'Book paris for tomorrow.'}],
            [{'role': 'user', 'content': 'Now use a missing tool.'}],
            [{'role': 'user', 'content': 'It is available now.'}]]
        original['injections'] = {'2': [{'name': 'FUTURE_TOOL', 'parameters': {}}]}
        before = copy.deepcopy(original)
        item, = tool_use_multi_turn.prompted_items([original])
        tok = MessageTokenizer()
        replies = []
        with patch.dict(driver._STATE, {}, clear=True), patch(
                'bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils.execute_multi_turn_func_call',
                return_value=(['OBSERVED_RESULT'], {})) as execute:
            rendered = [tool_use_multi_turn.render(tok, item)]
            for reply in ('[book(city="paris", date="tomorrow")]', 'Done.', 'No suitable tool.'):
                replies.append(reply)
                rendered.append(driver.next_turn(tok, item, replies))
            self.assertIn('OBSERVED_RESULT', rendered[1])
            self.assertTrue(all('FUTURE_TOOL' not in text for text in rendered[:3]))
            self.assertIn('FUTURE_TOOL', rendered[3])
            for text in rendered:
                self.assertEqual(text.count(TARGET_INSTRUCTIONS['abstain']), 1)
                self.assertNotIn('REFERENCE_SECRET', text)
            execute.assert_called_once()
            self.assertNotIn(TARGET_INSTRUCTIONS['abstain'],
                             json.dumps(driver.state_of(item)['history']))
            outputs = [{'item_id': item['item_id'],
                        'turns': [{'output': reply} for reply in replies],
                        **driver.generation_record(item, replies)}]
            row, = judging.judge_rows('abstain', 'full', [original], outputs, 'prompted')
            self.assertNotIn(TARGET_INSTRUCTIONS['abstain'], multi_core.build_prompt(row))
        self.assertEqual(original, before)

    @unittest.skipUnless(HAS_BFCL, 'requires bfcl-eval')
    def test_real_tiny_models_generate_judge_score_and_resume(self):
        torch.set_num_threads(2)
        for config_cls in (Qwen3Config, LlamaConfig, Gemma2Config):
            for domain, item in ((tool_use, single_item()), (tool_use_multi_turn, multi_item())):
                with self.subTest(model=config_cls.__name__, domain=domain.NAME):
                    target, core = item['target'], judge_core if domain is tool_use else multi_core
                    wrapper = tiny_wrapper(config_cls)
                    run = Run(domain, wrapper.model_id, (), (), batch=1, max_new=1, limit=1)
                    seen = []

                    def judge(directories, force=False):
                        for directory in directories:
                            shards = sorted(directory.glob('judge_in_*.jsonl'))
                            for shard in shards:
                                records = read_jsonl(shard)
                                seen.extend(records)
                                verdicts = []
                                for record in records:
                                    fields = {'action': 'direct_or_other', 'coherent': True,
                                              'appropriate': False, 'note': 'controlled fixture'}
                                    verdicts.append({'uid': record['uid'], **fields,
                                                     'raw_response': json.dumps(fields)})
                                write_jsonl(shard.with_name(shard.name.replace('judge_in_', 'judge_out_')),
                                            verdicts)
                            write_json(directory / 'judge_run.json', core.run_spec(
                                [p.name for p in shards], core.JUDGE_MODEL, 'vllm', {}))

                    with tempfile.TemporaryDirectory() as tmp, \
                         patch.object(paths, 'DATA_ROOT', Path(tmp)), \
                         patch.object(domain, 'items', return_value=[item]), \
                         patch.object(domain, 'pairs', side_effect=AssertionError('must not fit')), \
                         patch.object(cb, 'load_model', return_value=(wrapper, wrapper.model, wrapper.tokenizer)), \
                         patch.object(wrapper, 'generate', wraps=wrapper.generate) as generate, \
                         patch.object(domain, 'judge', side_effect=judge):
                        run.generate('prompted', [target])
                        self.assertTrue(run.generated('prompted', target))
                        self.assertEqual(generate.call_count, 2)
                        for call in generate.call_args_list:
                            self.assertEqual(call.args[0][0].count(TARGET_INSTRUCTIONS[target]), 1)
                        out = run.directory('prompted', target, 'prompted')
                        meta = json.loads((out / 'meta.json').read_text())
                        self.assertIsNone(meta['plan'])
                        self.assertIsNone(meta['artifact_fingerprint'])
                        self.assertEqual(meta['input_fingerprint'],
                                         input_fingerprint(run.generation_items('prompted', target)))
                        run.judge('prompted', [target])
                        run.score('prompted', [target])
                        self.assertTrue(run.scored('prompted', target))
                        self.assertTrue(seen)
                        self.assertTrue(all(TARGET_INSTRUCTIONS[target] not in core.build_prompt(row)
                                            for row in seen))
                        with patch.object(cb, 'load_model', side_effect=AssertionError('must resume')):
                            run.generate('prompted', [target])
                        with patch.dict(TARGET_INSTRUCTIONS, {target: 'Changed instruction.'}):
                            self.assertFalse(run.generated('prompted', target))
                        self.assertTrue(run.generated('prompted', target))
                        self.assertNotEqual(out, run.directory('unsteered', target, 'unsteered'))
                        self.assertFalse(run.generated('unsteered', target))


if __name__ == '__main__':
    unittest.main()
