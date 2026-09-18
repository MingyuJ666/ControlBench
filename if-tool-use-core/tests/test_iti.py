"""Native ITI profiles, decoding, CLI scope and cache identity."""

import io
import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import repsteer as rs
from repsteer.artifacts import ITIArtifact
from repsteer.data import ContrastivePairs
from transformers import Qwen3Config, LlamaConfig, Gemma2Config

import control_bench as cb
from control_bench import paths
from control_bench.cli import campaign, run as cli
from control_bench.domains import instruction_following as domain, tool_use, tool_use_multi_turn
from control_bench.methods import METHODS, iti
from control_bench.methods.learn import artifact_key
from control_bench.runtime.experiment import Run
from test_native_methods import tiny_wrapper, training_pairs
from test_tool_use import TOOLS
from control_bench.runtime.search import read_jsonl
import test_sampling


class ITITest(unittest.TestCase):
    dry_run = test_sampling.SamplingTest.dry_run

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_native_profiles_roundtrip_generate_and_resume_on_three_architectures(self):
        for config_cls in (Qwen3Config, LlamaConfig, Gemma2Config):
            with self.subTest(model=config_cls.__name__):
                base = tiny_wrapper(config_cls, num_hidden_layers=2)
                model, tok, model_id = base.model, base.tokenizer, base.model_id
                base.close()
                pairs = training_pairs()
                train_n = None if config_cls is Qwen3Config else 4
                run = Run(domain, model_id, (), (.5, 1.), batch=2, max_new=3,
                          train_sample=train_n, train_seed=7, iti_top_k=2, iti_seed=13)
                def load_model(*args, **kwargs):
                    return rs.models.from_model(model, tok, model_id=model_id, revision='test-12'), model, tok
                items = [{'item_id': f'item-{i}', 'prompt_id': f'item-{i}', 'base_prompt': f'answer {c}'}
                         for i, c in enumerate(['red', 'blue'])]
                with tempfile.TemporaryDirectory() as tmp, patch.object(paths, 'DATA_ROOT', Path(tmp)), \
                     patch.object(domain, 'pairs', return_value=pairs), patch.object(domain, 'items', return_value=items), \
                     patch.object(cb, 'load_model', side_effect=load_model):
                    self.assertFalse(run.learned('iti', 'lowercase'))
                    run.learn('iti', ['lowercase'])
                    self.assertTrue(run.learned('iti', 'lowercase'))
                    key = run.learning_keys('iti')[0]
                    self.assertEqual(key, 'heads_k2_seed13')
                    restored = METHODS['iti'].load(domain, model_id, 'lowercase', key, training=run.training)
                    self.assertIsInstance(restored, ITIArtifact)
                    self.assertIsNone(restored.metadata.site)
                    self.assertEqual(len(restored.heads), 2)
                    self.assertEqual(restored.metadata.config['num_layers'], 2)
                    self.assertEqual(artifact_key(restored), key)
                    self.assertEqual(restored.metadata.config['iti_data_fingerprint'], run.training_pairs('lowercase').fingerprint)
                    wrapper, _, _ = load_model()
                    try:
                        expected = rs.learners.ITI(top_k=2, seed=13, batch_size=2, capture_dtype='float32',
                                                   **domain.rendering().learner_kwargs()).fit(wrapper, run.training_pairs('lowercase'))
                    finally:
                        wrapper.close()
                    self.assertTrue(torch.equal(restored.directions, expected.directions))
                    self.assertEqual(restored.heads, expected.heads)
                    plan = METHODS['iti'].plan(restored, {'strength': .5})
                    self.assertEqual(plan.to_dict(), rs.recipes.iti(restored, strength=.5).to_dict())
                    self.assertTrue(all(i.phase == 'decode' for i in plan.interventions))
                    self.assertTrue(all(isinstance(i.operator, rs.operators.ITIAdd) for i in plan.interventions))
                    run.generate('iti', ['lowercase'])
                    self.assertTrue(run.generated('iti', 'lowercase'))
                    for cand in run.candidates('iti'):
                        out = run.directory('iti', 'lowercase', cand.id)
                        meta = json.loads((out / 'meta.json').read_text())
                        self.assertEqual(meta['artifact_fingerprint'], restored.fingerprint())
                        self.assertEqual(meta['candidate']['parameters']['strength_unit'], 'iti_alpha')
                    with patch.object(cb, 'load_model', side_effect=AssertionError('must reuse completed profile and generations')):
                        run.learn('iti', ['lowercase'])
                        run.generate('iti', ['lowercase'])
                    other = Run(domain, model_id, (), (.5,), train_sample=train_n, train_seed=7, iti_top_k=2, iti_seed=14)
                    self.assertFalse(other.learned('iti', 'lowercase'))
                    changed = ContrastivePairs.from_records(pairs.records, metadata={'changed': True})
                    with patch.object(domain, 'pairs', return_value=changed):
                        rerun = Run(domain, model_id, (), (.5,), train_sample=train_n, train_seed=7, iti_top_k=2, iti_seed=13)
                        self.assertFalse(rerun.learned('iti', 'lowercase'))
                        self.assertFalse(rerun.generated('iti', 'lowercase'))

    def test_recipe_changes_decode_logits_but_preserves_prefill(self):
        wrapper = tiny_wrapper(num_hidden_layers=2)
        wrapper.model.generation_config.eos_token_id = None
        try:
            profile = rs.learners.ITI(top_k=2, seed=13, batch_size=2, capture_dtype='float32',
                                     **domain.rendering().learner_kwargs()).fit(wrapper, training_pairs())
            chat = [[{'role': 'user', 'content': 'answer red'}]]
            texts = wrapper.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            logits = []
            hook = wrapper.model.register_forward_hook(lambda m, args, out: logits.append(out.logits[:, -1].detach().clone()))
            try:
                cb.generate_texts(wrapper, texts, batch_size=1, max_new_tokens=3,
                                  messages=chat, rendering=domain.rendering())
                baseline = list(logits)
                logits.clear()
                cb.generate_texts(wrapper, texts, iti.plan(profile, {'strength': 20.}), batch_size=1,
                                  max_new_tokens=3, messages=chat, rendering=domain.rendering())
                self.assertEqual(len(logits), 3)
                self.assertEqual(len(baseline), 3)
                self.assertTrue(torch.equal(logits[0], baseline[0]))
                self.assertFalse(torch.equal(logits[1], baseline[1]))
            finally:
                hook.remove()
        finally:
            wrapper.close()

    def test_grouped_query_zero_alpha_and_hook_cleanup_on_three_architectures(self):
        for config_cls in (Qwen3Config, LlamaConfig, Gemma2Config):
            with self.subTest(model=config_cls.__name__):
                wrapper = tiny_wrapper(config_cls, num_hidden_layers=2, num_key_value_heads=1)
                wrapper.model.generation_config.eos_token_id = None
                try:
                    profile = METHODS['iti'].learn(wrapper, domain, 'lowercase', batch_size=2,
                                                   pairs=training_pairs(), top_k=2, seed=13)['heads_k2_seed13']
                    chat = [[{'role': 'user', 'content': text}]
                            for text in ['answer red', 'answer blue green']]
                    def generate(plan=None):
                        return cb.generate(wrapper, chat, plan, max_new_tokens=3, output_scores=True,
                                           **domain.rendering().generation_kwargs())
                    baseline = generate()
                    zero = generate(METHODS['iti'].plan(profile, {'strength': 0.}))
                    self.assertTrue(torch.equal(baseline.sequences, zero.sequences))
                    for left, right in zip(baseline.scores, zero.scores, strict=True):
                        self.assertTrue(torch.equal(left, right))
                    steered = generate(METHODS['iti'].plan(profile, {'strength': 5.}))
                    self.assertTrue(torch.equal(baseline.scores[0], steered.scores[0]))
                    self.assertFalse(torch.equal(baseline.scores[1], steered.scores[1]))
                    restored = generate()
                    self.assertTrue(torch.equal(baseline.sequences, restored.sequences))
                    for left, right in zip(baseline.scores, restored.scores, strict=True):
                        self.assertTrue(torch.equal(left, right))
                finally:
                    wrapper.close()

    def test_cli_defaults_and_mixed_campaign_counts(self):
        args = ['cbench-run', 'learn', '--method', 'iti', '--targets', 'lowercase', '--strengths', '.5,1',
                '--train-sample', '100', '--sample', '8']
        result = self.dry_run(cli, args)
        settings = result['methods']['iti']
        self.assertEqual(settings['top_k'], 48)
        self.assertEqual(settings['seed'], 42)
        self.assertEqual(settings['layers'], [])
        self.assertEqual(settings['layer_selection'], 'all_layers')
        self.assertEqual(settings['configurations_per_target'], 2)
        self.assertEqual(settings['pairs_per_target'], {'lowercase': 100})
        mixed = self.dry_run(campaign, ['cbench-campaign', '--gpus', '0,1', '--methods', 'unsteered,caa,lat,iti',
                                       '--layers', '9,12', '--strengths', '.5,1', '--sample', '8',
                                       '--targets', 'lowercase', '--iti-top-k', '16', '--iti-seed', '7'])
        self.assertEqual({m: v['configurations_per_target'] for m, v in mixed['methods'].items()},
                         {'unsteered': 1, 'caa': 4, 'lat': 4, 'iti': 2})
        self.assertEqual(mixed['methods']['iti']['top_k'], 16)
        self.assertEqual(mixed['methods']['iti']['seed'], 7)

    def test_campaign_forwards_iti_options_only_to_iti_workers(self):
        run = Run(domain, 'model', (9, 12), (.5, 1.), sample=3, train_sample=8,
                  iti_top_k=16, iti_seed=0)
        for method in ('unsteered', 'caa', 'lat', 'iti'):
            for stage in ('generate', 'judge', 'score') if method == 'unsteered' else ('learn', 'generate', 'judge', 'score'):
                with self.subTest(method=method, stage=stage):
                    argv = campaign.child_command(run, stage, method, ['lowercase'])
                    self.assertEqual('--iti-top-k' in argv, method == 'iti')
                    self.assertEqual('--layers' in argv, method in ('caa', 'lat'))
                    self.assertEqual(self.dry_run(cli, ['cbench-run', *argv[3:]]), run.description([method], ['lowercase']))

    def test_cli_rejects_misleading_options_without_loading(self):
        cases = [(['--method', 'iti'], 'requires --strengths'),
                 (['--method', 'iti', '--strengths', '1', '--layers', '9'], 'omit --layers'),
                 (['--method', 'iti', '--strengths', '1', '--iti-top-k', '0'], 'positive integer'),
                 (['--method', 'iti', '--strengths', '1', '--train-sample', '1'], 'at least 2'),
                 (['--method', 'unsteered', '--iti-top-k', '48'], 'require iti')]
        for args, message in cases:
            with self.subTest(args=args), patch.object(sys, 'argv', ['cbench-run', 'generate', *args]), \
                 redirect_stderr(io.StringIO()) as errors, \
                 patch.object(cb, 'load_model', side_effect=AssertionError('must not load weights')):
                with self.assertRaises(SystemExit) as raised:
                    cli.main()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(message, errors.getvalue())
        unsupported = SimpleNamespace(NAME='unsupported')
        with self.assertRaisesRegex(ValueError, 'not supported'):
            METHODS['iti'].learn(None, unsupported, 'call')
        with self.assertRaisesRegex(ValueError, 'not supported'):
            Run(unsupported, 'model', (), (1.,)).learn('iti', ['call'])

    def test_tool_use_iti_cli_samples_200_per_axis(self):
        for dom in (tool_use, tool_use_multi_turn):
            with self.subTest(domain=dom.NAME):
                self.assertTrue(METHODS['iti'].supports(dom.NAME))
                result = self.dry_run(cli, ['cbench-run', 'generate', '--domain', dom.NAME,
                    '--method', 'iti', '--strengths', '.5,1,1.5,2,2.5,3',
                    '--sample', '200', '--seed', '42', '--train-sample', '100'])
                self.assertEqual(result['items_per_target'], dict.fromkeys(dom.TARGETS, 200))
                self.assertEqual(result['methods']['iti']['outputs'], 3600)
                self.assertEqual(result['methods']['iti']['pairs_per_target'], dict.fromkeys(dom.TARGETS, 100))
                self.assertEqual(result['methods']['iti']['layer_selection'], 'all_layers')
                for target in dom.TARGETS:
                    pairs = dom.pairs(target)
                    for side in (pairs.positive_examples, pairs.negative_examples):
                        self.assertTrue(all(example.input[-1]['role'] == 'assistant' for example in side))

    def exercise_tool_profile(self, dom, target, item):
        base = tiny_wrapper(num_hidden_layers=2)
        model, tok, model_id = base.model, base.tokenizer, base.model_id
        base.close()
        colors = ['red', 'blue', 'green', 'yellow', 'black', 'white']
        pairs = ContrastivePairs.from_pairs(
            [[{'role': 'user', 'content': 'answer '+color},
              {'role': 'assistant', 'content': 'answer red'}] for color in colors],
            [[{'role': 'user', 'content': 'answer '+color},
              {'role': 'assistant', 'content': 'answer blue'}] for color in colors])
        run = Run(dom, model_id, (), (.5,), batch=2, max_new=3, sample=1,
                  iti_top_k=2, iti_seed=13)
        def load_model(*args, **kwargs):
            return rs.models.from_model(model, tok, model_id=model_id, revision='test-12'), model, tok
        with tempfile.TemporaryDirectory() as tmp, patch.object(paths, 'DATA_ROOT', Path(tmp)), \
             patch.object(dom, 'pairs', return_value=pairs), patch.object(dom, 'items', return_value=[item]), \
             patch.object(cb, 'load_model', side_effect=load_model):
            with patch.object(rs.learners, 'ITI', wraps=rs.learners.ITI) as learner:
                run.learn('iti', [target])
                self.assertFalse(learner.call_args.kwargs['add_generation_prompt'])
                self.assertFalse(learner.call_args.kwargs['chat_template_kwargs']['enable_thinking'])
            self.assertTrue(run.learned('iti', target))
            profile = METHODS['iti'].load(dom, model_id, target, 'heads_k2_seed13', training=run.training)
            self.assertEqual(profile.metadata.config['domain'], dom.NAME)
            self.assertEqual(len(profile.heads), 2)
            run.generate('iti', [target])
            self.assertTrue(run.generated('iti', target))
            out = run.directory('iti', target, run.candidates('iti')[0].id)
            rows = read_jsonl(out/'generations.jsonl')
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(rows[0]['turns']), 2)
            meta = json.loads((out/'meta.json').read_text())
            self.assertEqual(meta['artifact_fingerprint'], profile.fingerprint())
            with patch.object(cb, 'load_model', side_effect=AssertionError('must reuse profile and generations')):
                run.learn('iti', [target])
                run.generate('iti', [target])

    def test_tool_use_iti_profile_generates_scripted_followup_and_resumes(self):
        item = {'item_id': 'fixture', 'target': 'clarify', 'tools': TOOLS,
                'messages': [{'role': 'user', 'content': 'answer red'}],
                'gold': {'tool_names': ['book'], 'followup': {
                    'user_reply': 'paris', 'slot': 'city', 'value': 'paris',
                    'tool_names': ['book'], 'action': 'call'}}}
        self.exercise_tool_profile(tool_use, 'clarify', item)

    @unittest.skipUnless(importlib.util.find_spec('bfcl_eval'), 'requires bfcl-eval')
    def test_tool_use_iti_profile_generates_executable_conversation_and_resumes(self):
        item = {'item_id': 'multi_turn_miss_func_fixture', 'target': 'abstain', 'category': 'miss_func',
                'tools': [], 'injections': {}, 'answer': [[], []],
                'entry': {'id': 'multi_turn_miss_func_fixture', 'involved_classes': [], 'initial_config': {},
                          'question': [[{'role': 'user', 'content': 'answer red'}],
                                       [{'role': 'user', 'content': 'answer blue'}]]}}
        self.exercise_tool_profile(tool_use_multi_turn, 'abstain', item)

    def test_profile_identity_separates_heads_seed_and_training_selection(self):
        runs = [Run(domain, 'model', (), (1.,)),
                Run(domain, 'model', (), (1.,), iti_top_k=16),
                Run(domain, 'model', (), (1.,), iti_seed=7),
                Run(domain, 'model', (), (1.,), train_sample=100)]
        self.assertEqual(len({r.artifact_directory('iti', 'lowercase', r.learning_keys('iti')[0]) for r in runs}), 4)
        self.assertEqual(len({r.directory('iti', 'lowercase', r.candidates('iti')[0].id) for r in runs}), 4)
        for r in runs[1:]:
            self.assertNotEqual(r.score_fingerprint('iti', 'lowercase'), runs[0].score_fingerprint('iti', 'lowercase'))


if __name__ == '__main__':
    unittest.main()
