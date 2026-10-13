"""ITI recipe cache identity and intervention across successive tool replies."""

from contextlib import contextmanager
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import repsteer as rs
from repsteer.data import ContrastivePairs
from transformers import Qwen3Config, LlamaConfig, Gemma2Config

import core as cb
from core import paths
from core.domains import instruction_following, tool_use, tool_use_multi_turn
from core.methods import METHODS
from core.runtime import generate as generation, search
from core.runtime.experiment import Run
from test_native_methods import tiny_wrapper
from test_tool_use import TOOLS


@contextmanager
def trial(domain, config_cls=Qwen3Config, strength=.5):
    base = tiny_wrapper(config_cls, num_hidden_layers=2, num_key_value_heads=1)
    model, tok, model_id = base.model, base.tokenizer, base.model_id
    model.generation_config.eos_token_id = None
    base.close()
    if domain is instruction_following:
        target = 'lowercase'
        item = {'item_id': 'fixture', 'prompt_id': 'fixture', 'base_prompt': 'answer red'}
    elif domain is tool_use:
        target = 'clarify'
        item = {'item_id': 'fixture', 'target': target, 'tools': TOOLS,
                'messages': [{'role': 'user', 'content': 'answer red'}],
                'gold': {'tool_names': ['book'], 'followup': {
                    'user_reply': 'paris', 'slot': 'city', 'value': 'paris',
                    'tool_names': ['book'], 'action': 'call'}}}
    else:
        target = 'abstain'
        item = {'item_id': 'multi_turn_miss_func_fixture', 'target': target, 'category': 'miss_func',
                'tools': [], 'injections': {}, 'answer': [[], []],
                'entry': {'id': 'multi_turn_miss_func_fixture', 'involved_classes': [], 'initial_config': {},
                          'question': [[{'role': 'user', 'content': 'answer red'}],
                                       [{'role': 'user', 'content': 'answer blue'}]]}}
    colors = ['red', 'blue', 'green', 'yellow', 'black', 'white']
    pairs = ContrastivePairs.from_pairs(
        [[{'role': 'user', 'content': 'answer '+color},
          {'role': 'assistant', 'content': 'answer red'}] for color in colors],
        [[{'role': 'user', 'content': 'answer '+color},
          {'role': 'assistant', 'content': 'answer blue'}] for color in colors])
    run = Run(domain, model_id, (), (strength,), batch=1, max_new=3,
              iti_top_k=2, iti_seed=13)
    def load_model(*args, **kwargs):
        return rs.models.from_model(model, tok, model_id=model_id, revision='test-12'), model, tok
    with tempfile.TemporaryDirectory() as tmp, patch.object(paths, 'DATA_ROOT', Path(tmp)), \
         patch.object(domain, 'pairs', return_value=pairs), patch.object(domain, 'items', return_value=[item]), \
         patch.object(cb, 'load_model', side_effect=load_model):
        run.learn('iti', [target])
        yield run, target, item, load_model


class ITIReviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_run_cache_rejects_changed_native_recipe_in_both_domains(self):
        for domain in (instruction_following, tool_use):
            with self.subTest(domain=domain.NAME), trial(domain) as (run, target, item, load_model):
                run.generate('iti', [target])
                self.assertTrue(run.generated('iti', target))
                changed = replace(METHODS['iti'], plan=lambda profile, params:
                    rs.recipes.iti(profile, strength=params['strength'], phase='both'))
                with patch.dict(METHODS, {'iti': changed}):
                    self.assertFalse(run.generated('iti', target), 'old decode-only outputs must not satisfy a changed recipe')
                    run.generate('iti', [target])
                    self.assertTrue(run.generated('iti', target))
                    out = run.directory('iti', target, run.candidates('iti')[0].id)
                    saved = json.loads((out/'meta.json').read_text())['plan']
                    profile = METHODS['iti'].load(domain, run.model, target, run.learning_keys('iti')[0], training=run.training)
                    self.assertEqual(saved, changed.plan(profile, {'strength': .5}).to_dict())
                    with patch.object(cb, 'load_model', side_effect=AssertionError('unchanged recipe should resume')):
                        run.generate('iti', [target])

    def test_candidate_cache_rejects_changed_recipe(self):
        with trial(tool_use) as (run, target, item, load_model):
            run.generate('iti', [target])
            profile = METHODS['iti'].load(tool_use, run.model, target, run.learning_keys('iti')[0], training=run.training)
            changed = lambda profile, params: rs.recipes.iti(profile, strength=params['strength'], phase='both')
            wrapper, _, _ = load_model()
            try:
                with patch.object(search, 'generate_texts', wraps=search.generate_texts) as generate:
                    out = search.generate_candidate(wrapper, tool_use, model=run.model,
                        experiment=run.experiment, method='iti', target=target,
                        cand=run.candidates('iti')[0], artifact=profile, items=[item],
                        batch_size=run.batch, max_new_tokens=run.max_new, plan_fn=changed)
                    self.assertEqual(generate.call_count, 2, 'recipe change must regenerate both replies')
                saved = json.loads((out/'meta.json').read_text())['plan']
                self.assertEqual(saved, changed(profile, {'strength': .5}).to_dict())
            finally:
                wrapper.close()

    def test_layer_methods_also_reject_changed_recipe(self):
        for method in ('caa', 'lat'):
            with self.subTest(method=method), trial(tool_use) as (run, target, item, load_model):
                run = replace(run, blocks=(0,))
                run.learn(method, [target])
                run.generate(method, [target])
                self.assertTrue(run.generated(method, target))
                original = METHODS[method]
                def decode_only(artifact, params):
                    plan = original.plan(artifact, params)
                    return rs.SteeringPlan([edit for edit in plan.interventions if edit.phase == 'decode'])
                with patch.dict(METHODS, {method: replace(original, plan=decode_only)}):
                    self.assertFalse(run.generated(method, target))
                    run.generate(method, [target])
                    self.assertTrue(run.generated(method, target))

    def check_tool_replies(self, domain):
        for config_cls in (Qwen3Config, LlamaConfig, Gemma2Config):
            with self.subTest(model=config_cls.__name__), trial(domain, config_cls, strength=20.) as (run, target, _, __):
                calls = []
                original = generation.generate
                def check_generate(wrapper, inputs, plan, **kwargs):
                    baseline = original(wrapper, inputs, None, output_scores=True, **kwargs)
                    steered = original(wrapper, inputs, plan, output_scores=True, **kwargs)
                    restored = original(wrapper, inputs, None, output_scores=True, **kwargs)
                    self.assertEqual(len(steered.scores), 3)
                    self.assertTrue(torch.equal(baseline.scores[0], steered.scores[0]))
                    self.assertFalse(torch.equal(baseline.scores[1], steered.scores[1]))
                    for left, right in zip(baseline.scores, restored.scores, strict=True):
                        self.assertTrue(torch.equal(left, right), 'hook must be removed after each reply')
                    self.assertTrue(all(edit.phase == 'decode' for edit in plan.interventions))
                    calls.append(plan.to_dict())
                    return steered
                with patch.object(generation, 'generate', side_effect=check_generate):
                    run.generate('iti', [target])
                self.assertEqual(len(calls), 2)
                self.assertEqual(calls[0], calls[1])
                self.assertTrue(run.generated('iti', target))

    def test_steering_reaches_both_scripted_replies_on_three_architectures(self):
        self.check_tool_replies(tool_use)

    @unittest.skipUnless(importlib.util.find_spec('bfcl_eval'), 'requires bfcl-eval')
    def test_steering_reaches_both_executable_replies_on_three_architectures(self):
        self.check_tool_replies(tool_use_multi_turn)


if __name__ == '__main__':
    unittest.main()
