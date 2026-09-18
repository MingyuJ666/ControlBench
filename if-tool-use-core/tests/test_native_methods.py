"""Offline integration checks for native repsteer methods (no model downloads).

Run: python -m unittest discover -s tests -v
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import repsteer as rs
from repsteer.artifacts import DirectionArtifact, ProbeArtifact, SubspaceArtifact
from repsteer.data import ContrastivePairs
from repsteer.positions import LastNonPaddingToken
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3Config, LlamaConfig, Gemma2Config

import control_bench as cb
from control_bench import paths
from control_bench.methods import METHODS
from control_bench.runtime.search import candidate, generate_candidate


def tiny_wrapper(config_cls=Qwen3Config, num_hidden_layers=1, num_key_value_heads=2):
    words = ['<pad>', '<bos>', '<eos>', '<unk>', 'user', 'assistant', 'answer',
             'lowercase', 'red', 'blue', 'green', 'yellow', 'black', 'white']
    vocab = {word: i for i, word in enumerate(words)}
    backend = Tokenizer(WordLevel(vocab, unk_token='<unk>'))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single='<bos> $A', special_tokens=[('<bos>', 1)])
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='<pad>',
                                 bos_token='<bos>', eos_token='<eos>', unk_token='<unk>',
                                 padding_side='left')
    tok.chat_template = ("{{ bos_token }}{% for m in messages %}{{ m['role'] }} "
                         "{{ m['content'] }} {{ eos_token }}{% endfor %}"
                         "{% if add_generation_prompt %}assistant {% endif %}")
    cfg = config_cls(vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
                     num_hidden_layers=num_hidden_layers, num_attention_heads=2, num_key_value_heads=num_key_value_heads,
                     head_dim=8, max_position_embeddings=128,
                     bos_token_id=1, eos_token_id=2, pad_token_id=0)
    torch.manual_seed(12)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16, attn_implementation='eager')
    model.eval()
    return rs.models.from_model(model, tok, model_id='tiny-'+cfg.model_type, revision='test-12')


def messages(text):
    return [{'role': 'user', 'content': text}]


def training_pairs(weights=None):
    colors = ['red', 'blue', 'green', 'yellow', 'black', 'white']
    return ContrastivePairs.from_pairs(
        [messages('answer '+color+' lowercase') for color in colors],
        [messages('answer '+color) for color in colors], weights=weights)


def make_domain(pairs):
    return SimpleNamespace(NAME='test', pairs=lambda target: pairs,
                           rendering=lambda: cb.Rendering.chat(enable_thinking=False))


class NativeMethodsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_same_artifact(self, expected, actual):
        self.assertIs(type(actual), type(expected))
        self.assertEqual(set(actual.tensors()), set(expected.tensors()))
        for key, tensor in expected.tensors().items():
            self.assertTrue(torch.equal(tensor, actual.tensors()[key]), key)

    def test_native_learners_roundtrip_and_generation_on_three_architectures(self):
        cases = {
            'caa': (rs.learners.CAA, DirectionArtifact, {'normalize': 'l2'}),
            'linear_probe': (rs.learners.LinearProbe, ProbeArtifact, {'normalize': 'l2'}),
            'lat': (rs.learners.LAT, DirectionArtifact, {'normalize': 'l2'}),
            'pca': (rs.learners.PCA, SubspaceArtifact, {'n_components': 1, 'center': True}),
        }
        pairs = training_pairs(weights=[1, 2, 3, 4, 5, 6])
        dom, site = make_domain(pairs), rs.sites.resid_post(0)
        with tempfile.TemporaryDirectory() as tmp, patch.object(paths, 'DATA_ROOT', Path(tmp)):
            for config_cls in (Qwen3Config, LlamaConfig, Gemma2Config):
                wrapper = tiny_wrapper(config_cls)
                self.assertEqual(type(wrapper.adapter).__module__.split('.')[0], 'repsteer')
                try:
                    for name, (learner_cls, expected_type, kwargs) in cases.items():
                        with self.subTest(model=config_cls.__name__, method=name):
                            expected = learner_cls(
                                site=site, positions=LastNonPaddingToken(), batch_size=3,
                                capture_dtype='float32', **dom.rendering().learner_kwargs(), **kwargs,
                            ).fit(wrapper, pairs)
                            actual = METHODS[name].learn(wrapper, dom, 'target', [site], batch_size=3)['resid_post_0']
                            self.assertIs(type(actual), expected_type)
                            self.assert_same_artifact(expected, actual)
                            dest = cb.save_artifact(actual, domain=dom.NAME, model=wrapper.model_id,
                                                    method=name, target='target')
                            restored = METHODS[name].load(dom, wrapper.model_id, 'target', 'resid_post_0')
                            self.assert_same_artifact(actual, restored)
                            self.assertEqual(restored.metadata.config['pipeline'], cb.PIPELINE_ID)
                            self.assertTrue((dest/'checksums.json').exists())
                            with self.assertRaises(FileExistsError):
                                cb.save_artifact(actual, domain=dom.NAME, model=wrapper.model_id,
                                                 method=name, target='target')
                            plan = METHODS[name].plan(restored, {'strength': -0.4 if name == 'pca' else 0.4})
                            chat = [messages('answer red')]
                            texts = wrapper.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                            rows = cb.generate_texts(wrapper, texts, plan, batch_size=1, max_new_tokens=2,
                                                     messages=chat, rendering=dom.rendering())
                            self.assertEqual(len(rows), 1)
                            self.assertLessEqual(rows[0]['n_new_tokens'], 2)
                finally:
                    wrapper.close()

    def test_negative_candidate_really_applies_steering(self):
        wrapper = tiny_wrapper()
        dom, site = make_domain(training_pairs()), rs.sites.resid_post(0)
        artifact = METHODS['pca'].learn(wrapper, dom, 'target', [site], batch_size=3)['resid_post_0']
        dom.render = lambda tok, item: tok.apply_chat_template(messages(item['text']), tokenize=False, add_generation_prompt=True)
        items = [{'item_id': 'a', 'text': 'answer red'}]
        chat = [messages(items[0]['text'])]
        texts = [dom.render(wrapper.tokenizer, items[0])]
        logits = []
        hook = wrapper.model.register_forward_hook(lambda m, a, out: logits.append(out.logits[:, -1].detach().clone()))
        try:
            cb.generate_texts(wrapper, texts, batch_size=1, max_new_tokens=2)
            unsteered = logits[0]
            logits.clear()
            expected = cb.generate_texts(wrapper, texts, METHODS['pca'].plan(artifact, {'strength': -0.4}),
                                         batch_size=1, max_new_tokens=2, messages=chat, rendering=dom.rendering())
            expected_logits = list(logits)
            logits.clear()
            with tempfile.TemporaryDirectory() as tmp, patch.object(paths, 'DATA_ROOT', Path(tmp)):
                cand = candidate('resid_post_0', -0.4)
                # A legacy completion with the same candidate id must not be reused.
                legacy = Path(tmp)/dom.NAME/'runs'/paths.model_slug(wrapper.model_id)/'calibration'/'pca'/'target'/cand.id
                legacy.mkdir(parents=True)
                (legacy/'generations.jsonl').write_text('legacy result')
                out = generate_candidate(wrapper, dom, model=wrapper.model_id, experiment='calibration', method='pca',
                                         target='target', cand=cand, artifact=artifact, items=items, batch_size=1,
                                         max_new_tokens=2, plan_fn=METHODS['pca'].plan,
                                         messages_fn=lambda item: messages(item['text']), log=lambda _: None)
                meta = json.loads((out/'meta.json').read_text())
                actual = json.loads((out/'generations.jsonl').read_text())
                self.assertIsNotNone(meta['plan'])
                self.assertEqual(meta['pipeline'], cb.PIPELINE_ID)
                self.assertEqual(actual['output'], expected[0]['output'])
                self.assertEqual(len(expected_logits), len(logits))
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected_logits, logits)))
                self.assertFalse(torch.equal(unsteered, logits[0]))
                self.assertIn(cb.PIPELINE_ID, out.parts)
                self.assertEqual((legacy/'generations.jsonl').read_text(), 'legacy result')
        finally:
            hook.remove()
            wrapper.close()

    def test_enabled_methods_require_an_explicit_grid(self):
        self.assertEqual(set(METHODS), {'caa', 'linear_probe', 'lat', 'pca', 'iti'})
        with self.assertRaises(TypeError):
            METHODS['caa'].candidates(['resid_post_0'])
        strengths = {c.parameters['strength'] for c in METHODS['pca'].candidates(['resid_post_0'], [-1., 1.])}
        self.assertEqual(strengths, {-1., 1.})


if __name__ == '__main__': unittest.main()
