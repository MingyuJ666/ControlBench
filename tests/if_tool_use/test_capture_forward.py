"""Check that capture optimization preserves learned signals and normal forwards."""
import unittest
import torch
import repsteer as rs
from repsteer.capture import CaptureRequest
from repsteer.positions import LastNonPaddingToken
from transformers import Qwen3Config, LlamaConfig, Gemma2Config
from test_native_methods import tiny_wrapper, training_pairs
from core.runtime.models import capture_forward

class CaptureForwardTest(unittest.TestCase):
    def test_capture_and_caa_unchanged_on_three_architectures(self):
        torch.set_num_threads(2)
        for architecture in (Qwen3Config, LlamaConfig, Gemma2Config):
            with self.subTest(model=architecture.__name__):
                wrapper = tiny_wrapper(architecture, num_hidden_layers=2)
                try:
                    pairs = training_pairs()
                    inputs = [example.input for example in pairs.positive_examples[:2]]
                    for site in (rs.sites.resid_post(0), rs.sites.resid_post(1), rs.sites.head_result(1)):
                        request = CaptureRequest(inputs=inputs, site=site, positions=LastNonPaddingToken(),
                                                 apply_chat_template=True, dtype='float32')
                        expected = wrapper.capture(request).activations
                        with capture_forward(wrapper.model):
                            actual = wrapper.capture(request).activations
                        self.assertTrue(torch.equal(expected, actual), str(site))
                    learner = rs.learners.CAA(site=rs.sites.resid_post(1), positions=LastNonPaddingToken(), batch_size=2)
                    expected = learner.fit(wrapper, pairs)
                    with capture_forward(wrapper.model):
                        actual = learner.fit(wrapper, pairs)
                    for key, value in expected.tensors().items():
                        self.assertTrue(torch.equal(value, actual.tensors()[key]), key)
                finally:
                    wrapper.close()

    def test_output_reduction_and_restoration_after_exception(self):
        torch.set_num_threads(2)
        for architecture in (Qwen3Config, LlamaConfig, Gemma2Config):
            with self.subTest(model=architecture.__name__):
                wrapper = tiny_wrapper(architecture)
                model = wrapper.model
                batch = wrapper.tokenizer(['answer red blue green'], return_tensors='pt')
                batch.pop('token_type_ids', None)
                config = model.config.to_dict()
                hooks = len(model._forward_pre_hooks)
                try:
                    with torch.no_grad():
                        before = model(**batch)
                        with self.assertRaisesRegex(RuntimeError, 'test cleanup'):
                            with capture_forward(model):
                                during = model(**batch)
                                self.assertEqual(during.logits.shape[1], 1)
                                self.assertIsNone(during.past_key_values)
                                raise RuntimeError('test cleanup')
                        after = model(**batch)
                    self.assertEqual(len(model._forward_pre_hooks), hooks)
                    self.assertEqual(model.config.to_dict(), config)
                    self.assertEqual(after.logits.shape, before.logits.shape)
                    self.assertIsNotNone(after.past_key_values)
                    self.assertTrue(torch.equal(after.logits, before.logits))
                finally:
                    wrapper.close()

if __name__ == '__main__':
    unittest.main()
