"""Regression checks for the shared tool-use prompt protocol and old caches."""
import importlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import core as cb
from core import paths
from core.domains import instruction_following, tool_use, tool_use_multi_turn
from core.methods import METHODS
from core.runtime.experiment import Run
from test_native_methods import tiny_wrapper, training_pairs
from test_tool_use_prompted import single_item, multi_item

render_module = importlib.import_module('core.domains.tool_use.render')


class ToolUseProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_version_namespaces_and_instruction_following_compatibility(self):
        for domain in (tool_use, tool_use_multi_turn):
            for sample in (None, 4):
                run = Run(domain, 'm', (0,), (.5,), sample=2, train_sample=sample)
                current_path = run.artifact_directory('caa', 'call', 'resid_post_0')
                current_output = run.directory('unsteered', 'call', 'unsteered')
                fingerprint = run.score_fingerprint('unsteered', 'call')
                self.assertIn(domain.PROTOCOL_VERSION, str(current_path))
                with patch.object(domain, 'PROTOCOL_VERSION', 'fixture-old-protocol'):
                    self.assertNotEqual(current_path, run.artifact_directory('caa', 'call', 'resid_post_0'))
                    self.assertNotEqual(current_output, run.directory('unsteered', 'call', 'unsteered'))
                    self.assertNotEqual(fingerprint, run.score_fingerprint('unsteered', 'call'))
        run = Run(instruction_following, 'm', (0,), (.5,), sample=2)
        self.assertIsNone(run.training)
        self.assertEqual(run.experiment, 'full-sample-2-seed-42')
        self.assertNotIn('protocol_version', run.learning_config('caa', 'lowercase'))
        self.assertNotIn('protocol_version', run.description(['unsteered'], ['lowercase']))

    def test_legacy_artifact_copied_to_new_path_is_rejected_for_dense_and_iti_learners(self):
        for domain in (tool_use, tool_use_multi_turn):
            # SAE requires pinned Gemma weights; its contracts run in test_sae.
            for method in ('caa', 'linear_probe', 'lat', 'pca', 'iti'):
                with self.subTest(domain=domain.NAME, method=method):
                    wrapper = tiny_wrapper()
                    run = Run(domain, wrapper.model_id, (0,), (.5,), batch=2,
                              max_new=1, iti_top_k=1)
                    with tempfile.TemporaryDirectory() as tmp, \
                         patch.object(paths, 'DATA_ROOT', Path(tmp)), \
                         patch.object(domain, 'pairs', return_value=training_pairs()), \
                         patch.object(cb, 'load_model', return_value=(wrapper, wrapper.model, wrapper.tokenizer)):
                        key = run.learning_keys(method)[0]
                        with patch.object(domain, 'PROTOCOL_VERSION', None):
                            run.learn(method, ['call'])
                            self.assertTrue(run.learned(method, 'call'))
                            old = run.artifact_directory(method, 'call', key)
                        new = run.artifact_directory(method, 'call', key)
                        shutil.copytree(old, new)
                        self.assertFalse(run.learned(method, 'call'))
                        self.assertFalse(run.generated(method, 'call'))

    def test_old_generations_cannot_be_reused_even_when_copied_to_new_path(self):
        for domain, item in ((tool_use, single_item()), (tool_use_multi_turn, multi_item())):
            for method in ('unsteered', 'prompted'):
                with self.subTest(domain=domain.NAME, method=method):
                    wrapper = tiny_wrapper()
                    run = Run(domain, wrapper.model_id, (), (), batch=1, max_new=1)
                    target = item['target']
                    with tempfile.TemporaryDirectory() as tmp, \
                         patch.object(paths, 'DATA_ROOT', Path(tmp)), \
                         patch.object(domain, 'items', return_value=[item]), \
                         patch.object(cb, 'load_model', return_value=(wrapper, wrapper.model, wrapper.tokenizer)):
                        run.generate(method, [target])
                        self.assertTrue(run.generated(method, target))
                        out = run.directory(method, target, method)
                        meta = json.loads((out / 'meta.json').read_text())
                        self.assertEqual(meta['protocol_version'], domain.PROTOCOL_VERSION)
                        original = (out / 'meta.json').read_bytes()
                        meta.pop('protocol_version')
                        (out / 'meta.json').write_text(json.dumps(meta))
                        self.assertFalse(run.generated(method, target))
                        (out / 'meta.json').write_bytes(original)
                        with patch.object(cb, 'load_model', side_effect=AssertionError('must resume')):
                            run.generate(method, [target])


if __name__ == '__main__':
    unittest.main()
