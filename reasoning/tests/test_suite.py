"""Exercise joint suite orchestration without loading a language model."""
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from reasoning.scripts import paper_baselines as suite


class SuiteTests(unittest.TestCase):
    def test_all_methods_select_on_joint_dev_and_use_matching_artifact(self):
        import reasoning.steering as steering
        for position in ('both', 'before_answer', 'after_answer'):
            with self.subTest(position=position), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory)
                split = root/'split.json'
                split.write_text(json.dumps(dict(extraction=['e'], dev=['d'], eval=['t'])))
                pairs = root/'pairs.jsonl'
                pairs.write_text(json.dumps(dict(id='e'))+'\n')
                rows = [dict(id=i, problem=i, answer='1') for i in ('e','d','t')]
                args = SimpleNamespace(model_index=0, methods=[*suite.DEFAULT_METHODS, 'sae'],
                    extraction_position=position, output=root, smoke=True, sae_layer=20,
                    sae_strengths=[0.5,1], sae_weights=pairs, max_tokens=32, dev_tokens=16,
                    batch_size=1, device='cpu', wait_pid=None)
                backend = SimpleNamespace(extraction_position=None)
                backend.contrastive_data = lambda p: (backend.extraction_position, ['e'])
                fitted, evaluated = [], []

                def fit(b, method, layer, data, path, weights):
                    self.assertEqual(data, b.extraction_position)
                    fitted.append((method, data, path))
                    return (method, data)

                def evaluate(b, items, artifact, candidate, budget, path, config, batch, prompt_suffix=None):
                    if prompt_suffix is not None:
                        self.assertEqual(prompt_suffix, suite.COT_SUFFIX)
                        self.assertIsNone(artifact)
                        self.assertIsNone(candidate)
                    if candidate:
                        expected = candidate.get('extraction_position', position)
                        self.assertEqual(artifact, (config['method'], expected))
                        correct = int(expected == 'after_answer')
                    else:
                        correct = 0
                    path.write_text(''.join(json.dumps(dict(id=r['id'], correct=bool(correct)))+'\n' for r in items))
                    evaluated.append((config['stage'], candidate))
                    return dict(candidate=candidate, correct=correct, total=len(items),
                                accuracy=correct/len(items), missing_final_answer=0, truncated=0)

                patches = dict(SPLIT=split, open_weight_model=lambda index: dict(model='google/gemma-2-9b-it',revision='test'),
                    load_suite=lambda: ({'math500':[rows[-1]]}, {}), load=lambda: rows,
                    prepare_pairs=lambda model, output: pairs, digest=lambda p: 'hash',
                    code_hashes=lambda: {}, prepare_run=lambda *a: None, state=lambda *a, **k: None,
                    fit=fit, evaluate=evaluate)
                for name, value in patches.items():
                    stack.enter_context(patch.object(suite, name, value))
                factory = stack.enter_context(patch.object(steering, 'RepSteerBackend', return_value=backend))
                suite.run(args)
                factory.assert_called_once()
                results = json.loads((root/'results.json').read_text())
                self.assertIn('cot', results['math500'])
                self.assertIn('| math500 | cot | — |', (root/'report.md').read_text())
                self.assertTrue((root/'dev_cot_baseline.jsonl').exists())
                count = 2 if position == 'both' else 1
                self.assertEqual(len(fitted), len(args.methods)*count)
                self.assertEqual(len({p for _,_,p in fitted}), len(fitted))
                self.assertEqual(sum(c is None for _,c in evaluated), 4)
                for method in args.methods:
                    selected = json.loads((root/f'{method}_selected.json').read_text())['candidate']
                    self.assertEqual(selected.get('extraction_position', position),
                                     'after_answer' if position == 'both' else position)
                tests = [c for stage,c in evaluated if stage == 'math500' and c]
                self.assertEqual(len(tests), len(args.methods))

    def test_sae_weight_validation(self):
        import numpy as np
        from reasoning.experiments.sae import validate_weights
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'layer_20'/'params.npz'
            path.parent.mkdir()
            params = dict(W_enc=np.zeros((3584,2)), W_dec=np.zeros((2,3584)),
                          b_enc=np.zeros(2), b_dec=np.zeros(3584), threshold=np.zeros(2))
            np.savez(path, **params)
            self.assertEqual(validate_weights(path, 20), (3584,2))
            with self.assertRaises(ValueError):
                validate_weights(path, 14)
            params['threshold'][0] = float('nan')
            np.savez(path, **params)
            with self.assertRaises(ValueError):
                validate_weights(path, 20)


class CotSuiteTests(unittest.TestCase):
    def test_cot_prompt_cache_and_resume_are_separate(self):
        from unittest.mock import Mock
        backend = SimpleNamespace(generate=Mock(return_value=[dict(
            completion='answer', tokens=1, truncated=False)]))
        rows = [dict(id='q', problem='question', answer='1')]
        meta = dict(run={}, stage='dev', candidate=None)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'dev_cot_baseline.jsonl'
            with patch('reasoning.baseline_cache.reuse') as reuse, patch.object(suite, 'score',
                    return_value=dict(correct=True, parsed_answer='1')), patch.object(suite, 'state'):
                args = (backend, rows, None, None, 32, output, meta, 1)
                first = suite.evaluate(*args, prompt_suffix=suite.COT_SUFFIX)
                backend.generate.assert_called_once_with(['question'], max_tokens=32,
                                                         prompt_suffix=suite.COT_SUFFIX)
                self.assertEqual(reuse.call_args.args[-1], suite.COT_SUFFIX)
                self.assertEqual(suite.evaluate(*args, prompt_suffix=suite.COT_SUFFIX), first)
                self.assertEqual(backend.generate.call_count, 1)
                stored = json.loads(output.with_suffix('.jsonl.meta.json').read_text())
                self.assertEqual(stored['prompt_suffix'], suite.COT_SUFFIX)
                self.assertEqual(stored['budget'], 32)
                with self.assertRaises(ValueError):
                    suite.evaluate(*args)
                with self.assertRaises(ValueError):
                    suite.evaluate(backend, rows, object(), None, 32, output, meta, 1,
                                   prompt_suffix=suite.COT_SUFFIX)
