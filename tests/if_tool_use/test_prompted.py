"""Prompted baseline smoke checks: actual model inputs, cache isolation and CLI."""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import torch
import repsteer as rs
from transformers import Qwen3Config, LlamaConfig

import core as cb
from core import paths
from core.cli import campaign, run as cli
from core.domains import instruction_following as domain
from core.domains.instruction_following.tasks import make_prompted_prompt
from core.runtime.experiment import Run
from core.runtime.search import read_jsonl, input_fingerprint
from test_native_methods import tiny_wrapper
import test_sampling


class PromptedTest(unittest.TestCase):
    dry_run = test_sampling.SamplingTest.dry_run

    def test_standalone_and_mixed_cli_keep_one_baseline_configuration(self):
        args = ['cbench-campaign', '--gpus', '0', '--methods', 'prompted', '--sample', '2']
        result = self.dry_run(campaign, args)
        self.assertEqual(result['methods']['prompted'],
                         {'layers': [], 'strengths': [], 'configurations_per_target': 1, 'outputs': 10})
        run = Run(domain, 'm', (9, 12), (.5, 1.), sample=2, train_sample=4)
        for stage in ('generate', 'judge', 'score'):
            argv = campaign.child_command(run, stage, 'prompted', ['lowercase'])
            self.assertNotIn('--layers', argv)
            self.assertFalse(any(a.startswith('--strengths') for a in argv))
            self.assertEqual(self.dry_run(cli, ['cbench-run', *argv[3:]]),
                             run.description(['prompted'], ['lowercase']))
        mixed = self.dry_run(campaign, ['cbench-campaign', '--gpus', '0', '--sample', '2',
                             '--methods', 'unsteered,prompted,caa,lat', '--layers', '9,12',
                             '--strengths', '.5,1'])
        self.assertEqual({m: c['configurations_per_target'] for m,c in mixed['methods'].items()},
                         {'unsteered': 1, 'prompted': 1, 'caa': 4, 'lat': 4})
        iti = self.dry_run(campaign, ['cbench-campaign', '--gpus', '0', '--sample', '2',
                           '--methods', 'prompted,iti', '--strengths', '.5,1'])
        self.assertEqual(iti['methods']['prompted']['configurations_per_target'], 1)
        self.assertEqual(iti['methods']['iti']['configurations_per_target'], 2)

    def test_prompted_has_no_learning_stage(self):
        with patch.object(cb, 'load_model', side_effect=AssertionError('must not load weights')), \
             patch.object(domain, 'pairs', side_effect=AssertionError('must not fit')):
            run = Run(domain, 'm', (), ())
            self.assertTrue(run.learned('prompted', 'lowercase'))
            with self.assertRaisesRegex(ValueError, 'no learning step'):
                run.learn('prompted', ['lowercase'])
            with patch.object(sys, 'argv', ['cbench-run', 'learn', '--method', 'prompted']), \
                 redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit) as raised:
                cli.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertIn('prompted has no learning step', errors.getvalue())

    def test_same_eval_ids_with_only_prompted_receiving_instructions(self):
        run = Run(domain, 'm', (9,), (.5,), sample=8)
        for target in domain.TARGETS:
            before = json.dumps(run.items(target), sort_keys=True)
            prompted = run.generation_items('prompted', target)
            self.assertEqual([r['item_id'] for r in prompted], [r['item_id'] for r in run.items(target)])
            for item in prompted:
                self.assertEqual(domain.messages(item)[0]['content'],
                                 make_prompted_prompt(item['base_prompt'], target))
            for method in ('unsteered', 'caa', 'lat', 'linear_probe', 'pca', 'iti'):
                for item in run.generation_items(method, target):
                    self.assertEqual(domain.messages(item)[0]['content'], item['base_prompt'])
            self.assertEqual(json.dumps(run.items(target), sort_keys=True), before)
            self.assertNotEqual(input_fingerprint(prompted), input_fingerprint(run.items(target)))

    def test_real_qwen_and_llama_generation_matches_direct_model_and_resumes(self):
        torch.set_num_threads(2)
        for config_cls in (Qwen3Config, LlamaConfig):
            with self.subTest(architecture=config_cls.__name__):
                base = tiny_wrapper(config_cls)
                model, tok, model_id = base.model, base.tokenizer, base.model_id
                base.close()
                rows = [{'item_id': str(i), 'base_prompt': text,
                         'prompted_prompt': text + ' lowercase'}
                        for i,text in enumerate(('answer red', 'answer blue green'))]
                def load_model(*args, **kwargs):
                    return rs.models.from_model(model, tok, model_id=model_id), model, tok
                run = Run(domain, model_id, (), (), batch=2, max_new=3)
                with tempfile.TemporaryDirectory() as tmp, patch.object(paths, 'DATA_ROOT', Path(tmp)), \
                     patch.object(domain, 'items', return_value=rows), \
                     patch.object(domain, 'pairs', side_effect=AssertionError('no training needed')), \
                     patch.object(cb, 'load_model', side_effect=load_model):
                    for method in ('unsteered', 'prompted'):
                        selected = run.generation_items(method, 'lowercase')
                        texts = [domain.render(tok, item) for item in selected]
                        encoded = tok(texts, return_tensors='pt', padding=True, add_special_tokens=False)
                        with torch.no_grad():
                            direct = model.generate(**encoded, max_new_tokens=3, do_sample=False,
                                                    temperature=None, top_p=None, top_k=None,
                                                    pad_token_id=tok.pad_token_id)
                        expected = tok.batch_decode(direct[:, encoded['input_ids'].shape[1]:],
                                                    skip_special_tokens=True)
                        run.generate(method, ['lowercase'])
                        self.assertTrue(run.generated(method, 'lowercase'))
                        out = run.directory(method, 'lowercase', method)
                        actual = read_jsonl(out / 'generations.jsonl')
                        self.assertEqual([r['output'] for r in actual], expected)
                        self.assertTrue(all(0 < r['n_new_tokens'] <= 3 for r in actual))
                        meta = json.loads((out / 'meta.json').read_text())
                        self.assertIsNone(meta['plan'])
                        self.assertIsNone(meta['artifact_fingerprint'])
                        self.assertEqual(meta['input_fingerprint'], input_fingerprint(selected))
                    self.assertNotEqual(run.directory('prompted', 'lowercase'),
                                        run.directory('unsteered', 'lowercase'))
                    with patch.object(cb, 'load_model', side_effect=AssertionError('must resume')):
                        run.generate('prompted', ['lowercase'])
                        run.generate('unsteered', ['lowercase'])
                    # A baseline completion generated from plain inputs is not a prompted cache hit.
                    out = run.directory('prompted', 'lowercase', 'prompted')
                    meta = json.loads((out / 'meta.json').read_text())
                    meta['input_fingerprint'] = input_fingerprint(run.items('lowercase'))
                    (out / 'meta.json').write_text(json.dumps(meta))
                    self.assertFalse(run.generated('prompted', 'lowercase'))
                    self.assertTrue(run.generated('unsteered', 'lowercase'))


if __name__ == '__main__':
    unittest.main()
