"""Training selection with real inputs, native learners and sampled artifacts."""

import io
import json
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import repsteer as rs
from repsteer.data import ContrastivePairs, ContrastiveRecord

import core as cb
from core import paths
from core.cli import campaign, run as cli
from core.domains import instruction_following as domain, tool_use, tool_use_multi_turn
from core.methods import METHODS
from core.methods import sae
from core.runtime.experiment import Run
from test_native_methods import tiny_wrapper, training_pairs
import test_sampling


class TrainingSamplingTest(unittest.TestCase):
    dry_run = test_sampling.SamplingTest.dry_run

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_real_if_pairs_are_reproducible_and_independent_of_evaluation(self):
        state = random.getstate()
        run = Run(domain, 'model-a', (9,), (.5,), train_sample=32, sample=8, seed=7)
        selected = run.training_pairs('lowercase')
        full = domain.pairs('lowercase')
        indices = selected.metadata['sampling']['indices']
        self.assertEqual(len(selected), 32)
        self.assertEqual(len(set(indices)), 32)
        self.assertEqual(indices, sorted(indices))
        self.assertNotEqual(indices, list(range(32)))
        self.assertEqual(selected.records, tuple(full.records[i] for i in indices))
        for target in domain.TARGETS:
            other = Run(domain, 'model-b', (11, 20), (1., 2.), train_sample=32, sample=16, seed=99)
            self.assertEqual(other.training_pairs(target).negatives, selected.negatives)
        self.assertEqual(run.items('lowercase'), Run(domain, 'm', (), (), sample=8, seed=7).items('lowercase'))
        self.assertNotEqual(selected.negatives,
                            Run(domain, 'm', (), (), train_sample=32, train_seed=43).training_pairs('lowercase').negatives)
        self.assertEqual(Run(domain, 'm', (), (), train_sample=1014).training_pairs('lowercase').records,
                         full.records)
        self.assertEqual(Run(domain, 'm', (), (), sample=3).training_pairs('lowercase').records, full.records)
        self.assertEqual(random.getstate(), state)

    def test_pair_ids_weights_and_side_metadata_survive_selection(self):
        records = [ContrastiveRecord(positive=f'p{i}', negative=f'n{i}', id=f'id{i}', weight=i+1,
                                    positive_weight=i+2, negative_weight=i+3,
                                    metadata={'row': i}, positive_metadata={'side': 'p'},
                                    negative_metadata={'side': 'n'}) for i in range(20)]
        source = ContrastivePairs.from_records(records, paired=True, metadata={'origin': 'fixture', 'pairs': 20})
        dom = SimpleNamespace(NAME='fixture', pairs=lambda _: source)
        selected = Run(dom, 'm', (), (), train_sample=7, train_seed=0).training_pairs('axis')
        self.assertEqual(selected.metadata['pairs'], 7)
        self.assertEqual(selected.metadata['origin'], 'fixture')
        self.assertEqual(selected.metadata['sampling']['source_fingerprint'], source.fingerprint)
        for record in selected:
            self.assertEqual(record, records[int(record.id[2:])])
        self.assertEqual(len(source), 20)
        self.assertEqual(source.metadata, {'origin': 'fixture', 'pairs': 20})
        unpaired = ContrastivePairs.from_groups(['p'], ['n'])
        with patch.object(dom, 'pairs', return_value=unpaired), self.assertRaisesRegex(ValueError, 'requires paired'):
            Run(dom, 'm', (), (), train_sample=1).training_pairs('axis')

    def test_tool_use_sampling_covers_each_axis_and_both_tracks(self):
        run = Run(tool_use, 'm', (9,), (.5,), train_sample=400, train_seed=42)
        for target in tool_use.TARGETS:
            selected = run.training_pairs(target)
            self.assertEqual(len(selected), 400)
            self.assertEqual(selected.metadata['sampling']['source_pairs'], 2700)
            self.assertTrue(all(p.paired for p in selected))
            self.assertEqual(len({json.dumps(p.positive[:-1], sort_keys=True) for p in selected}), 400)
        other = Run(tool_use_multi_turn, 'm', (9,), (.5,), train_sample=400)
        self.assertEqual(other.training_pairs('clarify').fingerprint, run.training_pairs('clarify').fingerprint)

    def test_campaign_forwards_training_options_to_all_stages(self):
        run = Run(domain, sae.MODEL, (9,), (.5,), revision=sae.MODEL_REVISION,
                  train_sample=4, train_seed=0, sample=2, seed=7)
        for method in ('unsteered', *METHODS):
            stages = ('generate', 'judge', 'score') if method == 'unsteered' else ('learn', 'generate', 'judge', 'score')
            for stage in stages:
                with self.subTest(method=method, stage=stage):
                    argv = campaign.child_command(run, stage, method, ['lowercase'])
                    result = self.dry_run(cli, ['cbench-run', *argv[3:]])
                    self.assertEqual(result, run.description([method], ['lowercase']))
                    self.assertEqual(result['training']['pairs_per_target'], {'lowercase': 4})
                    self.assertEqual(result['items_per_target'], {'lowercase': 2})
        args = ['cbench-campaign', '--methods', 'caa', '--gpus', '0', '--layers', '9',
                '--strengths', '.5', '--train-sample', '4', '--targets', 'lowercase']
        default = self.dry_run(campaign, args)
        self.assertEqual(default, self.dry_run(campaign, args + ['--train-seed', '42']))
        self.assertEqual(default['experiment'], 'full-train-sample-4-seed-42')
        self.assertEqual(default['items_per_target'], {'lowercase': 1014})

    def test_invalid_training_selection_fails_before_loading_weights(self):
        bad_args = [(['--train-sample', '0'], 'positive'),
                    (['--train-sample', '-1'], 'positive'),
                    (['--train-seed', '42'], 'requires --train-sample'),
                    (['--train-sample', '1015'], 'between 1 and 1014'),
                    (['--domain', 'tool_use', '--targets', 'call', '--train-sample', '2701'], 'between 1 and 2700'),
                    (['--train-sample', '8', '--seed', '42'], '--seed requires --sample')]
        for args, message in bad_args:
            with self.subTest(args=args), redirect_stderr(io.StringIO()) as errors, \
                 patch.object(sys, 'argv', ['cbench-run', 'learn', '--method', 'caa',
                                           '--layers', '0', '--strengths', '.5', *args]), \
                 patch.object(cb, 'load_model', side_effect=AssertionError('must not load weights')):
                with self.assertRaises(SystemExit) as raised:
                    cli.main()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(message, errors.getvalue())

    def test_training_paths_are_separate_and_evaluation_reuses_artifacts(self):
        runs = [Run(domain, 'm', (0,), (.5,)),
                Run(domain, 'm', (0,), (.5,), train_sample=4),
                Run(domain, 'm', (0,), (.5,), train_sample=4, train_seed=43),
                Run(domain, 'm', (0,), (.5,), train_sample=8)]
        self.assertEqual(len({r.artifact_directory('caa', 'lowercase', 'resid_post_0') for r in runs}), 4)
        self.assertEqual(len({r.directory('caa', 'lowercase') for r in runs}), 4)
        self.assertEqual(runs[0].artifact_directory('caa', 'lowercase', 'resid_post_0'),
                         cb.artifact_dir(domain.NAME, 'm', 'caa', 'lowercase', 'resid_post_0'))
        self.assertNotIn('training', runs[0].description(['caa'], ['lowercase']))
        other_eval = Run(domain, 'm', (0,), (.5,), train_sample=4, sample=2, seed=7)
        self.assertEqual(other_eval.artifact_directory('caa', 'lowercase', 'resid_post_0'),
                         runs[1].artifact_directory('caa', 'lowercase', 'resid_post_0'))
        self.assertNotEqual(other_eval.experiment, runs[1].experiment)

    def test_layer_methods_fit_only_selected_pairs_generate_and_resume(self):
        original = training_pairs(weights=[1, 2, 3, 4, 5, 6])
        base = tiny_wrapper()
        model, tok, model_id = base.model, base.tokenizer, base.model_id
        base.close()
        def load_model(*args, **kwargs):
            return rs.models.from_model(model, tok, model_id=model_id, revision='test-12'), model, tok
        items = [{'item_id': f'eval-{i}', 'prompt_id': f'eval-{i}', 'base_prompt': f'answer {color}'}
                 for i, color in enumerate(['red', 'blue'])]
        full = Run(domain, model_id, (0,), (.5,), batch=2, max_new=2)
        sampled = Run(domain, model_id, (0,), (.5,), batch=2, max_new=2, train_sample=3)
        with tempfile.TemporaryDirectory() as tmp, patch.object(paths, 'DATA_ROOT', Path(tmp)), \
             patch.object(domain, 'pairs', return_value=original), patch.object(domain, 'items', return_value=items), \
             patch.object(cb, 'load_model', side_effect=load_model):
            for method in ('caa', 'linear_probe', 'lat', 'pca'):
                with self.subTest(method=method):
                    full.learn(method, ['lowercase'])
                    old_path = full.artifact_directory(method, 'lowercase', 'resid_post_0')
                    old_files = {p.name: p.read_bytes() for p in old_path.iterdir()}
                    self.assertFalse(sampled.learned(method, 'lowercase'))
                    sampled.learn(method, ['lowercase'])
                    self.assertTrue(sampled.learned(method, 'lowercase'))
                    restored = METHODS[method].load(domain, model_id, 'lowercase', 'resid_post_0', training=sampled.training)
                    wrapper, _, _ = load_model()
                    try:
                        expected = METHODS[method].learn(wrapper, domain, 'lowercase', [rs.sites.resid_post(0)],
                                                         batch_size=2, pairs=sampled.training_pairs('lowercase'))['resid_post_0']
                    finally:
                        wrapper.close()
                    self.assertEqual(set(expected.tensors()), set(restored.tensors()))
                    for key, tensor in expected.tensors().items():
                        self.assertTrue(torch.equal(tensor, restored.tensors()[key]), key)
                    self.assertEqual(restored.metadata.to_dict()['config']['training_selection'],
                                     sampled.training_config('lowercase')['training_selection'])
                    self.assertEqual({p.name: p.read_bytes() for p in old_path.iterdir()}, old_files)
                    sampled.generate(method, ['lowercase'])
                    self.assertTrue(sampled.generated(method, 'lowercase'))
                    with patch.object(cb, 'load_model', side_effect=AssertionError('must reuse completed outputs')):
                        sampled.learn(method, ['lowercase'])
                        sampled.generate(method, ['lowercase'])
                    wrong_seed = Run(domain, model_id, (0,), (.5,), train_sample=3, train_seed=43)
                    cb.save_artifact(restored, domain=domain.NAME, model=model_id, method=method,
                                     target='lowercase', training=wrong_seed.training)
                    self.assertFalse(wrong_seed.learned(method, 'lowercase'))
            changed = ContrastivePairs.from_records(original.records, metadata={'source': 'changed'})
            with patch.object(domain, 'pairs', return_value=changed):
                rerun = Run(domain, model_id, (0,), (.5,), batch=2, max_new=2, train_sample=3)
                self.assertFalse(rerun.learned('caa', 'lowercase'))
                self.assertFalse(rerun.generated('caa', 'lowercase'))


if __name__ == '__main__':
    unittest.main()
