"""Sampling reproducibility and CLI-to-worker integration, using real IF inputs."""

import io
import json
import random
import sys
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch

import core as cb
from core.cli import campaign, run as cli
from core.domains import instruction_following as domain
from core.runtime.experiment import Run


class SamplingTest(unittest.TestCase):
    def ids(self, run, target='lowercase'):
        return [item['prompt_id'] for item in run.items(target)]

    def dry_run(self, entry, argv):
        output = io.StringIO()
        with patch.object(sys, 'argv', argv + ['--dry-run']), redirect_stdout(output), \
             patch.object(cb, 'load_model', side_effect=AssertionError('must not load weights')):
            entry.main()
        return json.loads(output.getvalue())

    def test_sample_is_unique_repeatable_and_shared_across_targets_and_models(self):
        rng_before = random.getstate()
        first = Run(domain, 'model-a', (9,), (.5,), sample=16)
        expected = self.ids(first)
        self.assertEqual(len(expected), 16)
        self.assertEqual(len(set(expected)), 16)
        self.assertNotEqual(expected, self.ids(Run(domain, 'model-a', (), (), limit=16)))
        for target in reversed(domain.TARGETS):
            other = Run(domain, 'model-b', (11, 20), (1., 2.), sample=16)
            self.assertEqual(self.ids(other, target), expected)
        self.assertEqual(random.getstate(), rng_before)
        self.assertEqual(self.ids(first), expected)

    def test_seed_changes_sample_and_isolates_outputs(self):
        common = (domain, 'model', (), ())
        full, limited = Run(*common), Run(*common, limit=8)
        sample = Run(*common, sample=8, seed=42)
        other = Run(*common, sample=8, seed=43)
        self.assertNotEqual(self.ids(sample), self.ids(other))
        runs = [full, limited, sample, other]
        self.assertEqual([r.experiment for r in runs],
                         ['full', 'full-limit-8', 'full-sample-8-seed-42', 'full-sample-8-seed-43'])
        self.assertEqual(len({r.directory('unsteered', 'lowercase') for r in runs}), 4)
        self.assertEqual(len(full.items('lowercase')), 1014)
        self.assertEqual(self.ids(limited), self.ids(full)[:8])
        self.assertEqual(self.ids(Run(*common, sample=1014)), self.ids(full))

    def test_campaign_default_seed_and_output_counts(self):
        args = ['cbench-campaign', '--gpus', '0,1', '--methods', 'unsteered,caa,lat',
                '--layers', '9', '--strengths', '0.5', '--sample', '8']
        result = self.dry_run(campaign, args)
        self.assertEqual(result['experiment'], 'full-sample-8-seed-42')
        self.assertEqual(result['items_per_target'], {t: 8 for t in domain.TARGETS})
        self.assertEqual(sum(m['outputs'] for m in result['methods'].values()), 120)
        self.assertEqual(result, self.dry_run(campaign, args + ['--seed', '42']))

    def test_sampling_survives_every_child_stage_and_method(self):
        run = Run(domain, 'model', (9, 12), (-1., 1.), sample=8, seed=0)
        for method in ('unsteered', 'caa', 'lat'):
            stages = ('generate', 'judge', 'score') if method == 'unsteered' else ('learn', 'generate', 'judge', 'score')
            for stage in stages:
                with self.subTest(method=method, stage=stage):
                    argv = campaign.child_command(run, stage, method, ['lowercase'])
                    result = self.dry_run(cli, ['cbench-run', *argv[3:]])
                    self.assertEqual(result, run.description([method], ['lowercase']))

    def test_cli_rejects_invalid_sampling_before_model_loading(self):
        bad_args = [(['--sample', '0'], 'positive'), (['--sample', '-1'], 'positive'),
                    (['--sample', '8', '--limit', '8'], 'not allowed'),
                    (['--seed', '42'], 'requires --sample'),
                    (['--sample', '1015'], 'exceeds 1014')]
        for args, message in bad_args:
            with self.subTest(args=args), redirect_stderr(io.StringIO()) as errors, \
                 patch.object(sys, 'argv', ['cbench-run', 'generate', '--method', 'unsteered', *args]), \
                 patch.object(cb, 'load_model', side_effect=AssertionError('must not load weights')):
                with self.assertRaises(SystemExit) as raised:
                    cli.main()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(message, errors.getvalue())

    def test_sample_validates_only_requested_targets(self):
        rows = [{'item_id': str(i)} for i in range(20)]
        with patch.object(domain, 'items', side_effect=lambda t, _: rows if t == 'lowercase' else []) as items:
            result = self.dry_run(cli, ['cbench-run', 'generate', '--method', 'unsteered',
                                       '--targets', 'lowercase', '--sample', '8'])
            self.assertEqual(result['items_per_target'], {'lowercase': 8})
            items.assert_called_once_with('lowercase', 'full')


if __name__ == '__main__':
    unittest.main()
