"""Explicit output roots leave inputs, experiment settings and scoring unchanged."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from core import domains, paths
from core.cli import campaign, run as run_cli
from core.runtime.experiment import Run


class OutputDirectoryTests(unittest.TestCase):
    def test_override_preserves_inputs_and_experiment_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / 'results with spaces'
            for name in domains.names():
                domain = domains.get(name)
                target = domain.TARGETS[0]
                run = Run(domain, 'Qwen/Qwen3-8B', (9,), (1.,), sample=2, train_sample=4)
                inputs = domain.INPUTS
                original_root = paths.output_root(name)
                original_run = run.directory('caa', target)
                original_artifact = run.artifact_directory('caa', target, 'resid_post_9')
                original_judge = domain.judging_dir(run.model, run.experiment, 'caa', target)
                description = run.description(('caa',), (target,))
                items = run.items(target)
                pairs = str(run.training_pairs(target).fingerprint)
                fingerprint = run.score_fingerprint('caa', target)
                with self.subTest(domain=name), paths.output_directory(destination):
                    self.assertEqual(paths.output_root(name), destination)
                    self.assertEqual(run.directory('caa', target), destination / original_run.relative_to(original_root))
                    self.assertEqual(run.artifact_directory('caa', target, 'resid_post_9'),
                                     destination / original_artifact.relative_to(original_root))
                    self.assertEqual(domain.judging_dir(run.model, run.experiment, 'caa', target),
                                     destination / original_judge.relative_to(original_root))
                    self.assertEqual(domain.INPUTS, inputs)
                    self.assertEqual(run.items(target), items)
                    self.assertEqual(str(run.training_pairs(target).fingerprint), pairs)
                    # Existing run fingerprints include the output path.
                    self.assertNotEqual(run.score_fingerprint('caa', target), fingerprint)
                    self.assertEqual(run.description(('caa',), (target,)),
                                     {**description, 'output_root': str(destination)})
                self.assertEqual(paths.output_root(name), original_root)
                with paths.output_directory(original_root):
                    self.assertEqual(run.score_fingerprint('caa', target), fingerprint)
            self.assertFalse(destination.exists())

    def test_external_inputs_and_output_precedence(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'STEERING_DATA': tmp}), \
                patch.object(paths, 'DATA_ROOT', Path(tmp)):
            with paths.output_directory(Path(tmp) / 'results'):
                for name in ('instruction_following', 'tool_use'):
                    self.assertEqual(paths.input_root(name), Path(tmp) / name / 'inputs')
                    self.assertEqual(paths.output_root(name), Path(tmp) / 'results')
            self.assertEqual(paths.output_root('tool_use'), Path(tmp) / 'tool_use/outputs' / paths.PIPELINE_ID)

    def test_cli_scope_resets_after_success_and_error(self):
        original = paths.output_root('instruction_following')
        with tempfile.TemporaryDirectory() as tmp:
            argv = ['run', 'generate', '--method', 'unsteered', '--out', tmp, '--dry-run']
            with patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()) as output:
                run_cli.main()
            self.assertEqual(json.loads(output.getvalue())['output_root'], tmp)
            self.assertEqual(paths.output_root('instruction_following'), original)
            with patch.object(sys, 'argv', argv[:-1]), \
                    patch.object(Run, 'generate', side_effect=RuntimeError('test failure')), \
                    patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
                run_cli.main()
            self.assertEqual(paths.output_root('instruction_following'), original)

    def test_campaign_log_defaults_and_explicit_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in domains.names():
                for out in (None, Path(tmp) / name):
                    for logs in (None, Path(tmp) / 'separate logs'):
                        argv = ['campaign', '--domain', name, '--methods', 'unsteered', '--gpus', '0']
                        if out is not None:
                            argv += ['--out', str(out)]
                        if logs is not None:
                            argv += ['--logs', str(logs)]
                        with self.subTest(domain=name, out=out, logs=logs), \
                                patch.object(sys, 'argv', argv), patch.object(campaign, 'schedule') as schedule:
                            campaign.main()
                        run = schedule.call_args.args[0]
                        root = out / 'logs' if out else paths.REPO_ROOT / 'logs' / name / paths.PIPELINE_ID
                        expected = logs or root / paths.model_slug(run.model) / run.experiment
                        self.assertEqual(schedule.call_args.args[4], expected)

    def test_real_stage_children_receive_absolute_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / 'output with spaces'
            relative = os.path.relpath(destination)
            for name in domains.names():
                domain = domains.get(name)
                run = Run(domain, 'Qwen/Qwen3-8B', (9,), (1.,), sample=2)
                with paths.output_directory(relative):
                    for stage in ('learn', 'generate', 'judge', 'score'):
                        argv = campaign.child_command(run, stage, 'caa', (domain.TARGETS[0],))
                        self.assertEqual(argv[argv.index('--out') + 1], str(destination))
                        # Job executes from the repository root, independent of its caller.
                        result = subprocess.run([*argv, '--dry-run'], cwd=paths.REPO_ROOT,
                                                capture_output=True, text=True, timeout=120)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(json.loads(result.stdout)['output_root'], str(destination))
                self.assertNotIn('--out', campaign.child_command(run, 'generate', 'caa', (domain.TARGETS[0],)))
            self.assertFalse(destination.exists())


if __name__ == '__main__':
    unittest.main()
