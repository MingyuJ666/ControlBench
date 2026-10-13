import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from core.cli import campaign
from core.runtime.experiment import Run


class FakeRun:
    eval_split = 'full'
    domain = SimpleNamespace(NAME='test')
    model = 'test/model'
    blocks, grid = (0, 1), (.5, 1.)
    batch, max_new, limit, revision = 2, 3, None, None
    sample, seed = None, 42
    train_sample, train_seed = None, 42

    def __init__(self):
        self.ready = {stage: set() for stage in ('learn', 'generate', 'judge', 'score')}

    def description(self, methods, targets):
        return {'model': self.model, 'methods': list(methods), 'targets': list(targets)}

    def items(self, target):
        return [{'item_id': target}]

    def complete(self, stage, method, targets):
        return all((method, t) in self.ready[stage] for t in targets)

    def learned(self, method, target):
        return method == 'unsteered' or self.complete('learn', method, [target])

    def generated(self, method, target):
        return self.complete('generate', method, [target])

    def judged(self, method, target):
        return self.complete('judge', method, [target])

    def scored(self, method, target):
        return self.complete('score', method, [target])

    def score(self, method, targets):
        self.ready['score'].update((method, t) for t in targets)


class SchedulerTest(unittest.TestCase):
    def test_partial_learning_does_not_start_duplicate_or_early_generation(self):
        run, jobs, tick = FakeRun(), [], [0]
        class Job:
            def __init__(self, name, argv, gpus, **kwargs):
                self.name, self.gpus, self.request = name, gpus, kwargs['request']
                self.returncode = 0
                jobs.append(name)
                stage, method, targets = (self.request[k] for k in ('stage', 'method', 'targets'))
                if stage == 'learn':
                    run.ready['learn'].add((method, targets[0]))
                else:
                    if stage == 'generate':
                        self_test.assertGreaterEqual(tick[0], 2)
                    run.ready[stage].update((method, t) for t in targets)
            def done(self):
                if self.request['stage'] == 'learn':
                    if tick[0] < 2:
                        return False
                    run.ready['learn'].update((self.request['method'], t) for t in self.request['targets'])
                return True
        self_test = self
        def sleep(_):
            tick[0] += 1
            self.assertLess(tick[0], 20)
        with tempfile.TemporaryDirectory() as tmp, patch.object(campaign, 'Job', Job), \
             patch.object(campaign.time, 'sleep', side_effect=sleep), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            campaign.schedule(run, ['caa'], ['a', 'b'], [0, 1], Path(tmp), 0)
        self.assertEqual(jobs.count('learn_caa'), 1)
        self.assertEqual(jobs.count('generate_caa_a'), 1)
        self.assertEqual(jobs.count('generate_caa_b'), 1)
        self.assertTrue(run.complete('score', 'caa', ['a', 'b']))

    def test_exit_zero_without_outputs_is_retried_then_fails(self):
        run, launches = FakeRun(), []
        class Job:
            def __init__(self, name, argv, gpus, **kwargs):
                self.name, self.gpus, self.request = name, gpus, kwargs['request']
                self.returncode = 0
                launches.append(name)
            def done(self):
                return True
        with tempfile.TemporaryDirectory() as tmp, patch.object(campaign, 'Job', Job), \
             patch.object(campaign.time, 'sleep'), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            with self.assertRaisesRegex(RuntimeError, 'failed after 3 attempts'):
                campaign.schedule(run, ['unsteered'], ['a'], [0], Path(tmp), 0)
        self.assertEqual(launches, ['generate_unsteered_a'] * 3)

    def test_score_failures_have_bounded_retries(self):
        run = FakeRun()
        run.ready['generate'].add(('unsteered', 'a'))
        run.ready['judge'].add(('unsteered', 'a'))
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(run, 'score', side_effect=RuntimeError('broken scores')) as score, \
             patch.object(campaign.time, 'sleep'), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            with self.assertRaisesRegex(RuntimeError, 'score_unsteered_a failed after 3 attempts'):
                campaign.schedule(run, ['unsteered'], ['a'], [0], Path(tmp), 0)
            self.assertIn('broken scores', (Path(tmp) / 'score_unsteered_a.log').read_text())
        self.assertEqual(score.call_count, 3)

    def test_failed_judge_leaves_other_methods_and_targets_running(self):
        run, launches = FakeRun(), []
        run.ready['learn'].update(('caa', t) for t in ('bad', 'good'))
        run.ready['generate'].update(('caa', t) for t in ('bad', 'good'))
        class Job:
            def __init__(self, name, argv, gpus, **kwargs):
                self.name, self.gpus, self.request = name, gpus, kwargs['request']
                launches.append(name)
                stage, method, targets = (self.request[k] for k in ('stage', 'method', 'targets'))
                self.returncode = 1 if stage == 'judge' and method == 'caa' and 'bad' in targets else 0
                run.ready[stage].update((method, t) for t in targets
                                        if not (stage == 'judge' and method == 'caa' and t == 'bad'))
            def done(self):
                return True
        with tempfile.TemporaryDirectory() as tmp, patch.object(campaign, 'Job', Job), \
             patch.object(campaign.time, 'sleep'), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            with self.assertRaisesRegex(RuntimeError, 'judge_caa failed after 3 attempts'):
                campaign.schedule(run, ['caa', 'lat'], ['bad', 'good'], [0], Path(tmp), 0)
            state = json.loads((Path(tmp) / 'state.json').read_text())
            self.assertEqual(state['state']['caa'], {'bad': 'failed', 'good': 'done'})
            self.assertEqual(state['state']['lat'], {'bad': 'done', 'good': 'done'})
            self.assertEqual(state['running'], [])
            self.assertEqual(len(state['failures']), 1)
        self.assertEqual(launches.count('judge_caa'), 3)
        self.assertLess(launches.index('judge_caa'), launches.index('learn_lat'))
        self.assertTrue(run.complete('score', 'lat', ['bad', 'good']))
        self.assertNotIn(('caa', 'bad'), run.ready['score'])

    def test_detached_job_can_be_adopted_and_dead_marker_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            job = campaign.Job('fixture', [sys.executable, '-c', 'import sys; sys.stdin.read()'], [0],
                               request={'stage': 'generate', 'method': 'unsteered', 'targets': ['a']},
                               running_dir=folder, log_dir=folder, run_id='test', devices={0: 'test-device'})
            # Use a separate child whose stdin remains open so adoption is deterministic.
            job.proc.wait(timeout=10)
            self.assertTrue(job.done())
            child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'], stdin=subprocess.PIPE)
            try:
                marker = folder / 'adopt.json'
                marker.write_text(json.dumps({'pid': child.pid, 'start': campaign._process_state(child.pid)[1],
                                              'gpus': [0], 'request': job.request, 'run_id': 'test'}))
                with self.assertRaisesRegex(ValueError, 'another configuration'):
                    campaign.Job.adopt(marker, 'different')
                adopted = campaign.Job.adopt(marker, 'test')
                self.assertEqual(adopted.pid, child.pid)
                self.assertFalse(adopted.done())
                child.communicate(timeout=10)
                self.assertTrue(adopted.done())
                self.assertFalse(marker.exists())
                marker.write_text(json.dumps({'pid': child.pid, 'start': None, 'run_id': 'test'}))
                self.assertIsNone(campaign.Job.adopt(marker, 'test'))
                self.assertFalse(marker.exists())
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()

    def test_child_command_preserves_explicit_signed_grid_and_revision(self):
        run = Run(SimpleNamespace(NAME='test'), 'test/model', (2, 5), (-1., 1.),
                  batch=3, max_new=99, limit=7, revision='commit')
        argv = campaign.child_command(run, 'generate', 'pca', ['a', 'b'])
        self.assertEqual(argv[argv.index('--layers') + 1], '2,5')
        self.assertIn('--strengths=-1.0,1.0', argv)
        for key, value in [('--revision', 'commit'), ('--limit', '7'), ('--batch', '3'), ('--max-new', '99')]:
            self.assertEqual(argv[argv.index(key) + 1], value)
        reference = campaign.child_command(run, 'generate', 'unsteered', ['a'])
        self.assertNotIn('--layers', reference)
        self.assertFalse(any(a.startswith('--strengths') for a in reference))


if __name__ == '__main__':
    unittest.main()
