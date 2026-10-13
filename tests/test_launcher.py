"""Exercise real child processes without importing model libraries or reading credentials."""

import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import unittest
import venv


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'bench.py'


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.python = self.folder / 'python with spaces'
        self.python.write_text(f'#!{sys.executable}\n' + '''import json, os, signal, sys
print(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd(),
                  'pythonpath': os.environ.get('PYTHONPATH'),
                  'gpu_python': os.environ.get('L2S_GPU_PYTHON'),
                  'pairs_python': os.environ.get('L2S_PYTHON')}), flush=True)
if '--probe-signal' in sys.argv:
    os.kill(os.getpid(), signal.SIGTERM)
sys.exit(int(os.environ.get('BENCH_TEST_EXIT', '0')))
''')
        self.python.chmod(0o755)
        self.env = {**os.environ, 'HOME': str(self.folder),
                    'PYTHONPATH': str(self.folder / 'prior imports'),
                    'PYTHONDONTWRITEBYTECODE': '1'}
        # Deliberately incompatible local choices must not override --python.
        (self.folder / '.env').write_text(
            'L2S_GPU_PYTHON=/missing/gpu-python\nL2S_PYTHON=/missing/pairs-python\n'
            f'L2S_TMPDIR={shlex.quote(str(self.folder / "runtime tmp"))}\n')

    def launch(self, domain, native=(), entry=None, options=(), python=None):
        command = [sys.executable, str(LAUNCHER), '--domain', domain,
                   '--python', str(python or self.python), *options]
        if entry:
            command += ['--entry', entry]
        return subprocess.run([*command, '--', *native], cwd=self.folder,
                              env=self.env, text=True, capture_output=True, timeout=20)

    def output(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_if_and_both_tool_tracks_preserve_native_arguments(self):
        native = ['--model', 'model path with spaces', '--strengths=-1,1',
                  '--eval-split', 'full', '--out', 'output path with spaces', '--dry-run']
        for domain in ('instruction_following', 'tool_use', 'tool_use_multi_turn'):
            for entry in ('campaign', 'run', 'selfcheck'):
                with self.subTest(domain=domain, entry=entry):
                    value = self.output(self.launch(domain, native, entry))
                    self.assertEqual(value['argv'], ['-m', f'core.cli.{entry}',
                                                    *native, '--domain', domain])
                    self.assertEqual(value['cwd'], str(ROOT))
                    project = 'instruction-following' if domain == 'instruction_following' else 'tool-use'
                    self.assertEqual(value['pythonpath'], str(ROOT / project)
                                     + os.pathsep + self.env['PYTHONPATH'])

    def test_reasoning_and_safety_use_repo_root(self):
        for domain, prefix, native in [
            ('reasoning', ['-m', 'reasoning'], ['suite', 'dry-run', '--methods', 'caa', 'iti']),
            ('safety', [str(ROOT / 'safety/run.py')], ['audit', '--model', 'qwen3_8b', '--out', 'out']),
        ]:
            with self.subTest(domain=domain):
                value = self.output(self.launch(domain, native))
                self.assertEqual(value['argv'], prefix + native)
                self.assertEqual(value['cwd'], str(ROOT))
                self.assertEqual(value['pythonpath'], self.env['PYTHONPATH'])

    def test_l2s_original_wrappers_respect_interpreter_and_tracks(self):
        for entry, track in [('non-iti', 'non-iti'), ('iti', 'iti'), ('inference', None)]:
            with self.subTest(entry=entry):
                value = self.output(self.launch('long-to-short', ['plan', '--pairs', 'data/l2s'], entry))
                expected = ['-m', 'l2s_inference.cli', 'plan', '--pairs', 'data/l2s']
                if track:
                    expected += ['--track', track]
                self.assertEqual(value['argv'], expected)
                self.assertEqual(value['cwd'], str(ROOT / 'long-to-short'))
                self.assertEqual(value['gpu_python'], str(self.python))
                self.assertEqual(value['pythonpath'], str(ROOT / 'long-to-short/src')
                                 + os.pathsep + self.env['PYTHONPATH'])
        self.assertFalse((self.folder / 'runtime tmp').exists())

    def test_l2s_pairs_and_non_plan_temp_setup(self):
        value = self.output(self.launch('long_to_short', ['--prepare-only'], 'pairs'))
        self.assertEqual(value['argv'], ['-m', 'l2s.craft_pairs', '--prepare-only'])
        self.assertEqual(value['pairs_python'], str(self.python))
        value = self.output(self.launch('long_to_short', ['fit', '--model', 'Qwen3-8B']))
        self.assertTrue((self.folder / 'runtime tmp').is_dir())
        self.assertEqual(value['argv'][-2:], ['--track', 'non-iti'])

    def test_legacy_l2s_commands_still_use_env_interpreters(self):
        (self.folder / '.env').write_text(
            f'L2S_GPU_PYTHON={shlex.quote(str(self.python))}\n'
            f'L2S_PYTHON={shlex.quote(str(self.python))}\n')
        for script, native, expected in [
            ('inference_non_iti.sh', ['plan'], ['-m', 'l2s_inference.cli', 'plan', '--track', 'non-iti']),
            ('craft_pairs.sh', ['--prepare-only'], ['-m', 'l2s.craft_pairs', '--prepare-only']),
        ]:
            with self.subTest(script=script):
                result = subprocess.run(['bash', str(ROOT / 'long-to-short/src' / script), *native],
                                        cwd=ROOT / 'long-to-short', env=self.env,
                                        text=True, capture_output=True, timeout=20)
                self.assertEqual(self.output(result)['argv'], expected)

    def test_preview_does_not_read_dotenv_or_launch(self):
        (self.folder / '.env').write_text('exit 93\nSECRET=do-not-display\n')
        value = self.output(self.launch('long_to_short', ['plan'], options=['--preview']))
        self.assertEqual(value['cwd'], str(ROOT / 'long-to-short'))
        self.assertEqual(value['python'], str(self.python))
        self.assertEqual(value['environment_overrides'], {})
        self.assertNotIn('do-not-display', json.dumps(value))

    def test_exit_status_and_signal_propagate(self):
        self.env['BENCH_TEST_EXIT'] = '17'
        self.assertEqual(self.launch('reasoning', ['status']).returncode, 17)
        self.env['BENCH_TEST_EXIT'] = '0'
        self.assertEqual(self.launch('reasoning', ['--probe-signal']).returncode, -signal.SIGTERM)

    def test_relative_interpreter_and_domain_alias(self):
        value = self.output(self.launch('instruction-following', ['--help'],
                                       python='./python with spaces'))
        self.assertEqual(value['argv'][-1], 'instruction_following')

    def test_virtualenv_interpreter_symlink_is_not_resolved_away(self):
        envdir = self.folder / 'venv'
        venv.EnvBuilder(with_pip=False, symlinks=True).create(envdir)
        executable = envdir / 'bin/python'
        # Run a real standard-library-only domain through the selected venv.
        value = self.output(self.launch('reasoning', ['--help'],
                                       options=['--preview'], python=executable))
        self.assertEqual(value['argv'][0], str(executable))
        self.assertNotEqual(value['argv'][0], str(executable.resolve()))
        self.output_prefix = subprocess.run([value['argv'][0], '-c', 'import sys; print(sys.prefix)'],
                                           text=True, capture_output=True, timeout=20)
        self.assertEqual(self.output_prefix.stdout.strip(), str(envdir))

    def test_invalid_requests_fail_before_launch(self):
        cases = [
            ('unknown', [], None, None),
            ('safety', [], 'campaign', None),
            ('tool_use', ['--domain', 'instruction_following'], None, None),
            ('reasoning', [], None, './missing-python'),
        ]
        for domain, native, entry, python in cases:
            with self.subTest(domain=domain, entry=entry, python=python):
                result = self.launch(domain, native, entry, python=python)
                self.assertEqual(result.returncode, 2)
                self.assertIn('error:', result.stderr)
                self.assertEqual(result.stdout, '')

    def test_real_reasoning_and_safety_help_match_direct_commands(self):
        for domain, native in [('reasoning', ['-m', 'reasoning', '--help']),
                               ('safety', [str(ROOT / 'safety/run.py'), '--help'])]:
            with self.subTest(domain=domain):
                direct = subprocess.run([sys.executable, *native], cwd=ROOT, env=self.env,
                                        text=True, capture_output=True, timeout=20)
                launched = self.launch(domain, ['--help'], python=sys.executable)
                self.assertEqual(direct.returncode, 0, direct.stderr)
                self.assertEqual((launched.returncode, launched.stdout, launched.stderr),
                                 (direct.returncode, direct.stdout, direct.stderr))


if __name__ == '__main__':
    unittest.main()
