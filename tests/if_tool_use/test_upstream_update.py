"""Regression coverage for parser guards and full-input judge retries."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from core.domains import tool_use_multi_turn as domain
from core.domains.tool_use import calls


class ParserResourceErrorsTest(unittest.TestCase):
    def test_python_parser_resource_errors_are_invalid_calls(self):
        for operation in ('parse', 'literal_eval'):
            for error in (MemoryError, RecursionError):
                with self.subTest(operation=operation, error=error), \
                     patch.object(calls.ast, operation, side_effect=error):
                    attempted, parsed, *_ = calls.parse('book(city="paris")')
                    self.assertTrue(attempted)
                    self.assertIsNone(parsed)


class LongContextRetryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.inputs = self.folder / 'judge_in_0.jsonl'
        self.inputs.write_text('{"uid":"pending","full_prompt":"do not truncate"}\n')
        self.existing = self.folder / 'judge_out_0.jsonl'
        self.existing.write_text('{"uid":"cached","note":"keep verdict"}\n')
        self.before = self.inputs.read_bytes(), self.existing.read_bytes()
        self.preparation = patch.object(domain.judging, 'preparation',
                                        return_value={'mechanical_only': False})
        self.preparation.start()
        self.addCleanup(self.preparation.stop)

    def overflow(self, required):
        (self.folder / 'judge_unparsed_0.jsonl').write_text(json.dumps({
            'uid': 'pending', 'error': 'context_length_exceeded',
            'required_model_len': required}) + '\n')

    def test_retry_preserves_inputs_verdicts_and_drops_force(self):
        def execute(command, **kwargs):
            if '--long-context' not in command:
                self.overflow(42000)
                return subprocess.CompletedProcess(command, 1)
            self.assertNotIn('--force', command)
            self.assertEqual(command[command.index('--max-model-len') + 1], '45056')
            self.assertTrue(kwargs['check'])
            self.assertEqual((self.inputs.read_bytes(), self.existing.read_bytes()), self.before)
            return subprocess.CompletedProcess(command, 0)

        with patch.object(domain.subprocess, 'run', side_effect=execute) as run:
            domain.judge([self.folder], force=True)
        self.assertEqual(run.call_count, 2)

    def test_existing_overflow_skips_another_short_context_attempt(self):
        self.overflow(20000)
        with patch.object(domain.subprocess, 'run') as run:
            domain.judge([self.folder])
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertIn('--long-context', command)
        self.assertEqual(command[command.index('--max-model-len') + 1], '40960')

    def test_unsupported_length_fails_without_truncation_or_launch(self):
        self.overflow(131073)
        with patch.object(domain.subprocess, 'run') as run, \
             self.assertRaisesRegex(RuntimeError, 'no truncation'):
            domain.judge([self.folder])
        run.assert_not_called()
        self.assertEqual((self.inputs.read_bytes(), self.existing.read_bytes()), self.before)

    def test_unrelated_subprocess_failure_is_not_retried(self):
        with patch.object(domain.subprocess, 'run',
                          return_value=subprocess.CompletedProcess(['judge'], 7)) as run, \
             self.assertRaises(subprocess.CalledProcessError) as error:
            domain.judge([self.folder])
        self.assertEqual(error.exception.returncode, 7)
        run.assert_called_once()

    def test_mechanical_conversations_need_no_judge(self):
        with patch.object(domain.judging, 'preparation', return_value={'mechanical_only': True}), \
             patch.object(domain.subprocess, 'run') as run:
            domain.judge([self.folder])
        run.assert_not_called()
