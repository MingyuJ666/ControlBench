import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from reasoning.datasets import add_extraction_argument, extraction_text
from reasoning.common import worker_arguments


class ExtractionPositionTests(unittest.TestCase):
    def test_boundary_uses_last_boxed_line(self):
        text = 'Intermediate \\boxed{2}\nDerive 4.\nFinal: \\boxed{4}\nExplanation after answer.  \n'
        self.assertEqual(extraction_text(text), 'Intermediate \\boxed{2}\nDerive 4.')
        self.assertEqual(extraction_text(text, 'after_answer'), text.rstrip())

    def test_missing_or_first_line_answer(self):
        for text in ['No boxed answer', '\\boxed{4}\nPostscript']:
            self.assertEqual(extraction_text(text, 'before_answer'), '')
            self.assertEqual(extraction_text(text, 'after_answer'), text)
        self.assertEqual(extraction_text('  \n', 'after_answer'), '')
        with self.assertRaises(ValueError):
            extraction_text('answer', 'invalid')

    def test_cli_default_and_validation(self):
        parser = argparse.ArgumentParser()
        add_extraction_argument(parser)
        self.assertEqual(parser.parse_args([]).extraction_position, 'before_answer')
        args = parser.parse_args(['--extraction-position', 'after_answer'])
        args.action = 'start'
        self.assertIn('after_answer', worker_arguments(args))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires experiment environment')
class BackendExtractionTests(unittest.TestCase):
    def test_capture_inputs_change_with_position_and_preserve_source(self):
        from reasoning.steering import RepSteerBackend
        tokenizer = SimpleNamespace(decode=lambda *a, **k: 'PROMPT\n',
                                    encode=lambda s, **k: list(s))
        backend = RepSteerBackend.__new__(RepSteerBackend)
        backend.base = SimpleNamespace(tokenizer=tokenizer, prompt=lambda s: [1],
            model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=16384)))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'pairs.jsonl'
            rows = [dict(id=str(i),problem='p',teacher='Reason.\n\\boxed{4}',
                         negative='Other.\n\\boxed{5}') for i in range(8)]
            original = ''.join(json.dumps(r)+'\n' for r in rows)
            path.write_text(original)
            for position, expected in [('before_answer','PROMPT\nReason.'),
                                       ('after_answer','PROMPT\nReason.\n\\boxed{4}')]:
                backend.extraction_position = position
                data, ids = backend.contrastive_data(path)
                self.assertEqual(data.positive_examples[0].input, expected)
                self.assertEqual(ids, [str(i) for i in range(8)])
            self.assertEqual(path.read_text(), original)

    def test_artifact_position_mismatch_is_rejected(self):
        from reasoning.steering import RepSteerBackend
        backend = RepSteerBackend.__new__(RepSteerBackend)
        backend.extraction_position = 'after_answer'
        for config in [{}, {'extraction_position':'before_answer'}]:
            with self.assertRaises(ValueError):
                backend.check_extraction_position(SimpleNamespace(metadata=SimpleNamespace(config=config)))
        backend.check_extraction_position(SimpleNamespace(metadata=SimpleNamespace(
            config={'extraction_position':'after_answer'})))


class JointExtractionTests(unittest.TestCase):
    def test_joint_cli_grid_and_selected_artifact(self):
        from reasoning.common import steering_artifact_path
        from reasoning.evaluation import label
        from reasoning.experiments.tune import candidate_grid
        from reasoning.experiments.evaluate import conditions
        parser = argparse.ArgumentParser()
        add_extraction_argument(parser, allow_both=True)
        args = parser.parse_args(['--extraction-position', 'both'])
        args.action = 'start'
        self.assertEqual(worker_arguments(args), ['--extraction-position', 'both'])
        grid = candidate_grid(layers=[14, 20], strengths=[0.5, 1.0],
                              extraction_position=args.extraction_position)
        self.assertEqual(len(grid), 8)
        self.assertEqual(len({label(c) for c in grid}), 8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {steering_artifact_path(root/'artifacts', c) for c in grid}
            self.assertEqual(len(paths), 4)
            for position in ('before_answer', 'after_answer'):
                candidate = next(c for c in grid if c['extraction_position'] == position)
                (root/'config.json').write_text(json.dumps(dict(method='lat', grid=grid)))
                (root/'selected_steering.json').write_text(json.dumps(dict(candidate=candidate)))
                self.assertEqual(conditions(SimpleNamespace(tuning_dirs=[root])),
                    {'lat': (root/'artifacts'/position/'L14', candidate)})

    def test_joint_results_resume_without_cross_position_reuse(self):
        import sys
        from unittest.mock import Mock, patch
        from reasoning.evaluation import evaluate
        rows = [dict(id='a', problem='p', answer='4')]
        generate = Mock(return_value=[dict(completion='answer', truncated=False)])
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {'reasoning.steering': SimpleNamespace(generate=generate)}):
                with patch('reasoning.evaluation.score', return_value={'correct': True}), patch('reasoning.evaluation.state'):
                    for position in ('before_answer', 'after_answer'):
                        candidate = dict(layer=14, alpha=1.0, phase='decode', extraction_position=position)
                        artifact = object()
                        args = (None, artifact, candidate, rows, 32, 'screen', {}, Path(directory))
                        first = evaluate(*args)
                        self.assertEqual(evaluate(*args), first)
                    self.assertEqual(generate.call_count, 2)
                    self.assertEqual(len(list(Path(directory).glob('screen_*.jsonl'))), 2)
