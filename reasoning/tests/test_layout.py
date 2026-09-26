"""Check public entry points and checkpoint behavior without inference."""
import json
import subprocess
import sys
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from reasoning.cli import COMMANDS
from reasoning.common import ROOT, Journal, code_hashes, worker_arguments
from reasoning.evaluation import evaluate
from reasoning.experiments.tune import candidate_grid
from reasoning.experiments.evaluate import conditions


class LayoutTests(unittest.TestCase):
    def test_command_help(self):
        for command in COMMANDS:
            with self.subTest(command=command):
                result = subprocess.run(
                    [sys.executable, '-m', 'reasoning', command, '--help'],
                    cwd=ROOT.parent, capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)

    def test_evaluation_dry_run_uses_current_data_path(self):
        result = subprocess.run(
            [sys.executable,'-m','reasoning','evaluate','dry-run','--methods','lat'],
            cwd=ROOT.parent,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['counts'],
                         {'amc2023':40,'aime2024':30,'math500':280})

    def test_cot_baseline_only_dry_run(self):
        from reasoning.common import COT_SUFFIX, SUFFIX
        result = subprocess.run(
            [sys.executable, '-m', 'reasoning', 'evaluate', 'dry-run',
             '--baseline-only', '--cot-baseline', '--max-tokens', '2048'],
            cwd=ROOT.parent, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        config = json.loads(result.stdout)
        self.assertEqual(config['conditions'], {})
        self.assertEqual(config['counts'], {'amc2023':40,'aime2024':30,'math500':280})
        self.assertEqual(config['prompts']['baseline'], SUFFIX)
        self.assertEqual(config['prompts']['cot'], COT_SUFFIX)

    def test_cot_only_does_not_schedule_baseline(self):
        from reasoning.experiments.evaluate import condition_names
        self.assertEqual(condition_names(SimpleNamespace(cot_only=True)), ['cot'])
        result = subprocess.run(
            [sys.executable, '-m', 'reasoning', 'evaluate', 'dry-run', '--cot-only'],
            cwd=ROOT.parent, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['evaluation_conditions'], ['cot'])

    def test_cot_prompt_preserves_chat_roles_and_answer_instruction(self):
        from reasoning.models import TransformersLLM
        from reasoning.common import SYSTEM, SUFFIX, COT_SUFFIX
        model = TransformersLLM.__new__(TransformersLLM)
        model.tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kw: messages)
        baseline = model.prompt('question')
        cot = model.prompt('question', suffix=COT_SUFFIX)
        self.assertEqual(baseline, [dict(role='system',content=SYSTEM),
                                   dict(role='user',content='question'+SUFFIX)])
        self.assertEqual(cot[0], baseline[0])
        self.assertEqual(cot[1]['content'], "question\nLet's think step by step."+SUFFIX)

    def test_thinking_mode_applies_to_both_prompt_baselines(self):
        from reasoning.models import TransformersLLM
        from reasoning.common import SUFFIX, COT_SUFFIX
        from unittest.mock import Mock
        model = TransformersLLM.__new__(TransformersLLM)
        model.enable_thinking = False
        model.tokenizer = SimpleNamespace(apply_chat_template=Mock(return_value=[]))
        for suffix in (SUFFIX,COT_SUFFIX):
            model.prompt('question',suffix=suffix)
            self.assertIs(model.tokenizer.apply_chat_template.call_args.kwargs['enable_thinking'],False)

    def test_explicit_evaluation_model_does_not_use_default(self):
        from reasoning.experiments.evaluate import selected_model
        args=SimpleNamespace(model_path=Path('/tmp/qwen'),model_id='Qwen/Qwen3-8B',
                             model_revision='pinned',device='cuda:2')
        with patch('reasoning.experiments.evaluate.open_weight_model',side_effect=AssertionError('wrong model')):
            self.assertEqual(selected_model(args),dict(model='Qwen/Qwen3-8B',revision='pinned',
                local_path='/tmp/qwen',device='cuda:2'))

    def test_cot_is_unsteered_and_resume_checks_prompt(self):
        from unittest.mock import Mock
        from reasoning.common import COT_SUFFIX
        backend = SimpleNamespace(generate=Mock(return_value=[
            dict(completion='answer', truncated=False, tokens=1)]))
        steering_generate = Mock(side_effect=AssertionError('CoT must bypass steering'))
        rows = [dict(id='a', problem='question', answer='1')]
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {'reasoning.steering':SimpleNamespace(generate=steering_generate)}):
                with patch('reasoning.evaluation.score', return_value={'correct':True}), patch('reasoning.evaluation.state'):
                    args = (backend,None,None,rows,32,'math500_cot',{},Path(directory))
                    result = evaluate(*args,prompt_suffix=COT_SUFFIX)
                    self.assertEqual(result['correct'],1)
                    backend.generate.assert_called_once_with(['question'],max_tokens=32,prompt_suffix=COT_SUFFIX)
                    self.assertEqual(evaluate(*args,prompt_suffix=COT_SUFFIX),result)
                    self.assertEqual(backend.generate.call_count,1)
                    with self.assertRaises(ValueError):
                        evaluate(*args,prompt_suffix='changed')
                    with self.assertRaises(ValueError):
                        evaluate(backend,object(),None,rows,32,'invalid',{},Path(directory),prompt_suffix=COT_SUFFIX)

    def test_code_hashes_cover_maintained_sources(self):
        expected = set(ROOT.glob('*.py'))
        for directory in ('experiments', 'datasets'):
            expected.update((ROOT / directory).glob('*.py'))
        self.assertEqual(set(code_hashes()), {str(p.relative_to(ROOT)) for p in expected})

    def test_journal_resume_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'answers.jsonl'
            journal = Journal(path, {'budget': 32})
            journal.append({'id': 'a'})
            resumed = Journal(path, {'budget': 32})
            self.assertEqual(resumed.ids, {'a'})
            with self.assertRaises(ValueError):
                resumed.append({'id': 'a'})
            with self.assertRaises(ValueError):
                Journal(path, {'budget': 64})

    def test_evaluation_resume_preserves_batches(self):
        rows = [dict(id=str(i), problem=str(i), answer='1') for i in range(5)]
        calls = []

        def generate(backend, questions, artifact, candidate, budget):
            calls.append(questions)
            return [dict(completion='answer', truncated=False) for _ in questions]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config = {'seed': 42}
            journal = Journal(output / 'eval_baseline.jsonl', dict(
                run=config, stage='eval', candidate=None, budget=32,
                ids=[r['id'] for r in rows]))
            journal.append(dict(rows[0], completion='answer', correct=True, truncated=False))
            with patch.dict(sys.modules, {'reasoning.steering': SimpleNamespace(generate=generate)}):
                with patch('reasoning.evaluation.score', return_value={'correct': True}), patch('reasoning.evaluation.state'):
                    result = evaluate(None, None, None, rows, 32, 'eval', config, output)
                    self.assertEqual(calls, [['1', '2', '3'], ['4']])
                    self.assertEqual(result['completed'], 5)
                    self.assertEqual(result['correct'], 5)
                    calls.clear()
                    self.assertEqual(evaluate(None, None, None, rows, 32, 'eval', config, output), result)
                    self.assertEqual(calls, [])

    def test_worker_arguments_keep_negative_alphas(self):
        args = SimpleNamespace(action='start', alphas=[-1, 0.5], methods=['lat', 'probe'],
                               smoke=True, output=Path('/tmp/example'))
        self.assertEqual(worker_arguments(args), [
            '--alphas=-1,0.5', '--methods', 'lat', 'probe', '--smoke', '--output', '/tmp/example'])

    def test_baseline_only_worker_omits_empty_methods(self):
        args = SimpleNamespace(action='start', baseline_only=True, cot_baseline=True, methods=[])
        self.assertEqual(worker_arguments(args), ['--baseline-only', '--cot-baseline'])

    def test_required_model_grids(self):
        cases = {
            'Qwen/Qwen3-8B': [9,12,15,18,21,24,27],
            'google/gemma-2-9b-it': [14,20,26,32],
            'meta-llama/Llama-3.1-8B-Instruct': [8,11,14,17,20,23,26],
        }
        for model,layers in cases.items():
            with self.subTest(model=model):
                grid = candidate_grid(model=model)
                self.assertEqual([(c['layer'],c['alpha']) for c in grid],
                                 [(l,a) for l in layers for a in [0.5,1.0,1.5,2.0,2.5,3.0]])
                self.assertTrue(all(c['phase']=='decode' for c in grid))
                self.assertEqual(candidate_grid(smoke=True,model=model),grid[:2])
        with self.assertRaises(ValueError):
            candidate_grid(model='unknown-model')
        self.assertEqual(candidate_grid(layers=[3],strengths=[2.0],model='custom-model'),
                         [dict(layer=3,alpha=2.0,phase='decode')])

    def test_evaluation_uses_selected_grid_vector(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            candidate=dict(layer=11,alpha=1.5,phase='decode')
            (path/'config.json').write_text(json.dumps(dict(method='probe',grid=[candidate])))
            (path/'selected_steering.json').write_text(json.dumps(dict(candidate=candidate)))
            args=SimpleNamespace(tuning_dirs=[path])
            self.assertEqual(conditions(args),{'probe':(path/'artifacts/L11',candidate)})
            (path/'selected_steering.json').write_text(json.dumps(dict(candidate=dict(candidate,layer=16))))
            with self.assertRaises(ValueError):
                conditions(args)


if __name__ == '__main__':
    unittest.main()
