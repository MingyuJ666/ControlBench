import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import repsteer as rs
import core as cb
from core import paths
from core.cli import run as run_cli, campaign, judge_vllm
from core.domains import instruction_following as domain
from core.domains.instruction_following import judge_core
from core.runtime.experiment import Run
from test_native_methods import tiny_wrapper, training_pairs


class CampaignFlowTest(unittest.TestCase):
    def test_three_baselines_full_grid_scoring_and_restart(self):
        self._run_three_baselines()

    def test_three_baselines_sampled_grid_scoring_and_restart(self):
        self._run_three_baselines(sample=2, seed=137)

    def test_three_baselines_training_and_evaluation_sampled(self):
        self._run_three_baselines(sample=2, seed=137, train_sample=3, train_seed=7)

    def test_iti_mixed_with_layer_methods_scoring_and_restart(self):
        self._run_three_baselines(sample=2, train_sample=4, include_iti=True)

    def test_all_methods_scoring_and_restart(self):
        self._run_three_baselines(sample=2, train_sample=4, include_iti=True, include_all=True)

    def test_explicit_output_root_scoring_and_restart(self):
        self._run_three_baselines(sample=2, train_sample=3, explicit_output=True)

    def test_prompted_only_scoring_and_restart(self):
        self._run_three_baselines(sample=2, baseline_only=True)

    def test_prompted_mixed_with_caa_lat_scoring_and_restart(self):
        self._run_three_baselines(sample=2, train_sample=3, include_prompted=True)

    def _run_three_baselines(self, sample=None, seed=42, train_sample=None, train_seed=42,
                             include_iti=False, include_all=False, include_prompted=False, baseline_only=False, explicit_output=False):
        torch.set_num_threads(2)
        wrapper = tiny_wrapper(num_hidden_layers=2)
        model, tok, model_id = wrapper.model, wrapper.tokenizer, wrapper.model_id
        encoded, labels = [], []
        for color in ['red', 'blue']:
            rendered = tok.apply_chat_template([{'role': 'user', 'content': f'answer {color}'}],
                                               tokenize=False, add_generation_prompt=True)
            prompt = tok.encode(rendered, add_special_tokens=False)
            answer = tok.encode(color, add_special_tokens=False) + [tok.eos_token_id]
            encoded.append(prompt + answer)
            labels.append([-100] * len(prompt) + answer)
        batch, labels = torch.tensor(encoded), torch.tensor(labels)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.02)
        for _ in range(60):
            optimizer.zero_grad()
            model(input_ids=batch, labels=labels).loss.backward()
            optimizer.step()
        model.eval()
        wrapper.close()
        targets = tuple(domain.TARGETS)
        methods = ('unsteered', 'caa', 'lat', 'iti') if include_iti else ('unsteered', 'caa', 'lat')
        if include_all:
            methods = ('unsteered', 'prompted', 'caa', 'linear_probe', 'lat', 'pca', 'iti')
        elif include_prompted:
            methods = ('unsteered', 'prompted', 'caa', 'lat')
        elif baseline_only:
            methods = ('prompted',)
        judge_calls, jobs = [], []
        verdict_status = ["adequate"]

        def items(target, split='full'):
            self.assertEqual(split, 'full')
            return [{'item_id': f'{target}-{split}-{i}', 'prompt_id': f'{split}-{i}',
                     'base_prompt': f'answer {color}', 'prompted_prompt': f'answer {color} lowercase'}
                    for i, color in enumerate(['red', 'blue'] * (4 if sample is not None else 1))]

        def load_model(*args, **kwargs):
            fresh = rs.models.from_model(model, tok, model_id=model_id, revision='test-12')
            return fresh, model, tok

        class InlineJob:
            def __init__(self, name, argv, gpus, **kwargs):
                self.name, self.gpus, self.request = name, gpus, kwargs['request']
                jobs.append(name)
                with patch.object(sys, 'argv', ['cbench-run', *argv[3:]]):
                    run_cli.main()
                self.returncode = 0
            def done(self):
                return True

        def verdict_outputs(prompts, *args, **kwargs):
            judge_calls.extend(prompts)
            return [SimpleNamespace(outputs=[SimpleNamespace(text=json.dumps({
                'id': re.search(r'"id": "([0-9a-f]{12})"', p).group(1),
                'coherent': True, 'answer_status': verdict_status[0]}))]) for p in prompts]

        def judge(dirs, force=False):
            # Controlled model responses; actual shard parsing and scoring code.
            for folder in dirs:
                if force:
                    for cached in folder.glob('judge_out_*.jsonl'):
                        cached.unlink()
                shards = sorted(folder.glob('judge_in_*.jsonl'))
                (folder / 'judge_run.json').write_text(json.dumps(judge_core.run_spec(
                    [p.name for p in shards], judge_core.JUDGE_MODEL, 'vllm', {})))
                for source in shards:
                    for row in map(json.loads, source.read_text().splitlines()):
                        self.assertIn(row['question'], ('answer red', 'answer blue'))
                    judge_vllm.judge_shard(SimpleNamespace(generate=verdict_outputs), tok,
                                           judge_core, None, source)

        ticks = [0]
        def sleep(_):
            ticks[0] += 1
            if ticks[0] > 100:
                raise AssertionError('campaign did not finish')

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(paths, 'DATA_ROOT', Path(tmp) / 'data'), \
             paths.output_directory(Path(tmp) / 'chosen output' if explicit_output else None), \
             patch.object(domain, 'items', side_effect=items), \
             patch.object(domain, 'pairs', side_effect=lambda _: training_pairs()), \
             patch.object(domain, 'judge', side_effect=judge), \
             patch.object(cb, 'load_model', side_effect=load_model), \
             patch.object(campaign, 'Job', InlineJob), \
             patch.object(campaign.time, 'sleep', side_effect=sleep), \
             patch.object(sys, 'argv', ['cbench-campaign', '--model', model_id, '--layers', '0,1',
                                       '--strengths', '0.5,1', '--batch', '2', '--max-new', '3',
                                       '--gpus', '0,1,2',
                                       *(['--out', str(Path(tmp) / 'chosen output')] if explicit_output
                                         else ['--logs', str(Path(tmp) / 'logs')]),
                                       *(['--methods', ','.join(methods)] if include_iti or include_prompted or baseline_only else []),
                                       *(['--iti-top-k', '2', '--iti-seed', '13'] if include_iti else []),
                                       *(['--sample', str(sample), '--seed', str(seed)] if sample is not None else []),
                                       *(['--train-sample', str(train_sample), '--train-seed', str(train_seed)]
                                         if train_sample is not None else [])]):
            campaign.main()
            run = Run(domain, model_id, (0, 1), (.5, 1.), batch=2, max_new=3, sample=sample, seed=seed,
                      train_sample=train_sample, train_seed=train_seed, iti_top_k=2 if include_iti else 48,
                      iti_seed=13 if include_iti else 42)
            if train_sample is not None:
                for method in ('caa', 'lat'):
                    for target in targets:
                        for block in (0, 1):
                            art = cb.load_artifact(domain.NAME, model_id, method, target, f'resid_post_{block}',
                                                   training=run.training)
                            self.assertEqual(art.metadata.config['training_selection']['sample'], train_sample)
                            self.assertEqual(art.metadata.config['pairs'], train_sample)
            if explicit_output:
                self.assertFalse((Path(tmp) / 'data').exists())
                self.assertTrue((Path(tmp) / 'chosen output' / 'artifacts').is_dir())
                self.assertTrue(list((Path(tmp) / 'chosen output' / 'logs').rglob('state.json')))
            total_outputs = 0
            for method in methods:
                for target in targets:
                    self.assertTrue(run.scored(method, target), (method, target))
                    parent = run.directory(method, target).parent
                    summary = json.loads((parent / 'summary.json').read_text())
                    self.assertEqual(summary['split'], 'full')
                    self.assertEqual(len(summary['rows']), 1 if method in ('unsteered', 'prompted') else 2 if method == 'iti' else 4)
                    self.assertFalse((parent / 'selection.json').exists())
                    for candidate in run.candidates(method):
                        out = run.directory(method, target, candidate.id)
                        result = json.loads((out / 'summary.json').read_text())
                        self.assertEqual(result['n'], 2)
                        self.assertGreaterEqual(result['accuracy'], 0.)
                        self.assertLessEqual(result['accuracy'], 1.)
                        generations = [json.loads(line) for line in (out / 'generations.jsonl').read_text().splitlines()]
                        self.assertEqual([r['item_id'] for r in generations],
                                         [r['item_id'] for r in run.items(target)])
                        total_outputs += len(generations)
            self.assertEqual(total_outputs, 200 if include_all else 110 if include_iti else 100 if include_prompted else 10 if baseline_only else 90)
            self.assertEqual(len([j for j in jobs if j.startswith('learn_')]), 5 if include_all else 3 if include_iti else 0 if baseline_only else 2)
            self.assertGreater(len(judge_calls), 0)
            with patch.object(campaign, 'Job', side_effect=AssertionError('unexpected rerun')):
                campaign.main()
            # A changed verdict must replace the old metric without regenerating.
            baseline = 'prompted' if baseline_only or include_prompted or include_all else 'unsteered'
            folder = run.directory(baseline, 'lowercase', baseline)
            before_generations = (folder / 'generations.jsonl').read_bytes()
            self.assertEqual(json.loads((folder / 'summary.json').read_text())['accuracy'], 1.)
            verdict_status[0] = 'partial'
            run.judge(baseline, ['lowercase'], force=True)
            self.assertFalse(run.scored(baseline, 'lowercase'))
            run.score(baseline, ['lowercase'])
            self.assertEqual(json.loads((folder / 'summary.json').read_text())['accuracy'], 0.)
            self.assertEqual((folder / 'generations.jsonl').read_bytes(), before_generations)



if __name__ == '__main__':
    unittest.main()
