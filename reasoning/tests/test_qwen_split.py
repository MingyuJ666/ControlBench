"""Qwen split migration preserves test IDs and excludes dev from extraction."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from reasoning.common import ROOT, read_jsonl
from reasoning.datasets import load
from reasoning.datasets.benchmarks import audit_split
from reasoning.datasets.contrastive import prepare_pairs
from reasoning.scripts.qwen_pipeline import rebalance_split

class QwenSplitTests(unittest.TestCase):
    def setUp(self):
        self.original = json.loads((ROOT/'outputs/archive/data/math500/steering_run/split.json').read_text())
        self.split = rebalance_split(self.original)

    def test_migration_and_audit(self):
        self.assertEqual([len(self.split[k]) for k in ('extraction','dev','eval')], [160,60,280])
        self.assertEqual(self.split['eval'], self.original['eval'])
        self.assertEqual(self.split['dev'][:20], self.original['dev'])
        self.assertEqual(set(self.split['dev'][20:]), set(self.original['extraction']) - set(self.split['extraction']))
        self.assertEqual(self.split, rebalance_split(self.original))
        historical = json.loads((ROOT/'outputs/repsteer_tuning/config.json').read_text())
        audit = audit_split(self.split, load(), [], historical)
        self.assertEqual(audit['confirmation_eval_overlap'], 0)
        self.assertFalse(audit['confirmation_is_dev_tail'])
        bad = dict(self.split, extraction=self.split['extraction'][:-1]+[self.split['dev'][0]])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            audit_split(bad, load(), [], historical)

    def test_new_pairs_snapshot(self):
        model = 'Qwen/Qwen3-8B'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = load()
            datasets = {'original.jsonl': rows,
                        'positive.jsonl': [dict(r, solution='teacher') for r in rows],
                        'negative_qwen.jsonl': [dict(r, completion='student', model=model) for r in rows]}
            for name, records in datasets.items():
                (root/name).write_text(''.join(json.dumps(r)+'\n' for r in records))
            (root/'split.json').write_text(json.dumps(self.split))
            with patch('reasoning.datasets.contrastive.DATA', root), patch('reasoning.datasets.contrastive.SPLIT', root/'split.json'), patch('reasoning.datasets.contrastive.negative_path', return_value=root/'negative_qwen.jsonl'):
                path = prepare_pairs(model, root/'run')
                self.assertEqual([r['id'] for r in read_jsonl(path)], self.split['extraction'])
                self.assertEqual(prepare_pairs(model, root/'run'), path)
                (root/'split.json').write_text(json.dumps(self.original))
                with self.assertRaisesRegex(ValueError, 'sources changed'):
                    prepare_pairs(model, root/'run')
