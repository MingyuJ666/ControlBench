"""Check model routing, split isolation, and immutable pair snapshots."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from reasoning.datasets import problem_id
from reasoning.datasets.contrastive import negative_path, pair_records, prepare_pairs


class ContrastiveTests(unittest.TestCase):
    def setUp(self):
        self.model = 'Qwen/Qwen3-8B'
        self.originals = [dict(problem=f'Question {i}') for i in range(500)]
        self.ids = [problem_id(r['problem']) for r in self.originals]
        self.split = dict(extraction=self.ids[:200], dev=self.ids[200:220], eval=self.ids[220:])
        self.positive = [dict(r, id=i, solution='teacher', correct=False) for r, i in zip(self.originals,self.ids)]
        self.negative = [dict(r, id=i, completion='student', model=self.model, correct=True) for r, i in zip(self.originals,self.ids)]

    def pairs(self, **overrides):
        args = dict(positive=self.positive, negative=self.negative, originals=self.originals,
                    split=self.split, model=self.model)
        return pair_records(**dict(args, **overrides))

    def test_model_routing(self):
        for model, family in [('meta-llama/Llama-3.1-8B-Instruct','llama'),
                              ('google/gemma-2-9b-it','gemma'), ('Qwen/Qwen3-8B','qwen')]:
            self.assertEqual(negative_path(model).name, f'negative_{family}.jsonl')
        with self.assertRaises(ValueError):
            negative_path('unknown')

    def test_all_200_extraction_ids_without_correctness_filter(self):
        rows = self.pairs()
        self.assertEqual([r['id'] for r in rows], self.split['extraction'])
        self.assertEqual(len(rows), 200)
        self.assertFalse({r['id'] for r in rows} & set(self.split['dev']+self.split['eval']))
        self.assertTrue(all(r['negative']=='student' and 'llama' not in r for r in rows))

    def test_partial_duplicate_wrong_model_and_problem_rejected(self):
        cases = [self.negative[:-1], self.negative+[self.negative[0]],
                 [dict(r, model='wrong') for r in self.negative]]
        wrong = copy.deepcopy(self.negative)
        wrong[0]['problem']='wrong question'
        cases.append(wrong)
        for rows in cases:
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                self.pairs(negative=rows)

    def test_overlap_rejected(self):
        split = copy.deepcopy(self.split)
        split['extraction'][0]=split['eval'][0]
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.pairs(split=split)

    def test_snapshot_resume_and_changed_source_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def dump(name, rows):
                (root/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            dump('original.jsonl',self.originals)
            dump('positive.jsonl',self.positive)
            dump('negative_qwen.jsonl',self.negative)
            (root/'split.json').write_text(json.dumps(self.split))
            with patch('reasoning.datasets.contrastive.DATA',root), patch('reasoning.datasets.contrastive.SPLIT',root/'split.json'), patch('reasoning.datasets.contrastive.negative_path',return_value=root/'negative_qwen.jsonl'):
                path=prepare_pairs(self.model,root/'run')
                self.assertEqual(len(path.read_text().splitlines()),200)
                self.assertEqual(prepare_pairs(self.model,root/'run'),path)
                changed=copy.deepcopy(self.negative)
                changed[0]['completion']='changed'
                dump('negative_qwen.jsonl',changed)
                with self.assertRaisesRegex(ValueError,'sources changed'):
                    prepare_pairs(self.model,root/'run')


if __name__ == '__main__':
    unittest.main()
