import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from reasoning.baseline_cache import reuse, signature
from reasoning.common import Journal, SYSTEM, SUFFIX, COT_SUFFIX
from reasoning.datasets import problem_id
from reasoning.evaluation import evaluate


class BaselineReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = dict(model='llama', revision='pinned', max_tokens=32,
            batch_size=2, do_sample=False, seed=42, system=SYSTEM, suffix=SUFFIX)
        self.rows = [dict(id=problem_id(str(i)),problem=str(i),answer='1') for i in range(3)]
        self.meta = dict(run=self.config,candidate=None,stage='math500_baseline',budget=32)
        self.source = self.root/'source.jsonl'

    def save(self, rows=None, meta=None):
        records = [{**r, 'completion':r'\boxed{1}', 'tokens':4, 'truncated':False,
                    'correct':False} for r in (self.rows if rows is None else rows)]
        self.source.write_text(''.join(json.dumps(r)+'\n' for r in records))
        self.source.with_suffix('.jsonl.meta.json').write_text(json.dumps(meta or self.meta))

    def test_full_reuse_and_resume_never_generate(self):
        self.save()
        generate = Mock(side_effect=AssertionError('must not generate'))
        with patch('reasoning.baseline_cache.sources',return_value=[self.source]), patch.dict(
                'sys.modules',{'reasoning.steering':SimpleNamespace(generate=generate)}), patch(
                'reasoning.evaluation.score',return_value=dict(correct=True)):
            result=evaluate(None,None,None,self.rows,32,'math500_baseline',self.config,self.root,2)
            self.assertEqual(result['correct'],3)
            evaluate(None,None,None,self.rows,32,'math500_baseline',self.config,self.root,2)
        generate.assert_not_called()
        rows=[json.loads(s) for s in (self.root/'math500_baseline_baseline.jsonl').read_text().splitlines()]
        self.assertTrue(all(r['baseline_source']['path']==str(self.source) for r in rows))

    def test_partial_cache_only_generates_missing_questions(self):
        self.save(self.rows[:1])
        generate=Mock(side_effect=lambda backend, questions, *a:[
            dict(completion='answer',tokens=1,truncated=False) for _ in questions])
        with patch('reasoning.baseline_cache.sources',return_value=[self.source]), patch.dict(
                'sys.modules',{'reasoning.steering':SimpleNamespace(generate=generate)}), patch(
                'reasoning.evaluation.score',return_value=dict(correct=True)):
            result=evaluate(None,None,None,self.rows,32,'math500_baseline',self.config,self.root,2)
        self.assertEqual(result['completed'],3)
        self.assertEqual([c.args[1] for c in generate.call_args_list],[['1'],['2']])

    def test_reject_mismatched_settings_and_steering(self):
        mutations=[{'model':'other'},{'revision':'other'},{'batch_size':4},
                   {'backend':'vllm'},{'enable_thinking':True},{'temperature':1}]
        for i, change in enumerate(mutations):
            self.save(meta={**self.meta,'run':{**self.config,**change}})
            journal=Journal(self.root/f'out{i}.jsonl',{})
            with patch('reasoning.baseline_cache.sources',return_value=[self.source]):
                self.assertEqual(reuse(journal,self.rows,self.meta,32),0)
        for i,change in enumerate([{'budget':64},{'prompt_suffix':COT_SUFFIX},
                                   {'candidate':{'layer':1,'alpha':1}}]):
            self.save(meta={**self.meta,**change})
            journal=Journal(self.root/f'extra{i}.jsonl',{})
            with patch('reasoning.baseline_cache.sources',return_value=[self.source]):
                self.assertEqual(reuse(journal,self.rows,self.meta,32),0)

    def test_reject_wrong_answer_and_duplicate_ids(self):
        for i,rows in enumerate([[{**self.rows[0],'answer':'2'}],[self.rows[0],self.rows[0]]]):
            self.save(rows)
            with patch('reasoning.baseline_cache.sources',return_value=[self.source]):
                self.assertEqual(reuse(Journal(self.root/f'bad{i}.jsonl',{}),self.rows,self.meta,32),0)

    def test_dev_budget_is_distinct_from_test_budget(self):
        meta=dict(run={**self.config,'max_tokens':8192,'dev_tokens':2048},candidate=None,stage='dev')
        self.assertEqual(signature(meta)['budget'],2048)

if __name__=='__main__':
    unittest.main()
