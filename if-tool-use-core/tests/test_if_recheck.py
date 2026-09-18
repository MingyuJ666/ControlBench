import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import repsteer as rs
import control_bench as cb
from control_bench import paths
from control_bench.cli import run as cli, campaign
from control_bench.domains import instruction_following as domain
from control_bench.domains.instruction_following import data, judge_core, scoring, tasks
from control_bench.runtime.experiment import Run
from control_bench.runtime.search import input_fingerprint, write_json, write_jsonl


class InstructionFollowingRecheck(unittest.TestCase):
    def test_full_evaluation_uses_single_eval_file(self):
        expected=[r['prompt_id'] for r in data.load_prompts('eval')]
        self.assertEqual({p.name for p in (data.INPUTS/'prompts').glob('*.jsonl')},
                         {'extraction.jsonl', 'eval.jsonl'})
        extraction={r['prompt_id'] for r in data.load_prompts('extraction')}
        self.assertEqual(len(expected),1014)
        for target in domain.TARGETS:
            rows=domain.items(target)
            self.assertEqual([r['prompt_id'] for r in rows],expected)
            self.assertEqual(len({r['item_id'] for r in rows}),1014)
            self.assertFalse(extraction.intersection(r['prompt_id'] for r in rows))
        for split in ('dev', 'test'):
            with self.subTest(split=split), self.assertRaises(ValueError):
                domain.items('lowercase',split)

    def test_fixed_ending_must_be_a_whole_sentence(self):
        for answer in ['Hope this helps!', 'The answer is four. Hope this helps!',
                       'The answer is four. **Hope this helps!**', 'The answer is four.\n"Hope this helps!"']:
            with self.subTest(answer=answer): self.assertTrue(tasks.verify_fixed_ending(answer))
        for answer in ['NoHope this helps!', 'I Hope this helps!', 'The answer is four and Hope this helps!']:
            with self.subTest(answer=answer): self.assertFalse(tasks.verify_fixed_ending(answer))

    def test_campaign_rejects_generation_from_a_different_token_budget(self):
        items=[{'item_id':'a','base_prompt':'question'}]
        dom=SimpleNamespace(NAME='test',TARGETS=('lowercase',),items=lambda *a:items,
                            messages=lambda item:[{'role':'user','content':item['base_prompt']}],
                            rendering=lambda:cb.Rendering.chat(enable_thinking=False))
        run=Run(dom,'test/model',(),())
        with tempfile.TemporaryDirectory() as tmp, patch.object(paths,'DATA_ROOT',Path(tmp)):
            cand=run.candidates('unsteered')[0]
            out=run.directory('unsteered','lowercase',cand.id)
            write_jsonl(out/'generations.jsonl',[{'item_id':'a','output':'ok','truncated':False,'n_new_tokens':1}])
            meta={'pipeline':cb.PIPELINE_ID,'n_items':1,'input_fingerprint':input_fingerprint(items),
                  'artifact_fingerprint':None,'model':{'id':'test/model','revision':None},
                  'candidate':cand.to_dict(),'rendering':dom.rendering().generation_kwargs(),
                  'generation':{'max_new_tokens':3,'batch_size':16,'do_sample':False}}
            write_json(out/'meta.json',meta)
            self.assertFalse(run.generated('unsteered','lowercase'))
            meta['generation']=run.generation
            write_json(out/'meta.json',meta)
            self.assertTrue(run.generated('unsteered','lowercase'))

    def test_cli_rejects_invalid_requests_before_model_loading(self):
        for extra in [['--experiment','full'],['--targets','not_a_target'],['--strengths','nan'],
                      ['--strengths','0.4,0.40000001'],['--layers','-1'],['--layers','1,1']]:
            with self.subTest(extra=extra), \
                 patch.object(sys,'argv',['run','generate','--method','caa','--layers','0','--strengths','0.5',*extra]), \
                 patch.object(cb,'load_model',side_effect=AssertionError('model loading must not happen')):
                with self.assertRaises(SystemExit) as raised: cli.main()
                self.assertEqual(raised.exception.code,2)

    def test_layers_and_strengths_have_no_defaults(self):
        for method in ['caa','lat']:
            for supplied in [[],['--layers','0'],['--strengths','0.5']]:
                with self.subTest(method=method,supplied=supplied), \
                     patch.object(sys,'argv',['run','generate','--method',method,*supplied]):
                    with self.assertRaises(SystemExit) as raised: cli.main()
                    self.assertEqual(raised.exception.code,2)

    def test_generate_without_artifacts_fails_before_model_loading(self):
        run=Run(domain,'test/model',(0,),(.5,))
        with tempfile.TemporaryDirectory() as tmp,patch.object(paths,'DATA_ROOT',Path(tmp)), \
             patch.object(cb,'load_model',side_effect=AssertionError('must not load a model')):
            with self.assertRaisesRegex(ValueError,'learn'): run.generate('caa',['lowercase'])

    def test_grid_counts_and_smoke_output_isolation(self):
        for model,layers,n in [('Qwen/Qwen3-8B',(9,12,15,18,21,24,27),42),
                               ('google/gemma-2-9b-it',(11,14,17,20,23,26,29,32),48),
                               ('meta-llama/Llama-3.1-8B-Instruct',(8,11,14,17,20,23,26),42)]:
            run=Run(domain,model,layers,(.5,1,1.5,2,2.5,3))
            for method in ('caa','lat'): self.assertEqual(len(run.candidates(method)),n)
            self.assertEqual(len(run.candidates('unsteered')),1)
            self.assertEqual(run.description(('caa',),('lowercase',))['methods']['caa']['outputs'],1014*n)
            smoke=Run(domain,model,layers,(.5,),limit=3)
            self.assertNotEqual(run.directory('caa','lowercase'),smoke.directory('caa','lowercase'))
            self.assertEqual(len(smoke.items('lowercase')),3)

    @unittest.skipUnless(Path('/proc/self/stat').exists(),'Linux process state')
    def test_exited_unreaped_child_is_not_alive(self):
        proc=subprocess.Popen([sys.executable,'-c','pass'])
        try:
            for _ in range(300):
                state=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[0]
                if state=='Z': break
                time.sleep(.01)
            else: self.fail('child did not exit')
            self.assertFalse(campaign._alive(proc.pid))
        finally:
            proc.wait(timeout=5)

    def test_judge_requires_current_manifest_and_valid_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            write_jsonl(folder/'judge_in_0.jsonl',[{'uid':'u1'}])
            valid={'uid':'u1','coherent':True,'answer_status':'adequate'}
            write_jsonl(folder/'judge_out_0.jsonl',[valid])
            manifest=judge_core.run_spec(['judge_in_0.jsonl'],judge_core.JUDGE_MODEL,'vllm',{})
            write_json(folder/'judge_run.json',manifest)
            self.assertTrue(scoring.judging_complete(folder))
            write_jsonl(folder/'judge_out_0.jsonl',[{'uid':'u1'}])
            self.assertFalse(scoring.judging_complete(folder))
            write_jsonl(folder/'judge_out_0.jsonl',[valid])
            manifest['prompt_sha256']='old prompt'
            write_json(folder/'judge_run.json',manifest)
            self.assertFalse(scoring.judging_complete(folder))
            with self.assertRaisesRegex(ValueError,'prompt_sha256'): scoring.load_verdicts(folder)


if __name__=='__main__': unittest.main()
