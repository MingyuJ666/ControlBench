import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import repsteer as rs
from repsteer.artifacts import ArtifactMetadata, DirectionArtifact

import control_bench as cb
from control_bench import paths
from control_bench.runtime.experiment import Run
from control_bench.methods import learn
from control_bench.runtime import generate as generation
from test_native_methods import tiny_wrapper


def artifact(layer=0):
    return DirectionArtifact(direction=torch.ones(16), metadata=ArtifactMetadata(
        artifact_type='direction', model_id='test/model', hidden_size=16,
        method='caa', site=rs.sites.resid_post(layer)))


class RecoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_expected_artifacts_must_be_valid(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(paths,'DATA_ROOT',Path(tmp)):
            run=Run(SimpleNamespace(NAME='test'),'test/model',(0,1),(.5,))
            empty=cb.artifact_dir('test','test/model','caa','target','resid_post_0')
            empty.mkdir(parents=True)
            self.assertFalse(run.learned('caa','target'))
            cb.save_artifact(artifact(0),domain='test',model='test/model',method='caa',target='target',force=True)
            self.assertFalse(run.learned('caa','target'))
            second=cb.save_artifact(artifact(1),domain='test',model='test/model',method='caa',target='target')
            self.assertTrue(run.learned('caa','target'))
            (second/'tensors.safetensors').write_bytes(b'partial')
            self.assertFalse(run.learned('caa','target'))

    def test_explicit_revision_rejects_artifacts_from_another_revision(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(paths, 'DATA_ROOT', Path(tmp)):
            run = Run(SimpleNamespace(NAME='test'), 'test/model', (0,), (.5,), revision='new')
            cb.save_artifact(learn.with_config(artifact(), requested_revision='old'),
                             domain='test', model='test/model', method='caa', target='target')
            self.assertFalse(run.learned('caa', 'target'))
            cb.save_artifact(learn.with_config(artifact(), requested_revision='new'),
                             domain='test', model='test/model', method='caa', target='target', force=True)
            self.assertTrue(run.learned('caa', 'target'))

    def test_failed_force_save_preserves_previous_artifact(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(paths,'DATA_ROOT',Path(tmp)):
            dest=cb.save_artifact(artifact(),domain='test',model='test/model',method='caa',target='target')
            original=(dest/'tensors.safetensors').read_bytes()
            with patch.object(learn,'save_native',side_effect=RuntimeError('interrupted write')):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):
                    cb.save_artifact(artifact(),domain='test',model='test/model',method='caa',target='target',force=True)
            self.assertEqual((dest/'tensors.safetensors').read_bytes(),original)

    def test_learning_resume_skips_valid_sites_and_repairs_partial_sites(self):
        wrapper=SimpleNamespace(close=lambda:None, model=__import__("torch").nn.Identity())
        model=SimpleNamespace(config=SimpleNamespace(num_hidden_layers=2))
        run=Run(SimpleNamespace(NAME='test'),'test/model',(0,1),(.5,),batch=2)
        with tempfile.TemporaryDirectory() as tmp, patch.object(paths,'DATA_ROOT',Path(tmp)), \
             patch.object(cb,'load_model',return_value=(wrapper,model,None)):
            cb.save_artifact(artifact(),domain='test',model='test/model',method='caa',target='target')
            cb.artifact_dir('test','test/model','caa','target','resid_post_1').mkdir(parents=True)
            calls=[]
            def fit(wrapper,domain,target,sites,**kwargs):
                calls.extend(s.layer for s in sites)
                return {f'resid_post_{s.layer}':artifact(s.layer) for s in sites}
            from control_bench.methods import METHODS
            with patch.dict(METHODS,{'caa':SimpleNamespace(name='caa',learn=fit,supports=lambda _:True)}):
                run.learn('caa',['target'])
            self.assertEqual(calls,[1])

    def test_eos_at_token_limit_is_a_complete_answer(self):
        wrapper=tiny_wrapper()
        try:
            tok=wrapper.tokenizer
            for pad_equals_eos in (False,True):
                with self.subTest(pad_equals_eos=pad_equals_eos):
                    if pad_equals_eos: tok.pad_token=tok.eos_token
                    def raw(wrapper,inputs,*args,**kwargs):
                        prefix=tok(inputs,add_special_tokens=False,return_tensors='pt')['input_ids']
                        return SimpleNamespace(sequences=torch.cat([prefix,torch.tensor([[6,tok.eos_token_id]])],dim=1))
                    with patch.object(generation,'generate',side_effect=raw):
                        row=generation.generate_texts(wrapper,['answer'],batch_size=1,max_new_tokens=2)[0]
                    self.assertFalse(row['truncated'])
                    self.assertEqual(row['n_new_tokens'],2)
        finally:
            wrapper.close()


if __name__=='__main__': unittest.main()
