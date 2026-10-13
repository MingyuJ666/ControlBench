"""CPU-only SAE contract checks: python -m core.cli.selfcheck_sae."""
import tempfile
import unittest
from pathlib import Path

import torch
import repsteer as rs
from repsteer.artifacts import ArtifactMetadata, load_artifact, save_artifact
from repsteer.sae import select_supervised_features
from repsteer.operators import Add

from core.methods import METHODS
from core.methods.sae import GemmaScopeAdapter, MODEL, MODEL_REVISION, validate_run
from core.runtime.experiment import Run
from core import domains


def tiny_adapter():
    return GemmaScopeAdapter({'w_enc': torch.eye(2), 'w_dec': torch.tensor([[2., 0.], [0., 3.]]),
        'b_enc': torch.tensor([0.5, -0.5]), 'b_dec': torch.tensor([10., 20.]),
        'threshold': torch.tensor([1., 1.])})


class SAEChecks(unittest.TestCase):
    def test_raw_jumprelu_and_decoder_bias(self):
        sae = tiny_adapter()
        # x == threshold is inactive. b_dec must not be subtracted during encode.
        x = torch.tensor([[0.5, 1.5], [1.5, 2.5], [-2., -2.]])
        torch.testing.assert_close(sae.encode(x), torch.tensor([[0., 0.], [2., 2.], [0., 0.]]))
        torch.testing.assert_close(sae.decode(torch.tensor([[2., 2.]])), torch.tensor([[14., 26.]]))
        with self.assertRaises(IndexError):
            sae.decoder_direction(2)

    def test_official_uppercase_weight_names(self):
        base = tiny_adapter()
        weights = {"W_enc": base.w_enc, "W_dec": base.w_dec, "b_enc": base.b_enc,
                   "b_dec": base.b_dec, "threshold": base.threshold}
        loaded = GemmaScopeAdapter(weights)
        torch.testing.assert_close(loaded.encode(torch.ones(1, 2)), base.encode(torch.ones(1, 2)))
        with self.assertRaises(ValueError):
            GemmaScopeAdapter({**weights, "w_enc": base.w_enc})

    def test_training_selection_and_portable_direction(self):
        sae = tiny_adapter()
        # The largest absolute gap belongs to feature 0 but is negative.
        # Positive-only ranking must choose feature 1.
        acts = torch.tensor([[2., 4.], [2., 4.], [12., 2.], [12., 2.]])
        meta = ArtifactMetadata(model_id=MODEL, model_revision=MODEL_REVISION,
                                site=rs.sites.resid_post(20), method='sae')
        art, = select_supervised_features(sae, acts, torch.tensor([1, 1, 0, 0]),
                top_k=1, absolute=False, sample_weight=torch.tensor([1., 2., 1., 2.]), metadata=meta)
        self.assertEqual(art.feature_id, 1)
        with tempfile.TemporaryDirectory() as tmp:
            save_artifact(art, Path(tmp)/'artifact')
            restored = load_artifact(Path(tmp)/'artifact')
            self.assertEqual(restored.fingerprint(), art.fingerprint())
            torch.testing.assert_close(restored.decoder_direction, torch.tensor([0., 3.]))
            x = torch.ones(1, 2)
            torch.testing.assert_close(Add().apply(x, restored, 0., None), x)
            torch.testing.assert_close(Add().apply(x, restored, 2., None), torch.tensor([[1., 7.]]))
            plan = METHODS['sae'].plan(restored, {'strength': .1})
            self.assertEqual(len(plan.to_dict()['interventions']), 2)

    def test_model_revision_layer_and_domain_guards(self):
        validate_run(MODEL, MODEL_REVISION, (9, 20, 31))
        for model, revision, layers in [('Qwen/Qwen3-8B', MODEL_REVISION, (20,)),
                (MODEL, None, (20,)), (MODEL, MODEL_REVISION, (21,))]:
            with self.assertRaises(ValueError):
                validate_run(model, revision, layers)
        for name in ('instruction_following', 'tool_use', 'tool_use_multi_turn'):
            self.assertTrue(METHODS['sae'].supports(name))
            run = Run(domains.get(name), MODEL, (9, 20, 31), (.5, 1., 1.5, 2., 2.5),
                      revision=MODEL_REVISION, train_sample=100, sample=20)
            run.check_method('sae')
            self.assertEqual(len(run.candidates('sae')), 15)
        self.assertFalse(METHODS['sae'].supports('unknown'))

    def test_tool_training_and_evaluation_isolation(self):
        runs = [Run(domains.get(name), MODEL, (20,), (.5,), revision=MODEL_REVISION,
                    train_sample=100, sample=20) for name in ('tool_use', 'tool_use_multi_turn')]
        for target in ('call', 'clarify', 'abstain'):
            self.assertEqual(str(runs[0].training_pairs(target).fingerprint),
                             str(runs[1].training_pairs(target).fingerprint))
            self.assertNotEqual(runs[0].artifact_directory('sae', target, 'resid_post_20'),
                                runs[1].artifact_directory('sae', target, 'resid_post_20'))
            self.assertEqual(len(runs[0].items(target)), 20)
            self.assertEqual(len(runs[1].items(target)), 20)
        self.assertEqual(runs[0].domain.rendering(), runs[1].domain.rendering())

    def test_protocol_and_sampling_are_reproducible(self):
        dom = domains.get('instruction_following')
        run = Run(dom, MODEL, (20,), (0., .1, .25), revision=MODEL_REVISION,
                  train_sample=100, train_seed=42, sample=200, seed=42)
        other = Run(dom, MODEL, (9, 20), (0., .1), revision=MODEL_REVISION,
                    train_sample=100, train_seed=42, sample=20, seed=42)
        self.assertEqual(run.learning_config('sae', 'lowercase'), other.learning_config('sae', 'lowercase'))
        self.assertEqual(len(run.training_pairs('lowercase')), 100)
        self.assertEqual(len(run.items('lowercase')), 200)
        self.assertNotEqual(run.experiment, other.experiment)
        self.assertEqual(run.description(['sae'], ['lowercase'])['methods']['sae']['top_k'], 1)


if __name__ == '__main__':
    unittest.main()
