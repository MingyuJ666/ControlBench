"""Isolated synthetic unit fixtures, never benchmark results."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from safety_steering.config import load_config, configs
from safety_steering.data import read, write_new
from safety_steering import pipeline
from safety_steering.report import report
from safety_steering.storage import save_tensor
from safety_steering.sae import Dictionary, statistics
from safety_steering import validation
from safety_steering import provenance as run_records
from safety_steering.storage import sample_path

CONFIG = Path(__file__).resolve().parents[1] / "configs/v2_0916.json"


class PipelineContracts(unittest.TestCase):
    def test_run_identifier_is_stable_on_resume_and_checks_settings(self):
        with tempfile.TemporaryDirectory() as d:
            a = SimpleNamespace(out=Path(d))
            preflight = dict(inputs=dict(model_key="qwen3_8b"))
            write_new(a.out / "validation.json", dict(passed=True, preflight=preflight))
            with patch.object(
                run_records, "verified_preflight", return_value=preflight
            ):
                first = run_records.protocol(a)
                self.assertEqual(first, run_records.protocol(a))
                self.assertEqual(read(a.out / "protocol.json")["run_id"], first)
            changed = dict(inputs=dict(model_key="gemma2_9b"))
            with patch.object(run_records, "verified_preflight", return_value=changed):
                with self.assertRaises(ValueError):
                    run_records.protocol(a)

    def test_sample_filename_is_reversible_and_cannot_escape_cache(self):
        import base64

        root = Path("cache")
        sample_id = "../sample/带空格 ID"
        path = sample_path(root, sample_id)
        self.assertEqual(path.parent, root)
        encoded = path.stem
        decoded = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        self.assertEqual(decoded.decode(), sample_id)
        self.assertEqual(path, sample_path(root, sample_id))

    def test_acceptance_capture_passes_token_ids_and_configured_layers(self):
        """Exercise the real validation entrypoint through its capture call."""
        with tempfile.TemporaryDirectory() as d:
            a = SimpleNamespace(
                model="qwen3_8b",
                root=Path(d),
                out=Path(d),
                spec=load_config(CONFIG, "qwen3_8b"),
            )
            write_new(
                a.out / "preflight.json",
                dict(
                    inputs=dict(
                        model=dict(config=dict(max_position_embeddings=128)),
                        effective_max_context=128,
                    )
                ),
            )
            rows = [dict(sample_id="s", label=0), dict(sample_id="u", label=1)]

            class CaptureReached(Exception):
                pass

            with (
                patch.object(
                    validation, "model_for_run", return_value=("model", "tokenizer")
                ),
                patch.object(validation, "load", return_value=(rows, None)),
                patch.object(validation, "prompt_ids", return_value=[1, 2]),
                patch.object(
                    validation, "completion_ids", return_value=([1, 2, 3], {})
                ),
                patch.object(validation, "head_layout", return_value=[]),
                patch.object(
                    validation.torch.cuda, "get_device_name", return_value="fixture"
                ),
                patch.object(
                    validation.torch.cuda,
                    "get_device_properties",
                    return_value=SimpleNamespace(total_memory=0),
                ),
                patch.dict("os.environ", SLURM_JOB_ID="unit-fixture"),
                patch.object(
                    validation, "capture", side_effect=CaptureReached
                ) as capture,
            ):
                with self.assertRaises(CaptureReached):
                    validation.smoke(a)
                capture.assert_called_once_with(
                    "model", [1, 2, 3], [9, 18, 27], heads=True
                )

    def test_v2_grids_and_full_model_iti(self):
        for model, layers in dict(
            qwen3_8b=[9, 18, 27], gemma2_9b=[11, 21, 32], llama31_8b=[8, 16, 24]
        ).items():
            a = SimpleNamespace(model=model, spec=load_config(CONFIG, model))
            artifact = dict(
                layers={
                    layer: dict(vector=torch.tensor([3.0, 4.0])) for layer in layers
                }
            )
            for method in ("caa", "pca", "lat", "linear_probe"):
                grid = configs(a, method, artifact)
                self.assertEqual(
                    {(c["layer"], c["alpha"]) for c in grid},
                    {(layer, x) for layer in layers for x in [0.5, 1, 1.5, 2, 2.5]},
                )
                self.assertTrue(all(c["delta_norm"] == c["alpha"] * 5 for c in grid))
            selected = [dict(layer=i // 2, head=i % 2, sigma=2.0) for i in range(48)]
            grid = configs(a, "iti", dict(selected=selected))
            self.assertEqual([c["alpha"] for c in grid], [1, 5, 10, 15, 20])
            self.assertTrue(all(c["layer"] == -1 for c in grid))
            with self.assertRaises(ValueError):
                configs(a, "iti", dict(selected=selected[:-1]))
            with self.assertRaises(ValueError):
                configs(a, "sae", artifact)

    def test_sae_config_is_gemma_only_with_historical_grid(self):
        path = CONFIG.with_name("gemma_sae.json")
        for model in ("qwen3_8b", "llama31_8b"):
            with self.assertRaises(ValueError):
                load_config(path, model)
        a = SimpleNamespace(model="gemma2_9b", spec=load_config(path, "gemma2_9b"))
        artifact = dict(
            layers={
                layer: dict(vector=torch.tensor([1.0, 0.0])) for layer in (9, 20, 31)
            }
        )
        self.assertEqual(validation.validation_layers(a), [9, 20, 31])
        self.assertEqual(len(configs(a, "sae", artifact)), 18)
        self.assertEqual(max(c["alpha"] for c in configs(a, "sae", artifact)), 3.0)

    def test_gemma_jumprelu_does_not_subtract_decoder_bias(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "dictionary.npz"
            np.savez(
                p,
                W_enc=np.eye(2),
                W_dec=np.eye(2),
                b_enc=np.zeros(2),
                b_dec=np.array([100.0, 100.0]),
                threshold=np.array([0.5, 0.5]),
            )
            dictionary = Dictionary(dict(format="gemma_jumprelu", path=p), "cpu")
            torch.testing.assert_close(
                dictionary.encode(torch.tensor([[0.5, 1.0]])),
                torch.tensor([[0.0, 1.0]]),
            )
            pos = torch.tensor([[2.0, 1.0], [3.0, 2.0], [4.0, 3.0], [5.0, 4.0]])
            stats = statistics(dictionary, pos, pos - 1, [0, 0, 1, 1])
            self.assertGreater(stats["fvu"], 0.25)  # diagnostic; no admission cutoff
            self.assertGreater(stats["feature_sd"], 0)

    def test_sharded_resume_freeze_test_and_tamper_rejection(self):
        with (
            tempfile.TemporaryDirectory() as d,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            root = Path(d)
            a = SimpleNamespace(
                model="qwen3_8b",
                spec=load_config(CONFIG, "qwen3_8b"),
                root=root,
                out=root,
                shards=2,
                rank=0,
            )
            a.spec["methods"] = ["caa", "no_steering", "prompted"]
            rows = [
                dict(sample_id=str(i), task_group_id=str(i), label=i % 2)
                for i in range(35)
            ]
            datafile = root / "input.json"
            write_new(datafile, rows)
            preflight = dict(
                splits={
                    s: dict(
                        n=len(rows),
                        sample_ids=[r["sample_id"] for r in rows],
                        gold=[r["label"] for r in rows],
                    )
                    for s in ("dev", "test")
                }
            )
            write_new(root / "preflight.json", preflight)
            protocol = dict(
                inputs=dict(spec=a.spec, model_key=a.model),
                preflight=preflight,
                run_id="unit-run",
            )
            write_new(root / "protocol.json", protocol)
            provenance = protocol["run_id"]
            save_tensor(
                root / "artifacts/caa.pt",
                dict(
                    provenance=provenance,
                    layers={
                        layer: dict(vector=torch.tensor([1.0, 0.0]))
                        for layer in [9, 18, 27]
                    },
                ),
            )
            seen = []

            def predict(model, tok, batch, key, **kwargs):
                seen.extend(r["sample_id"] for r in batch)
                return [
                    dict(
                        sample_id=r["sample_id"],
                        task_group_id=r["task_group_id"],
                        gold=r["label"],
                        raw="unsafe" if r["label"] else "safe",
                        prediction=r["label"],
                        invalid=False,
                    )
                    for r in batch
                ]

            with (
                patch.object(pipeline, "load", return_value=(rows, datafile)),
                patch.object(pipeline, "model_for_run", return_value=(None, None)),
                patch.object(pipeline, "prompt_ids", return_value=[1, 2]),
                patch.object(pipeline, "prepare_prefix", return_value={}),
                patch.object(pipeline, "predict_batch", side_effect=predict),
            ):
                pipeline.calibrate(a, provenance)
                with self.assertRaises(ValueError):
                    pipeline.freeze(a, provenance)
                a.rank = 1
                pipeline.calibrate(a, provenance)
                count = len(seen)
                self.assertEqual(count, 35 * 17)
                a.rank, a.shards = 0, 1
                pipeline.calibrate(a, provenance)
                self.assertEqual(len(seen), count)  # all chunks reused on aggregation
                pipeline.freeze(a, provenance)
                chosen = read(root / "frozen.json")["configs"]["caa"]["chosen"]
                self.assertEqual((chosen["alpha"], chosen["layer"]), (0.5, 9))
                pipeline.test(a, provenance)
                report(root)
                report(root)  # idempotent CSV/JSON publication
                path = root / "test/caa-b9-a0.5/complete.json"
                cell = read(path)
                cell["predictions"][0]["raw"] = "unsafe"
                path.write_text(json.dumps(cell))
                with self.assertRaises(ValueError):
                    report(root)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
