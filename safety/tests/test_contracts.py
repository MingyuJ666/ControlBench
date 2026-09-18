import unittest
from unittest.mock import Mock
import tempfile
from pathlib import Path
import torch
from safety_steering.data import write_new, snapshot
from safety_steering.metrics import parse_label, metrics
from safety_steering.runtime import intervene, head_layout
from safety_steering.learners import fit_residual, probe, weights
from safety_steering.storage import save_tensor, read_tensor


class Contracts(unittest.TestCase):
    def test_tensor_checkpoint_roundtrip_and_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "state.pt"
            save_tensor(p, {"provenance": "test", "vector": torch.arange(4)})
            torch.testing.assert_close(
                read_tensor(p, "test")["vector"], torch.arange(4)
            )
            with self.assertRaises(ValueError):
                read_tensor(p, "wrong")
            with self.assertRaises(FileExistsError):
                save_tensor(p, {"provenance": "other"})

    def test_strict_parser(self):
        self.assertEqual(parse_label(" safe\n"), 0)
        self.assertEqual(parse_label("unsafe"), 1)
        for text in [
            "SAFE",
            "This is safe",
            "safe safe",
            "safe.",
            "unsafe/safe",
            "",
            "<think>safe</think>",
        ]:
            self.assertIsNone(parse_label(text), text)

    def test_invalid_denominators(self):
        rows = [
            {"sample_id": str(i), "gold": g, "prediction": p}
            for i, (g, p) in enumerate(
                [(0, 0), (0, 1), (0, None), (1, 1), (1, 0), (1, None)]
            )
        ]
        m = metrics(rows)
        self.assertEqual(m["accuracy"], 2 / 6)
        self.assertEqual(m["safe_false_positive_rate"], 1 / 3)
        self.assertEqual(m["unsafe_f1"], 2 / 5)
        self.assertEqual(m["fn"], 2)
        self.assertEqual(m["invalid_rate"], 1 / 3)

    def test_atomic_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            write_new(p, {"x": 1})
            with self.assertRaises(FileExistsError):
                write_new(p, {"x": 2})
            self.assertIn("1", p.read_text())

    def test_whitelist(self):
        row = {
            "task_description": "read",
            "input": [],
            "output": "inert",
            "label": 1,
            "note": "answer",
            "split": "test",
        }
        self.assertEqual(set(snapshot(row)), {"task_description", "input", "output"})

    def test_distinct_pca_lat(self):
        # Shared high-variance context dominates pooled PCA; pair subtraction cancels it.
        pos = torch.tensor([[10.0, -100.0], [10.0, 100.0], [10.0, -50.0], [10.0, 50.0]])
        neg = pos - torch.tensor([20.0, 0.0])
        labels = [0, 1, 0, 1]
        pca = fit_residual("pca", pos, neg, labels)["vector"]
        lat = fit_residual("lat", pos, neg, labels)["vector"]
        self.assertGreater(abs(float(pca[1])), 100 * abs(float(pca[0])))
        self.assertGreater(abs(float(lat[0])), 100 * abs(float(lat[1])))

    def test_probe_agrees_with_repsteer(self):
        from repsteer.learners.probe import fit_logistic_probe

        torch.manual_seed(8)
        x = torch.randn(40, 5)
        y = (x[:, 0] + x[:, 2] > 0.1).float()
        w = torch.arange(1, 41).float()
        a, b, info = probe(x, y, w)
        ref, rb = fit_logistic_probe(x, y, sample_weight=w)
        torch.testing.assert_close(a, ref, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(b, rb, atol=2e-3, rtol=2e-3)

    def test_probe_intercept_centering_preserves_objective(self):
        torch.manual_seed(12)
        x = torch.randn(80, 6, dtype=torch.float64)
        y = (x[:, 0] + 0.4 * x[:, 1] > 0).double()
        w = torch.arange(1, 81).double()
        offset = torch.tensor(
            [120.0, -80.0, 30.0, 50.0, -20.0, 10.0], dtype=torch.float64
        )
        a, b, info = probe(x, y, w)
        c, d, shifted = probe(x + offset, y, w)
        torch.testing.assert_close(a, c, atol=3e-4, rtol=3e-4)
        torch.testing.assert_close(
            b.double(), d.double() + offset @ c.double(), atol=3e-4, rtol=3e-4
        )
        self.assertAlmostEqual(info["objective"], shifted["objective"], places=10)
        self.assertLess(shifted["max_gradient"], 2e-5)

    def test_gemma_layout_and_prefill_hook(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = torch.nn.Module()
                self.self_attn.o_proj = torch.nn.Linear(12, 10, bias=False)

            def forward(self, x):
                return x * 2

        model = Mock()
        block = Block()
        model.model.layers = [block]
        model.config.num_attention_heads = 3
        model.config.head_dim = 4
        self.assertEqual(head_layout(model), [(3, 4, 12)])
        x = torch.zeros(1, 6, 10)
        events = []
        with intervene(model, {0: torch.ones(10)}, telemetry=events):
            y = block(x)
            self.assertTrue((y[:, -1] == 1).all())
            self.assertTrue((y[:, :-1] == 0).all())
        self.assertEqual(events[0]["sequence_length"], 6)
        self.assertTrue((block(x) == 0).all())
        with intervene(model, {0: torch.zeros(10)}):
            torch.testing.assert_close(block(x), x)

    def test_class_balance(self):
        w = weights([0, 0, 0, 1])
        self.assertAlmostEqual(float(w[:3].sum()), float(w[3]))


if __name__ == "__main__":
    unittest.main()
