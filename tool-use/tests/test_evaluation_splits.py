"""CPU-only checks of the actual data loaders; no model dependencies required."""

import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

# Import the data modules directly, bypassing package initializers that load
# model runtimes. These namespace packages point to the production source files.
for name, directory in {
    "_split_check": ROOT / "core",
    "_split_check.domains": ROOT / "src",
    "_split_check.domains.tool_use": ROOT / "src/tool_use",
    "_split_check.domains.tool_use.single_turn": ROOT / "src/tool_use/single_turn",
    "_split_check.domains.tool_use.multi_turn": ROOT / "src/tool_use/multi_turn",
}.items():
    package = ModuleType(name)
    package.__path__ = [str(directory)]
    sys.modules[name] = package

single = importlib.import_module("_split_check.domains.tool_use.single_turn.items")
multi = importlib.import_module("_split_check.domains.tool_use.multi_turn.items")
splits = importlib.import_module("_split_check.domains.tool_use.multi_turn.splits")


class EvaluationSplitTests(unittest.TestCase):
    def test_full_matches_pre_split_experiment_inputs(self):
        # Captured from the actual loaders at commit 513f8be, using the runtime's
        # input_fingerprint serialization. Protect content, order and cache IDs.
        fingerprints = {
            single: {
                "call": "505d21eff31e1de8d20fc505b765784e0e92f0708062f3ce13a8867b7fd9c819",
                "clarify": "cf4d4bb589633c103c560e46df1b424ed33dab995ce66dd6b9b1be240aaffd7e",
                "abstain": "71a5095918206466e0c92f7e79e88423d841243baa37dc4c1dd2e746afad4ea5",
            },
            multi: {
                "call": "46676f6c1ee3b591efca5d7af622be33542288377b6fe73c12226bfad53df427",
                "clarify": "6ecaddb851b12615ed61cf23a239f8df9c1c7c27b2fa5d4343d1cc1e9dd3bceb",
                "abstain": "236cd079defe20d1acb47131240e6cb3203cd0fb0ef342a818a839b37500ce41",
            },
        }
        for module, targets in fingerprints.items():
            for target, expected in targets.items():
                with self.subTest(module=module.__name__, target=target):
                    rows = module.eval_items(target)
                    actual = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                    self.assertEqual(actual, expected)

    def test_counts_coverage_and_file_order(self):
        expected = {
            single: {"call": (607, 619), "clarify": (436, 437), "abstain": (506, 504)},
            multi: {target: (100, 100) for target in ("call", "clarify", "abstain")},
        }
        for module, counts in expected.items():
            original = module.load_items()
            for target, (dev_count, test_count) in counts.items():
                with self.subTest(module=module.__name__, target=target):
                    full = module.eval_items(target)
                    dev = module.eval_items(target, "dev")
                    test = module.eval_items(target, "test")
                    self.assertEqual(full, [r for r in original if r["target"] == target])
                    self.assertEqual((len(dev), len(test)), (dev_count, test_count))
                    dev_ids = {r["item_id"] for r in dev}
                    test_ids = {r["item_id"] for r in test}
                    self.assertFalse(dev_ids & test_ids)
                    self.assertEqual(dev_ids | test_ids, {r["item_id"] for r in full})
                    self.assertEqual(dev, [r for r in full if r["item_id"] in dev_ids])
                    self.assertEqual(test, [r for r in full if r["item_id"] in test_ids])

    def test_single_turn_lineages_are_disjoint_across_targets(self):
        rows = single.load_items()
        self.assertFalse({r["lineage"] for r in rows if r["split"] == "dev"}
                         & {r["lineage"] for r in rows if r["split"] == "test"})

    def test_invalid_selection_is_rejected(self):
        for module in (single, multi):
            with self.subTest(module=module.__name__):
                with self.assertRaises(ValueError):
                    module.eval_items("call", "validation")
                with self.assertRaises(ValueError):
                    module.eval_items("unknown", "dev")

    def test_single_turn_rejects_leaking_lineage(self):
        row = single.load_items()[0]
        other = copy.deepcopy(row)
        other.update(item_id="different_id", split="test" if row["split"] == "dev" else "dev")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval.jsonl"
            path.write_text("\n".join(map(json.dumps, (row, other))) + "\n")
            with patch.object(single, "EVAL", path):
                self.assertEqual(len(single.eval_items(row["target"], "full")), 2)
            with patch.object(single, "EVAL", path), self.assertRaisesRegex(ValueError, "lineage"):
                single.eval_items("call", "dev")

    def test_full_single_turn_does_not_require_split_metadata(self):
        row = single.load_items()[0]
        row.pop("split")
        row.pop("lineage")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval.jsonl"
            path.write_text(json.dumps(row) + "\n")
            with patch.object(single, "EVAL", path):
                self.assertEqual(single.eval_items(row["target"]), [row])
                with self.assertRaises(ValueError):
                    single.eval_items(row["target"], "dev")

    def test_multi_turn_split_is_reproducible_and_grouped(self):
        rows = multi.load_items()
        saved = json.loads(multi.SPLIT.read_text())
        digest = hashlib.sha256(multi.MULTI_TURN.read_bytes()).hexdigest()
        self.assertEqual(splits.build_split(rows, digest), saved)
        dev, test = set(saved["dev"]), set(saved["test"])
        for group in splits.scenario_groups(rows):
            self.assertFalse(dev.intersection(group) and test.intersection(group))

    def test_full_multi_turn_does_not_require_split_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(multi, "SPLIT", Path(directory) / "absent.json"):
                self.assertEqual(len(multi.eval_items("call", "full")), 200)

    def test_multi_turn_rejects_corrupt_partitions(self):
        saved = json.loads(multi.SPLIT.read_text())
        bad_records = []
        stale = copy.deepcopy(saved)
        stale["source_sha256"] = "0" * 64
        bad_records.append(stale)
        missing = copy.deepcopy(saved)
        missing["dev"].pop()
        bad_records.append(missing)
        duplicate = copy.deepcopy(saved)
        duplicate["dev"].append(duplicate["dev"][0])
        bad_records.append(duplicate)
        leaked = copy.deepcopy(saved)
        leaked["dev"][0], leaked["test"][0] = leaked["test"][0], leaked["dev"][0]
        bad_records.append(leaked)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            for index, record in enumerate(bad_records):
                with self.subTest(case=index):
                    path.write_text(json.dumps(record))
                    with patch.object(multi, "SPLIT", path), self.assertRaises(ValueError):
                        multi.eval_items("call", "dev")


if __name__ == "__main__":
    unittest.main()
