import tempfile
import unittest
from pathlib import Path

from l2s.craft_pairs import load_pairs


class PairTest(unittest.TestCase):
    def test_reads_64_distinct_pairs_directly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "extraction.jsonl"
            path.write_text("".join(f'{{"pair_id":"p{i}","math_id":"train/{i}"}}\n' for i in range(64)))
            self.assertEqual(len(load_pairs(tmp)), 64)


if __name__ == "__main__":
    unittest.main()
