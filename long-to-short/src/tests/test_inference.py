import tempfile
import unittest
from pathlib import Path

from l2s_inference.artifacts import read_jsonl, repair_jsonl_tail


class ResumeTest(unittest.TestCase):
    def test_repairs_only_an_incomplete_final_jsonl_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_bytes(b'{"row": 1}\n{"row":')
            repair_jsonl_tail(path)
            self.assertEqual(read_jsonl(path), [{"row": 1}])
            self.assertTrue((Path(tmp) / "rows.jsonl.partial.bak").exists())


if __name__ == "__main__":
    unittest.main()
