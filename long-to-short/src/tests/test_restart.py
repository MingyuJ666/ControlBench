import tempfile
import unittest
from pathlib import Path

from l2s_inference.artifacts import read_jsonl


class RestartTest(unittest.TestCase):
    def test_complete_rows_resume_without_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text('{"row": 1}\n')
            self.assertEqual(read_jsonl(path, repair_tail=True), [{"row": 1}])


if __name__ == "__main__":
    unittest.main()
