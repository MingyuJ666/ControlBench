import unittest

from l2s_inference.progress import Progress


class ProgressTest(unittest.TestCase):
    def test_progress_reaches_completion(self):
        progress = Progress(2)
        progress.advance("test")
        progress.advance("test")
        self.assertEqual(progress.completed, 2)


if __name__ == "__main__":
    unittest.main()
