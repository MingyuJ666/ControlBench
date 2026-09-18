import unittest

from l2s_inference.settings import batches


class BatchTest(unittest.TestCase):
    def test_batches_preserve_order_and_final_partial_group(self):
        self.assertEqual(list(batches(list(range(5)), 2)), [[0, 1], [2, 3], [4]])


if __name__ == "__main__":
    unittest.main()
