import unittest

from l2s_inference.config import learned_cells, methods_for


class TrackTest(unittest.TestCase):
    def test_tracks_have_disjoint_steering_methods(self):
        self.assertEqual(methods_for("Llama-3.1-8B", "iti"), ["ITI"])
        self.assertNotIn("ITI", methods_for("Llama-3.1-8B", "non-iti"))
        self.assertEqual(len(list(learned_cells("Qwen3-8B", "iti"))), 5)


if __name__ == "__main__":
    unittest.main()
