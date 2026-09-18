import unittest

from l2s_inference.settings import eval_sets_for, token_limit


class EvalSettingsTest(unittest.TestCase):
    def test_math500_is_default_and_amc12_is_optional(self):
        self.assertEqual(eval_sets_for(), ["math500"])
        self.assertEqual(eval_sets_for(["math500", "amc12"]), ["math500", "amc12"])

    def test_token_cap(self):
        self.assertEqual(token_limit(8192), 3072)


if __name__ == "__main__":
    unittest.main()
