"""CPU integration tests; no downloads or benchmark model required."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import boxed, load, problem_id, score


class PipelineTest(unittest.TestCase):
    def test_boxed_and_grading(self):
        self.assertEqual(boxed(r"\boxed{0} then \boxed{\frac{1}{2}}"), r"\frac{1}{2}")
        self.assertIsNone(boxed(r"\boxed{unfinished"))
        self.assertTrue(score(r"\boxed{\frac{1}{2}}", "0.5")["correct"])
        self.assertFalse(score("the answer is 2", "2")["correct"])

    def test_real_data(self):
        suite, pairs = load(ROOT / "data")
        self.assertEqual([len(suite[k]) for k in ("extraction", "dev", "eval")], [200, 20, 280])
        self.assertEqual(len(pairs), 56)

    def test_cli_and_artifacts(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model_dir, data_dir = root / "model", root / "data"
            model_dir.mkdir()
            data_dir.mkdir()
            vocab = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "positive": 3,
                     "negative": 4, "solution": 5, "problem": 6}
            tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]",
                                           pad_token="[PAD]", eos_token="[EOS]")
            fast.chat_template = "{% for m in messages %}{{ m['content'] + ' ' }}{% endfor %}solution "
            fast.save_pretrained(model_dir)
            torch.manual_seed(42)
            LlamaForCausalLM(LlamaConfig(
                vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                max_position_embeddings=256, pad_token_id=1, eos_token_id=2,
            )).save_pretrained(model_dir)
            rows = [{"problem": f"problem {i}", "answer": "1"} for i in range(12)]
            ids = [problem_id(r["problem"]) for r in rows]
            split = dict(extraction=ids[:8], dev=ids[8:10], eval=ids[10:])
            pairs = [dict(id=ids[i], problem=rows[i]["problem"],
                          positive="solution " * (i + 1) + "positive",
                          negative="solution " * (i + 1) + "negative") for i in range(8)]
            for filename, values in [("problems.jsonl", rows), ("pairs.jsonl", pairs)]:
                (data_dir / filename).write_text("".join(json.dumps(r) + "\n" for r in values))
            (data_dir / "split.json").write_text(json.dumps(split))

            def run(command, name, *extra, success=True):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "run.py"), command, "--model", str(model_dir),
                     "--data", str(data_dir), "--device", "cpu", "--max-tokens", "2",
                     "--batch-size", "2", "--output", str(root / name), *extra],
                    text=True, capture_output=True, timeout=120,
                )
                if success:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                return root / name

            for style in ("normal", "cot"):
                output = run("eval", style, "--method", style)
                self.assertEqual(json.loads((output / "summary.json").read_text())["count"], 2)
            run("eval", "normal", "--method", "normal", success=False)
            for method in ("diffmean", "lat", "probe"):
                tuned = run("tune", method, "--method", method, "--layers", "0",
                            "--strengths", "-1", "1")
                evaluated = run("eval", method + "_eval", "--selection", str(tuned / "selected.json"))
                self.assertEqual(json.loads((evaluated / "summary.json").read_text())["count"], 2)
            run("eval", "wrong_seed", "--selection", str(tuned / "selected.json"),
                "--seed", "1", "--dry-run", success=False)
            split["dev"].append(split["extraction"][0])
            (data_dir / "split.json").write_text(json.dumps(split))
            with self.assertRaisesRegex(ValueError, "overlapping"):
                load(data_dir)


if __name__ == "__main__":
    unittest.main()
