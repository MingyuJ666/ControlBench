import argparse
import contextlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from core.cli.campaign import child_command
from core.cli.options import add_run_options, make_run
from core.domains import instruction_following as domain
from core.domains.instruction_following.data import load_prompts
from core.runtime.experiment import Run


def run_for(split="full", **kwargs):
    return Run(domain, "Qwen/Qwen3-8B", (9, 18, 27), (.5, 1.), eval_split=split, **kwargs)


class EvalSplitTests(unittest.TestCase):
    def test_original_partition_and_shared_questions(self):
        source = [r["prompt_id"] for r in load_prompts("eval")]
        extraction = {r["prompt_id"] for r in load_prompts("extraction")}
        for target in domain.TARGETS:
            ids = {s: [r["prompt_id"] for r in domain.items(target, s)]
                   for s in ("full", "dev", "test")}
            self.assertEqual(ids["full"], source)
            self.assertEqual(ids["dev"], source[:405])
            self.assertEqual(ids["test"], source[405:])
            self.assertEqual(len(ids["test"]), 609)
            self.assertFalse(set(ids["dev"]) & set(ids["test"]))
            self.assertFalse(extraction & set(source))

    def test_full_backward_compatibility(self):
        run = run_for(sample=200, train_sample=100)
        self.assertEqual(run.experiment, "full-sample-200-seed-42-train-sample-100-seed-42")
        indices = sorted(random.Random(42).sample(range(1014), 200))
        source = load_prompts("eval")
        items = run.items("lowercase")
        self.assertEqual([r["prompt_id"] for r in items], [source[i]["prompt_id"] for i in indices])
        self.assertTrue(all(r["item_id"] == f"lowercase_full_{r['prompt_id']}" for r in items))

    def test_sampling_within_split_and_limits(self):
        for split, size in (("dev", 405), ("test", 609)):
            run = run_for(split, sample=37, seed=7)
            source = domain.items("lowercase", split)
            indices = sorted(random.Random(7).sample(range(size), 37))
            self.assertEqual(run.items("lowercase"), [source[i] for i in indices])
            shared = [r["prompt_id"] for r in run.items("lowercase")]
            for target in domain.TARGETS:
                self.assertEqual([r["prompt_id"] for r in run.items(target)], shared)
            self.assertEqual(run_for(split, limit=3).items("lowercase"), source[:3])
            with self.assertRaises(ValueError):
                run_for(split, sample=size + 1).items("lowercase")

    def test_output_isolation_and_training_reuse(self):
        runs = [run_for(s, sample=20, train_sample=7) for s in ("full", "dev", "test")]
        self.assertEqual(len({r.directory("caa", "lowercase") for r in runs}), 3)
        self.assertEqual(len({r.artifact_directory("caa", "lowercase", "resid_post_9") for r in runs}), 1)
        self.assertEqual(len({str(r.training_pairs("lowercase").fingerprint) for r in runs}), 1)
        self.assertEqual(len({r.score_fingerprint("caa", "lowercase") for r in runs}), 3)

    def test_cli_and_campaign_forwarding(self):
        parser = argparse.ArgumentParser()
        add_run_options(parser)
        args = parser.parse_args(["--eval-split", "test", "--targets", "lowercase",
                                  "--layers", "18", "--strengths", "1.5", "--sample", "17"])
        run, targets = make_run(parser, args, ("caa",))
        child = child_command(run, "generate", "caa", targets)
        self.assertEqual(child[child.index("--eval-split") + 1], "test")
        self.assertEqual(len(run.items("lowercase")), 17)
        self.assertEqual(len(run.candidates("caa")), 1)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            args = parser.parse_args(["--domain", "tool_use", "--eval-split", "dev"])
            make_run(parser, args, ("unsteered",))

    def test_judging_and_scoring_receive_split(self):
        for split in ("dev", "test"):
            run = run_for(split, limit=2)
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "judging"
                with (patch.object(run, "directory", return_value=out),
                      patch.object(run, "generated", return_value=True),
                      patch.object(run, "judged", return_value=False),
                      patch.object(domain, "prepare_judging") as prepare,
                      patch.object(domain, "judging_complete", return_value=False),
                      patch.object(domain, "judge") as judge):
                    run.judge("unsteered", ["lowercase"])
                    self.assertEqual(prepare.call_args.args[-1], split)
                    judge.assert_called_once_with([out], force=False)
                with (patch.object(run, "directory", return_value=out),
                      patch.object(run, "generated", return_value=True),
                      patch.object(run, "judged", return_value=True),
                      patch.object(run, "scored", return_value=False),
                      patch.object(domain, "metrics", return_value={"accuracy": .5}) as metrics,
                      contextlib.redirect_stdout(io.StringIO())):
                    run.score("unsteered", ["lowercase"])
                    self.assertEqual(metrics.call_args.args[6], split)
                    self.assertEqual(json.loads((Path(tmp) / "summary.json").read_text())["split"], split)


if __name__ == "__main__":
    unittest.main()
