"""Check imports and standalone judging across the split source directories."""

import json
import os
from pathlib import Path
from unittest.mock import patch
import subprocess
import sys
import tempfile
import unittest

from control_bench import domains, paths
from control_bench.cli import campaign


class DomainLayoutTest(unittest.TestCase):
    def test_standalone_judge_finds_all_domain_rubrics_without_site_packages(self):
        script = paths.PACKAGE_ROOT / "cli" / "judge_vllm.py"
        probe = (
            "import json, runpy, sys; "
            "load = runpy.run_path(sys.argv[1])['load_core']; "
            "print(json.dumps({name: load(name).JUDGE_MODEL for name in sys.argv[2:]}))"
        )
        names = domains.names()
        self.assertEqual(names, ["instruction_following", "tool_use", "tool_use_multi_turn"])
        result = subprocess.run(
            [sys.executable, "-S", "-c", probe, str(script), *names],
            check=True, text=True, capture_output=True,
        )
        self.assertEqual(set(json.loads(result.stdout)), set(names))
        for name in names:
            self.assertTrue((domains.get(name).INPUTS / "MANIFEST.json").is_file())

    def test_worker_uses_the_same_package_even_with_another_pythonpath(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            # Starting a real child catches misplaced source paths that mocked
            # scheduler tests cannot detect. It imports no model weights.
            probe = (
                "import json, control_bench; "
                "from control_bench import domains; "
                "print(json.dumps([control_bench.__file__, domains.names()]))"
            )
            decoy = folder / "other" / "control_bench"
            decoy.mkdir(parents=True)
            (decoy / "__init__.py").write_text("raise AssertionError('wrong checkout')\n")
            with patch.dict(os.environ, {"PYTHONPATH": str(decoy.parent)}):
                job = campaign.Job(
                    "layout", [sys.executable, "-c", probe], [],
                    request={}, running_dir=folder, log_dir=folder,
                    run_id="layout-test", devices=[],
                )
            try:
                self.assertEqual(job.proc.wait(timeout=120), 0)
            finally:
                if job.proc.poll() is None:
                    job.proc.kill()
                    job.proc.wait()
            self.assertTrue(job.done())
            log = next(folder.glob("*.log")).read_text().strip().splitlines()[-1]
            package, names = json.loads(log)
            self.assertEqual(Path(package).parent, paths.PACKAGE_ROOT)
            self.assertEqual(names, domains.names())

    def test_external_data_root_keeps_the_original_output_layout(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"STEERING_DATA": tmp}), \
                patch.object(paths, "DATA_ROOT", Path(tmp)):
            for name in ("instruction_following", "tool_use", "tool_use_multi_turn"):
                self.assertEqual(paths.output_root(name), Path(tmp) / name / "outputs" / paths.PIPELINE_ID)
            for name in ("instruction_following", "tool_use"):
                self.assertEqual(paths.input_root(name), Path(tmp) / name / "inputs")
