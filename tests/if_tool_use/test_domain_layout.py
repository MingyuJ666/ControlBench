"""Check imports and standalone judging across the split source directories."""

import json
import os
from pathlib import Path
from unittest.mock import patch
import subprocess
import sys
import tempfile
import unittest

from core import domains, paths
from core.cli import campaign


class DomainLayoutTest(unittest.TestCase):
    def test_core_package_and_cli_load_from_the_selected_domain(self):
        import core
        self.assertEqual(paths.PACKAGE_ROOT, paths.CORE_ROOT / 'core')
        self.assertEqual(Path(core.__file__), paths.PACKAGE_ROOT / '__init__.py')
        self.assertEqual(core.__spec__.name, 'core')
        self.assertEqual(core.Rendering.__module__, 'core.runtime.rendering')
        result = subprocess.run([sys.executable, '-m', 'core.cli.run', '--help'],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('generate', result.stdout)

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

    def test_multi_turn_legacy_imports_share_globals_in_either_import_order(self):
        # Existing callers patch domain hooks through the legacy import. Check
        # fresh interpreters so import order cannot conceal a circular import.
        legacy = "core.domains.tool_use_multi_turn"
        canonical = "core.domains.tool_use.multi_turn.domain"
        probe = """
import importlib, sys
from unittest.mock import patch
first, second = [importlib.import_module(name) for name in sys.argv[1:]]
assert first is second
old_core = importlib.import_module('core.domains.tool_use_multi_turn.judge_core')
new_core = importlib.import_module('core.domains.tool_use.multi_turn.judge_core')
assert old_core is new_core is first.judge_core
with patch.object(first, 'eval_items', return_value=['sentinel']) as items:
    assert second.items('call') == ['sentinel']
    items.assert_called_once_with('call')
from core.domains import get, names
assert get('tool_use_multi_turn') is first
assert names() == ['instruction_following', 'tool_use', 'tool_use_multi_turn']
"""
        for order in ((legacy, canonical), (canonical, legacy)):
            with self.subTest(first=order[0]):
                result = subprocess.run([sys.executable, '-c', probe, *order],
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_worker_uses_the_same_package_even_with_another_pythonpath(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            # Starting a real child catches misplaced source paths that mocked
            # scheduler tests cannot detect. It imports no model weights.
            probe = (
                "import json, core; "
                "from core import domains; "
                "print(json.dumps([core.__file__, domains.names()]))"
            )
            decoy = folder / "other" / "core"
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
