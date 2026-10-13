"""The two domain-local runtimes must remain identical."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RuntimeCopiesTest(unittest.TestCase):
    def test_runtime_sources_and_dependencies_do_not_diverge(self):
        def contents(project):
            folder = ROOT / project
            files = list((folder / 'core').rglob('*.py'))
            self.assertTrue(files, f'Missing core sources in {project}')
            files += [folder / name for name in
                      ('requirements.txt', 'requirements-judge.txt')]
            return {str(p.relative_to(folder)): p.read_bytes() for p in files}

        self.assertEqual(contents('instruction-following'), contents('tool-use'),
                         'Apply runtime changes to both domain copies and run both regression suites')


if __name__ == '__main__':
    unittest.main()
