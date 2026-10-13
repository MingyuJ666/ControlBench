#!/usr/bin/env python3
"""Launch a ControlBench domain in its own environment and working directory.

Only this module's standard-library dependencies are needed to list or preview
commands. Arguments after -- belong to the existing domain CLI.
"""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parent
ENTRIES = {
    'instruction_following': ('campaign', 'run', 'selfcheck'),
    'tool_use': ('campaign', 'run', 'selfcheck'),
    'tool_use_multi_turn': ('campaign', 'run', 'selfcheck'),
    'reasoning': ('main',),
    'safety': ('main',),
    'long_to_short': ('non-iti', 'iti', 'inference', 'pairs'),
}


@dataclass
class Launch:
    domain: str
    entry: str
    python: str
    cwd: Path
    argv: list
    environment: dict

    def preview(self):
        return {
            'domain': self.domain, 'entry': self.entry, 'python': self.python,
            'cwd': str(self.cwd), 'argv': self.argv,
            'environment_overrides': self.environment,
        }


def interpreter(value):
    # Keep the virtualenv symlink: resolve() would select the system interpreter.
    expanded = os.path.expanduser(value)
    found = shutil.which(expanded)
    if found is None:
        raise ValueError(f'Python interpreter is not executable or not found: {value}')
    return os.path.abspath(found)


def plan(domain, entry, python, arguments):
    domain = domain.replace('-', '_')
    if domain not in ENTRIES:
        raise ValueError(f'Unknown domain {domain!r}; choose from {", ".join(ENTRIES)}')
    entry = entry or ENTRIES[domain][0]
    if entry not in ENTRIES[domain]:
        raise ValueError(f'{domain} entries: {", ".join(ENTRIES[domain])}')
    selected = interpreter(python)
    environment = {}
    cwd = ROOT
    if domain in ('instruction_following', 'tool_use', 'tool_use_multi_turn'):
        if any(arg == '--domain' or arg.startswith('--domain=') for arg in arguments):
            raise ValueError('Select --domain before the -- separator only')
        project = 'instruction-following' if domain == 'instruction_following' else 'tool-use'
        package = str(ROOT / project)
        previous = os.environ.get('PYTHONPATH')
        environment['PYTHONPATH'] = package + (os.pathsep + previous if previous else '')
        argv = [selected, '-m', f'core.cli.{entry}', *arguments, '--domain', domain]
    elif domain == 'reasoning':
        argv = [selected, '-m', 'reasoning', *arguments]
    elif domain == 'safety':
        argv = [selected, str(ROOT / 'safety/run.py'), *arguments]
    else:
        bash = shutil.which('bash')
        if bash is None:
            raise ValueError('Long-to-short requires bash on PATH')
        scripts = {
            'non-iti': 'inference_non_iti.sh', 'iti': 'inference_iti.sh',
            'inference': 'inference.sh', 'pairs': 'craft_pairs.sh',
        }
        cwd = ROOT / 'long-to-short'
        argv = [os.path.abspath(bash), str(cwd / 'src' / scripts[entry]),
                '--python', selected, *arguments]
    return Launch(domain, entry, selected, cwd, argv, environment)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    arguments = []
    if '--' in argv:
        boundary = argv.index('--')
        argv, arguments = argv[:boundary], argv[boundary + 1:]
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--domain', help='domain ID; use --list to see choices')
    parser.add_argument('--entry', help='native entry point; first entry from --list is the default')
    parser.add_argument('--python', default=sys.executable, metavar='EXECUTABLE',
                        help='domain Python interpreter (default: the Python running bench.py)')
    parser.add_argument('--preview', action='store_true',
                        help='print the command and working directory without executing or loading ~/.env')
    parser.add_argument('--list', action='store_true', help='list domains and native entry points')
    args = parser.parse_args(argv)
    if args.list:
        for domain, entries in ENTRIES.items():
            print(f'{domain:24} {", ".join(entries)}')
        return
    if not args.domain:
        parser.error('--domain is required; use --list to see choices')
    try:
        launch = plan(args.domain, args.entry, args.python, arguments)
    except ValueError as error:
        parser.error(str(error))
    if args.preview:
        print(json.dumps(launch.preview(), indent=2))
        return
    # Replace the launcher so signals, stdin/stdout and exit status reach the
    # existing CLI directly. It retains its own scheduling and resume behavior.
    try:
        os.chdir(launch.cwd)
        os.execvpe(launch.argv[0], launch.argv, {**os.environ, **launch.environment})
    except OSError as error:
        parser.exit(1, f'{parser.prog}: cannot launch {launch.domain}: {error}\n')


if __name__ == '__main__':
    main()
