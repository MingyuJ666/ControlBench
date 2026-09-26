"""Single user entry point with lazy command imports."""
import argparse
import importlib
import json
import sys
from .common import OUTPUTS, read_status

COMMANDS = {
    'suite':'reasoning.scripts.paper_baselines',
    'evaluate':'reasoning.experiments.evaluate',
    'teacher-repair':'reasoning.experiments.teacher',
    'sample':'reasoning.experiments.sample',
    'compare':'reasoning.experiments.compare',
    'tune':'reasoning.experiments.tune',
    'reverse':'reasoning.experiments.reverse',
}


def main():
    parser = argparse.ArgumentParser(description='Reasoning: LLM / steering / baseline / datasets / GPT workflows')
    parser.add_argument('command',choices=[*COMMANDS,'status'])
    parser.add_argument('arguments',nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == 'status':
        if args.arguments:
            parser.error('status takes no additional arguments')
        runs = {}
        for base in [OUTPUTS,OUTPUTS/'v2']:
            if base.exists():
                for path in sorted(base.iterdir()):
                    if path.is_dir() and (path/'status.json').exists():
                        runs[str(path.relative_to(OUTPUTS))] = read_status(path)
        print(json.dumps(runs,indent=2))
        return
    module = importlib.import_module(COMMANDS[args.command])
    sys.argv = [f'python -m reasoning {args.command}',*args.arguments]
    module.main()
