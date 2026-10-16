"""Paths, configuration, checkpoints and background experiment workers."""
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUTPUTS = ROOT / 'outputs'
NEW_OUTPUTS = OUTPUTS / 'v2'
LOCAL_CONFIG = ROOT / 'models.local.json'
SPLIT = Path(os.environ.get('REASONING_SPLIT', DATA / 'split.json')).resolve()


# User-defined search space; layer numbers are zero-based model block indices.
LAYER_GRID = {
    'Qwen3-8B': [9, 12, 15, 18, 21, 24, 27],
    'Gemma-2-9B': [14, 20, 26, 32],
    'Llama-3.1-8B': [8, 11, 14, 17, 20, 23, 26],
}
STRENGTH_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# One generation budget shared by every stage (screen, confirmation/dev, test).
MAX_TOKENS = 2048


def steering_grid(model):
    name = model.rsplit('/', 1)[-1].lower()
    for family, layers in LAYER_GRID.items():
        if name in (family.lower(), family.lower()+'-instruct', family.lower()+'-it'):
            return list(layers), list(STRENGTH_GRID)
    raise ValueError(f'No steering grid for {model}; specify --layers and --strengths')


def steering_artifact_path(root, candidate):
    """Use separate artifacts for positions in a joint search."""
    root = Path(root)
    position = candidate.get('extraction_position')
    if position is not None:
        if position not in ('before_answer', 'after_answer'):
            raise ValueError(f'Invalid candidate extraction position: {position}')
        root = root / position
    return root / f'L{candidate["layer"]}'


SYSTEM = 'You are a helpful assistant that solves mathematical problems.'
SUFFIX = '\nGive a clear mathematical solution. End with the final answer in \\boxed{}.'

COT_SUFFIX = "\nLet's think step by step." + SUFFIX


def load_local():
    return json.loads(LOCAL_CONFIG.read_text()) if LOCAL_CONFIG.exists() else {}

def open_weight_model(index=0):
    models = load_local().get('open_weight_models', [])
    if not 0 <= index < len(models):
        raise ValueError('No configured open-weight model at this index')
    return dict(models[index])

def api_credentials(default_base_url=None):
    api = load_local().get('api', {})
    keys, index = api.get('api_keys', []), api.get('active_key_index', 0)
    if keys and (not isinstance(index, int) or not 0 <= index < len(keys)):
        raise ValueError('Invalid active_key_index in local configuration')
    key = os.environ.get('OPENAI_API_KEY') or (keys[index] if keys else None)
    base_url = os.environ.get('OPENAI_BASE_URL') or api.get('base_url') or default_base_url
    return key, base_url


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class Journal:
    """Single-writer checkpoint; reject changes to inputs or settings on resume."""

    def __init__(self, path, config):
        self.path = Path(path)
        meta = self.path.with_suffix(self.path.suffix + '.meta.json')
        if meta.exists():
            if json.loads(meta.read_text()) != config:
                raise ValueError(f'Run configuration changed: use a new output path ({self.path})')
        elif self.path.exists():
            raise ValueError(f'Output exists without metadata: {self.path}')
        else:
            write_json(meta, config)
        self.rows = read_jsonl(self.path) if self.path.exists() else []
        self.ids = {r['id'] for r in self.rows}
        if len(self.ids) != len(self.rows):
            raise ValueError('Duplicate journal IDs')

    def append(self, row):
        if row['id'] in self.ids:
            raise ValueError('Duplicate journal ID')
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()
        self.rows.append(row)
        self.ids.add(row['id'])


def state(output, stage, **fields):
    value = dict(stage=stage, pid=os.getpid(),
                 updated_at=datetime.now(timezone.utc).isoformat(), **fields)
    write_json(Path(output)/'status.json', value)
    print(json.dumps(value), flush=True)


def read_status(output):
    output = Path(output)
    path = output/'status.json'
    result = json.loads(path.read_text()) if path.exists() else {'stage':'not_started'}
    result['worker_lock_held'] = False
    lock_path = output/'worker.lock'
    if lock_path.exists():
        with lock_path.open('r') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                result['worker_lock_held'] = True
    return result


def worker_arguments(args):
    values = []
    for key, value in vars(args).items():
        if key == 'action' or value is None:
            continue
        option = '--'+key.replace('_','-')
        if isinstance(value, bool):
            if value:
                values.append(option)
        elif key == 'alphas':
            values.append(option+'='+','.join(map(str,value)))
        elif isinstance(value, list):
            if value:
                values.extend([option,*map(str,value)])
        else:
            values.extend([option,str(value)])
    return values


def dispatch(args, run, module):
    args.output = args.output.resolve()
    if args.action == 'status':
        print(json.dumps(read_status(args.output),indent=2))
        return
    args.output.mkdir(parents=True,exist_ok=True)
    inherited = os.environ.pop('REASONING_WORKER_LOCK_FD',None)
    lock = (os.fdopen(int(inherited),'a') if inherited is not None else
            (args.output/'worker.lock').open('a'))
    try:
        if inherited is None:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit('This output directory already has a running worker')
        if args.action == 'start':
            command = [sys.executable,'-u','-m',module,'run',*worker_arguments(args)]
            with (args.output/'run.log').open('a') as log:
                process = subprocess.Popen(command,cwd=ROOT.parent,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=log,start_new_session=True,pass_fds=(lock.fileno(),),
                    env=dict(os.environ,REASONING_WORKER_LOCK_FD=str(lock.fileno()),OMP_NUM_THREADS='4'))
            print(json.dumps(dict(pid=process.pid,output=str(args.output))))
            return
        try:
            run(args)
        except Exception as exc:
            state(args.output,'failed',error_type=type(exc).__name__,error=str(exc))
            traceback.print_exc()
            raise SystemExit(1)
    finally:
        lock.close()


def code_hashes():
    """Hash maintained code only; exclude credentials, data and frozen sources."""
    paths = list(ROOT.glob('*.py'))
    paths += list((ROOT / 'datasets').glob('*.py'))
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def prepare_run(output, config, device):
    """Validate resume configuration and wait for enough GPU memory."""
    import time
    import torch
    path = output / 'config.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Config changed; use a new output directory')
    write_json(path, config)
    state(output, 'waiting_for_gpu', device=device)
    while torch.cuda.mem_get_info(device)[0] < 24 * 1024**3:
        time.sleep(30)
    state(output, 'loading_model')
