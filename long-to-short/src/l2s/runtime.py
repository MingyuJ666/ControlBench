"""Load the experiment environment before importing Hugging Face packages."""
import os
from pathlib import Path

CACHE_KEYS = ('HF_HOME', 'HF_HUB_CACHE', 'HF_DATASETS_CACHE', 'TRANSFORMERS_CACHE')


def load_environment(teacher=True):
    from dotenv import dotenv_values, load_dotenv
    path = Path.home() / '.env'
    if not path.is_file():
        raise RuntimeError(f'Required environment file is missing: {path}')
    keys = CACHE_KEYS + (('OPENAI_API_KEY', 'OPENAI_BASE_URL') if teacher else ())
    values = dotenv_values(path)
    missing = [key for key in keys if not values.get(key)]
    if missing:
        raise RuntimeError('Missing variables in ~/.env: ' + ', '.join(missing))
    load_dotenv(path, override=True)
    for key in CACHE_KEYS:
        if not Path(os.environ[key]).is_absolute():
            raise RuntimeError(f'{key} must be an absolute path')
