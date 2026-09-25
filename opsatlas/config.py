import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def load_config():
    load_dotenv(ROOT / '.env', override=False)
    path = Path(os.getenv('OPS_CONFIG', 'config/app.json'))
    if not path.is_absolute():
        path = ROOT / path
    config = json.loads(path.read_text(encoding='utf-8'))
    for collection in ('agents', 'mcp_servers'):
        ids = [x['id'] for x in config[collection]]
        if len(ids) != len(set(ids)) or any(not re.fullmatch(r'[a-z][a-z0-9_]{0,19}', x) for x in ids):
            raise ValueError('Agent/server identifiers must be unique short lowercase names')
    for key in ('knowledge_dir', 'index_path'):
        config[key] = str((ROOT / config[key]).resolve())
    for key in ('max_rounds', 'max_tool_calls', 'max_delegations', 'tool_timeout', 'top_k'):
        if not isinstance(config[key], int) or not 1 <= config[key] <= 120:
            raise ValueError(f'Invalid limit: {key}')
    return config
