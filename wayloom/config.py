import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def env_value(name, default=None):
    """Prefer the new setting; accept existing deployment keys without rewriting secrets."""
    if name in os.environ:
        return os.environ[name]
    if name.startswith('WAYLOOM_'):
        return os.getenv('SMARTVOYAGE_' + name.removeprefix('WAYLOOM_'), default)
    return default


def load_config():
    load_dotenv(ROOT / '.env', override=True)
    path = Path(env_value('WAYLOOM_CONFIG', 'config/travel.json'))
    if not path.is_absolute():
        path = ROOT / path
    config = json.loads(path.read_text(encoding='utf-8'))
    for collection in ('agents', 'mcp_servers'):
        ids = [item['id'] for item in config[collection]]
        if len(ids) != len(set(ids)) or any(not re.fullmatch(r'[a-z][a-z0-9_]{0,19}', x) for x in ids):
            raise ValueError('Agent/server IDs must be unique short identifiers')
    for key in ('knowledge_dir', 'index_path', 'booking_db'):
        config[key] = str((ROOT / config[key]).resolve())
    return config


def integrations():
    load_config()
    return {'model': bool(env_value('WAYLOOM_API_KEY') and env_value('WAYLOOM_MODEL')
                          and env_value('WAYLOOM_BASE_URL')),
            'weather': True, 'places': bool(os.getenv('AMAP_API_KEY')),
            'live_search': bool(os.getenv('TAVILY_API_KEY')),
            'tickets': bool(env_value('WAYLOOM_TICKET_BASE_URL')),
            'embeddings': bool(env_value('WAYLOOM_EMBEDDING_MODEL'))}
