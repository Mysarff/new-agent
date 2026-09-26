import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def load_config():
    load_dotenv(ROOT / '.env', override=True)
    path = Path(os.getenv('SMARTVOYAGE_CONFIG', 'config/travel.json'))
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
    return {'model': bool(os.getenv('SMARTVOYAGE_API_KEY') and os.getenv('SMARTVOYAGE_MODEL')
                          and os.getenv('SMARTVOYAGE_BASE_URL')),
            'weather': True, 'places': bool(os.getenv('AMAP_API_KEY')),
            'live_search': bool(os.getenv('TAVILY_API_KEY')),
            'tickets': bool(os.getenv('SMARTVOYAGE_TICKET_BASE_URL')),
            'embeddings': bool(os.getenv('SMARTVOYAGE_EMBEDDING_MODEL'))}
