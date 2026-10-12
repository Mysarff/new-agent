import argparse
import asyncio
import json
import logging

from .config import load_config
from .knowledge import ingest
from .mcp_client import ToolHub
from .model import Embeddings
from .runtime import chat, direct_tool
from .diagnostics import check_connections


def main():
    # Third-party INFO logs may include workspace-specific API hosts.
    for name in ('httpx', 'httpcore', 'python_a2a', 'urllib3'):
        logging.getLogger(name).setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(description='Wayloom travel agent')
    sub = parser.add_subparsers(dest='command', required=True)
    indexing = sub.add_parser('ingest')
    indexing.add_argument('--dense', action='store_true')
    asking = sub.add_parser('chat')
    asking.add_argument('query')
    asking.add_argument('--network', action='store_true')
    search = sub.add_parser('search')
    search.add_argument('query')
    search.add_argument('--date')
    search.add_argument('--region')
    sub.add_parser('discover')
    checking = sub.add_parser('check', help='调用一次真实模型，并检查独立专家进程')
    checking.add_argument('--network', action='store_true')
    args = parser.parse_args()
    config = load_config()
    if args.command == 'ingest':
        result = ingest(config, Embeddings() if args.dense else None)
    elif args.command == 'chat':
        result = asyncio.run(chat(args.query, network=args.network))
    elif args.command == 'search':
        result = asyncio.run(direct_tool('travel__search_knowledge',
                                         {'query': args.query, 'travel_date': args.date, 'region': args.region}))
    elif args.command == 'check':
        result = asyncio.run(check_connections(config, network=args.network))
    else:
        async def discovery():
            async with ToolHub(config) as hub:
                return hub.discovery
        result = asyncio.run(discovery())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == 'check':
        return 0 if all(item['status'] == 'ok' for item in result['checks']) else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
