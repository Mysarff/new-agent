import argparse
import asyncio
import json

from .config import load_config
from .knowledge import ingest
from .mcp_client import ToolHub
from .model import Embeddings
from .runtime import chat, direct_tool


def main():
    parser = argparse.ArgumentParser(description='SmartVoyage travel agent')
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
    args = parser.parse_args()
    config = load_config()
    if args.command == 'ingest':
        result = ingest(config, Embeddings() if args.dense else None)
    elif args.command == 'chat':
        result = asyncio.run(chat(args.query, network=args.network))
    elif args.command == 'search':
        result = asyncio.run(direct_tool('travel__search_knowledge',
                                         {'query': args.query, 'travel_date': args.date, 'region': args.region}))
    else:
        async def discovery():
            async with ToolHub(config) as hub:
                return hub.discovery
        result = asyncio.run(discovery())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
