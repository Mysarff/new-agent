import argparse
import asyncio
import json

from .config import load_config
from .engine import Engine
from .mcp_client import ToolHub
from .model import ChatModel, Embeddings
from .rag import Retriever, build_index


async def chat(query, config, history=None):
    model = ChatModel()
    async with ToolHub(config) as hub:
        result = await Engine(model, hub, config).run(query, history)
        result['discovery'] = hub.discovery
        return result


def main():
    parser = argparse.ArgumentParser(description='OpsAtlas — agents, MCP and domain RAG')
    sub = parser.add_subparsers(dest='command', required=True)
    ingest = sub.add_parser('ingest')
    ingest.add_argument('--dense', action='store_true', help='Use configured embedding API; incurs provider cost')
    search = sub.add_parser('search')
    search.add_argument('query')
    ask = sub.add_parser('chat')
    ask.add_argument('query')
    sub.add_parser('discover')
    args = parser.parse_args()
    config = load_config()
    if args.command == 'ingest':
        result = build_index(config, Embeddings() if args.dense else None)
    elif args.command == 'search':
        embedding = Embeddings()
        result = Retriever(config, embedding if embedding.model else None).search(args.query)
    elif args.command == 'chat':
        result = asyncio.run(chat(args.query, config))
    else:
        async def discover():
            async with ToolHub(config) as hub:
                return hub.discovery
        result = asyncio.run(discover())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
