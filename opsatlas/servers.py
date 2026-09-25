"""Actual MCP services: local domain retrieval and live public technical metadata."""
import argparse
import hashlib
import re
from datetime import datetime, timezone

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from .config import load_config
from .model import Embeddings
from .rag import Retriever

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)


def knowledge_server():
    config = load_config()
    server = FastMCP('OpsAtlasKnowledge')

    @server.tool(annotations=READ_ONLY)
    def search_knowledge(query: str, top_k: int = 5) -> dict:
        """检索内部运维知识库，返回原文知识单元、来源、模拟标记与引用ID。无证据时不要猜测。"""
        embedding = Embeddings()
        retriever = Retriever(config, embedding if embedding.model else None)
        return retriever.search(query, top_k)

    return server


async def fetch_json(url, client=None):
    if client is None:
        async with httpx.AsyncClient(timeout=15, follow_redirects=False,
                                     headers={'User-Agent': 'OpsAtlas/0.1', 'Accept': 'application/json'}) as session:
            return await fetch_json(url, session)
    response = await client.get(url)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def evidence(url, data):
    timestamp = datetime.now(timezone.utc).isoformat()
    identifier = 'W' + hashlib.sha256((url + timestamp).encode()).hexdigest()[:16]
    return {'status': 'success', 'evidence': [{'id': identifier, 'source': url,
            'retrieved_at': timestamp, 'metadata': {'provenance': 'live_public_api'}, 'data': data}]}


async def package_info(package, client=None):
    # Syntax validation only: there is no list of supported package names.
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,199}', package):
        raise ValueError('Invalid Python package name')
    url = f'https://pypi.org/pypi/{package}/json'
    result = await fetch_json(url, client)
    if result is None:
        return {'status': 'not_found', 'source': url}
    info = result['info']
    return evidence(url, {key: info.get(key) for key in ('name', 'version', 'summary', 'requires_python', 'project_urls')})


async def repository_info(owner, repository, client=None):
    for value in (owner, repository):
        if value in ('.', '..') or not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', value):
            raise ValueError('Invalid GitHub owner/repository')
    url = f'https://api.github.com/repos/{owner}/{repository}'
    result = await fetch_json(url, client)
    if result is None:
        return {'status': 'not_found', 'source': url}
    return evidence(url, {key: result.get(key) for key in ('full_name', 'description', 'html_url', 'language',
                    'stargazers_count', 'open_issues_count', 'archived', 'pushed_at', 'default_branch')})


def public_server():
    server = FastMCP('OpsAtlasPublicTech')

    @server.tool(annotations=READ_ONLY)
    async def get_python_package(package: str) -> dict:
        """从 PyPI 实时查询任意 Python 包的当前版本、Python要求和项目链接；不是漏洞安全审计。"""
        return await package_info(package)

    @server.tool(annotations=READ_ONLY)
    async def get_github_repository(owner: str, repository: str) -> dict:
        """从 GitHub 实时读取任意公开仓库元数据，需提供 owner 和 repository。可能触发匿名API限流。"""
        return await repository_info(owner, repository)

    return server


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('service', choices=['knowledge', 'public'])
    args = parser.parse_args()
    (knowledge_server() if args.service == 'knowledge' else public_server()).run(transport='stdio')
