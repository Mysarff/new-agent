import asyncio
import fnmatch
import json
import os
import re
import sys
from contextlib import AsyncExitStack
from datetime import timedelta

from jsonschema import validate
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from .config import ROOT, env_value


class ToolHub:
    """Trusted configured MCP endpoints; tool names/schemas discovered at runtime."""
    def __init__(self, config):
        self.config = config
        self.stack = AsyncExitStack()
        self.tools, self.bindings, self.discovery = {}, {}, []

    async def __aenter__(self):
        try:
            for server in self.config['mcp_servers']:
                if server['transport'] == 'stdio':
                    command = sys.executable if server['command'] == '{python}' else server['command']
                    # Do not pass LLM credentials to every third-party child process.
                    environment = {key: value for key, value in os.environ.items()
                                   if key in ('PATH', 'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'HOME', 'USERPROFILE', 'WAYLOOM_CONFIG')}
                    for key in server.get('pass_env', []):
                        value = env_value(key) if key.startswith('WAYLOOM_') else os.getenv(key)
                        if value is not None:
                            environment[key] = value
                    environment['PYTHONUTF8'] = '1'
                    streams = await self.stack.enter_async_context(stdio_client(StdioServerParameters(
                        command=command, args=server.get('args', []), cwd=str(ROOT), env=environment)))
                    read, write = streams
                elif server['transport'] == 'streamable_http':
                    headers = {name: os.environ[key] for name, key in server.get('headers_from_env', {}).items()}
                    read, write, _ = await self.stack.enter_async_context(streamablehttp_client(
                        server['url'], headers=headers, timeout=self.config['tool_timeout']))
                else:
                    raise ValueError('Unknown MCP transport')
                session = await self.stack.enter_async_context(ClientSession(
                    read, write, read_timeout_seconds=timedelta(seconds=self.config['tool_timeout'])))
                await session.initialize()
                cursor, seen = None, set()
                while True:
                    page = await session.list_tools(cursor=cursor)
                    for tool in page.tools:
                        name = server['id'] + '__' + tool.name
                        if name in self.tools or not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', name):
                            raise ValueError('Duplicate or incompatible MCP tool name')
                        if (tool.annotations and tool.annotations.readOnlyHint is False
                                and tool.name not in server.get('allowed_write_tools', [])):
                            continue
                        self.tools[name] = {'type': 'function', 'function': {'name': name,
                                            'description': tool.description or tool.name,
                                            'parameters': tool.inputSchema}}
                        self.bindings[name] = (session, tool.name)
                    cursor = page.nextCursor
                    if not cursor:
                        break
                    if cursor in seen:
                        raise ValueError('MCP pagination loop')
                    seen.add(cursor)
                self.discovery.append({'server': server['id'], 'transport': server['transport'],
                                       'tools': [n for n in self.tools if n.startswith(server['id'] + '__')]})
            return self
        except BaseException:
            await self.stack.aclose()
            raise

    async def __aexit__(self, *args):
        return await self.stack.__aexit__(*args)

    def available(self, patterns):
        return [tool for name, tool in self.tools.items() if any(fnmatch.fnmatchcase(name, p) for p in patterns)]

    async def call(self, name, arguments):
        if name not in self.bindings:
            raise ValueError('Unknown MCP tool')
        validate(arguments, self.tools[name]['function']['parameters'])
        session, original_name = self.bindings[name]
        result = await asyncio.wait_for(session.call_tool(original_name, arguments), self.config['tool_timeout'])
        if result.isError:
            # Never promote a failed tool call into evidence.
            return {'status': 'tool_error', 'message': 'MCP 工具返回失败，请检查服务或查询参数。'}
        if result.structuredContent:
            return result.structuredContent
        body = '\n'.join(c.text for c in result.content if c.type == 'text')
        try:
            return json.loads(body)
        except ValueError:
            return {'status': 'success', 'text': body[:16000]}
