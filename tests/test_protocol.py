import tempfile
import unittest
import json
from pathlib import Path

import httpx

from opsatlas.config import load_config
from opsatlas.engine import Engine
from opsatlas.mcp_client import ToolHub
from opsatlas.rag import build_index
from opsatlas.servers import package_info, repository_info


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_to_real_mcp_to_cited_answer(self):
        class ProtocolModelFixture:
            async def complete(self, messages, tools):
                names = [tool['function']['name'] for tool in tools]
                if messages[-1]['role'] == 'tool':
                    payload = json.loads(messages[-1]['content'])
                    text = payload['answer'] if 'answer' in payload else '演练资料 [' + payload['chunks'][0]['id'] + ']'
                    return {'role': 'assistant', 'content': text}
                name = 'delegate__runbook' if 'delegate__runbook' in names else 'knowledge__search_knowledge'
                arguments = {'task': '连接池耗尽'} if name.startswith('delegate') else {'query': '连接池耗尽'}
                return {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'test1', 'type': 'function',
                        'function': {'name': name, 'arguments': json.dumps(arguments)}}]}
        config = load_config()
        if not Path(config['index_path']).exists():
            build_index(config)
        async with ToolHub(config) as hub:
            result = await Engine(ProtocolModelFixture(), hub, config).run('连接池耗尽')
        self.assertEqual(result['citations'][0]['source'], '03-postgres.md')
        self.assertEqual(result['tool_calls'], 2)
        self.assertTrue(all(event.get('status', 'success') == 'success' or event.get('status') == 'candidates'
                            for event in result['trace']))

    async def test_real_mcp_discovery_and_retrieval(self):
        config = load_config()
        if not Path(config['index_path']).exists():
            build_index(config)
        async with ToolHub(config) as hub:
            self.assertEqual(len(hub.tools), 3)
            self.assertEqual(len(hub.available(['knowledge__*'])), 1)
            result = await hub.call('knowledge__search_knowledge', {'query': '数据库连接池耗尽', 'top_k': 3})
            self.assertTrue(result['chunks'])
            self.assertEqual(result['chunks'][0]['source'], '03-postgres.md')
            with self.assertRaises(Exception):
                await hub.call('knowledge__search_knowledge', {'query': 123})

    async def test_public_provider_accepts_unlisted_package_and_repo(self):
        seen = []
        def handler(request):
            seen.append(str(request.url))
            return httpx.Response(200, json={'info': {'name': 'unlisted-project-734', 'version': '1.2.3'},
                                           'full_name': 'someone/new-project-735'})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            package = await package_info('unlisted-project-734', client)
            repo = await repository_info('someone', 'new-project-735', client)
        self.assertEqual(package['evidence'][0]['data']['version'], '1.2.3')
        self.assertEqual(repo['evidence'][0]['data']['full_name'], 'someone/new-project-735')
        self.assertEqual(len(seen), 2)

    async def test_public_failure_and_path_rejection(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(404))) as client:
            self.assertEqual((await package_info('missing-package', client))['status'], 'not_found')
        with self.assertRaises(ValueError):
            await package_info('../../private')
        with self.assertRaises(ValueError):
            await repository_info('someone', '..')
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(429))) as client:
            with self.assertRaises(httpx.HTTPStatusError):
                await package_info('httpx', client)
