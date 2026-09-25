import os
import unittest
from unittest.mock import patch

import httpx

from opsatlas.model import ChatModel


class ModelTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_config_never_uses_fake_model(self):
        with patch.dict(os.environ, {'AGENT_MODEL': '', 'AGENT_API_KEY': ''}):
            with self.assertRaises(ValueError):
                ChatModel()

    async def test_wire_tool_calls_and_provider_errors(self):
        import json
        calls = []
        def handler(request):
            calls.append(json.loads(request.content))
            if len(calls) > 1:
                return httpx.Response(401)
            return httpx.Response(200, json={'choices': [{'message': {'role': 'assistant', 'content': None,
                        'tool_calls': [{'id': 'wire1', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{}'}}]}}]})
        # Inject transport only; this exercises the actual HTTP adapter and tool message schema.
        with patch.dict(os.environ, {'AGENT_MODEL': 'test-fixture', 'AGENT_API_KEY': 'test-fixture',
                                     'AGENT_BASE_URL': 'https://provider.example/v1'}):
            model = ChatModel()
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch('opsatlas.model.httpx.AsyncClient', return_value=client):
            result = await model.complete([{'role': 'user', 'content': 'hello'}],
                                           [{'type': 'function', 'function': {'name': 'lookup', 'parameters': {'type': 'object'}}}])
        self.assertEqual(result['tool_calls'][0]['id'], 'wire1')
        self.assertEqual(calls[0]['tool_choice'], 'auto')
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch('opsatlas.model.httpx.AsyncClient', return_value=client):
            with self.assertRaises(httpx.HTTPStatusError):
                await model.complete([{'role': 'user', 'content': 'hello'}], [])
