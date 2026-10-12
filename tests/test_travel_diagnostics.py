import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from wayloom.diagnostics import ModelResponseError, check_connections, model_failure


class DiagnosticsTests(unittest.IsolatedAsyncioTestCase):
    async def test_engine_retains_safe_cause_in_warning_and_trace(self):
        from wayloom.engine import Engine
        from tests.test_travel_engine import HubFixture, configuration
        model = AsyncMock()
        request = httpx.Request('POST', 'https://model.test/v1')
        model.complete.side_effect = httpx.HTTPStatusError('must-not-leak', request=request,
                                                        response=httpx.Response(401, request=request))
        result = await Engine(model, HubFixture(), configuration()).run('查询天气')
        self.assertIn('401', result['warnings'][0])
        event = next(row for row in result['trace'] if row['event'] == 'model_error')
        self.assertEqual(event['diagnostic']['code'], 'authentication')
        self.assertNotIn('must-not-leak', json.dumps(result))

    def test_failures_are_actionable_without_copying_secrets_or_provider_text(self):
        secret = 'must-not-expose-test-key'
        request = httpx.Request('POST', 'https://model.test/v1?key=' + secret,
                                headers={'Authorization': 'Bearer ' + secret})
        for status, code in ((401, 'authentication'), (403, 'permission'), (429, 'rate_limit'),
                             (400, 'configuration'), (404, 'configuration'), (503, 'provider_unavailable')):
            error = httpx.HTTPStatusError(secret, request=request, response=httpx.Response(status,
                json={'error': {'message': secret}}, request=request))
            result = model_failure(error)
            self.assertEqual(result['code'], code)
            self.assertEqual(result['http_status'], status)
            self.assertNotIn(secret, json.dumps(result))
        for error, code in ((httpx.ConnectError(secret), 'connection'),
                            (httpx.ReadTimeout(secret), 'timeout'),
                            (TimeoutError(secret), 'timeout'),
                            (ModelResponseError(secret), 'response_format')):
            result = model_failure(error)
            self.assertEqual(result['code'], code)
            self.assertNotIn(secret, json.dumps(result))

    async def test_model_plain_text_does_not_pass_tool_protocol_check(self):
        model = AsyncMock()
        model.complete.return_value = {'role': 'assistant', 'content': '连接成功'}
        report = await check_connections({'agents': []}, model=model)
        self.assertEqual(report['checks'][0]['status'], 'error')
        self.assertIn('未按要求', report['checks'][0]['message'])

    async def test_actual_tool_echo_passes_but_wrong_marker_fails(self):
        for correct in (True, False):
            async def complete(messages, tools, tool_choice):
                marker = messages[-1]['content'].removeprefix('marker=')
                return {'tool_calls': [{'function': {'name': 'wayloom_connection_check',
                    'arguments': json.dumps({'marker': marker if correct else 'invented'})}}]}
            model = AsyncMock()
            model.complete.side_effect = complete
            report = await check_connections({'agents': []}, model=model)
            self.assertEqual(report['checks'][0]['status'], 'ok' if correct else 'error')

    async def test_reachable_model_does_not_hide_stopped_or_wrong_agent(self):
        agents = [{'id': 'weather', 'name': '天气 Agent', 'port': 8611},
                  {'id': 'knowledge', 'name': '知识 Agent', 'port': 8613}]
        def handler(request):
            if request.url.port == 8611:
                raise httpx.ConnectError('refused', request=request)
            return httpx.Response(200, json={'status': 'ok', 'agent': 'wrong'})
        client = httpx.AsyncClient
        model = AsyncMock()
        model.complete.side_effect = httpx.ReadTimeout('secret')
        with patch('wayloom.diagnostics.httpx.AsyncClient', side_effect=lambda **kw:
                   client(transport=httpx.MockTransport(handler), **kw)):
            report = await check_connections({'agents': agents}, network=True, model=model)
        self.assertEqual(len(report['checks']), 3)
        self.assertTrue(all(row['status'] == 'error' for row in report['checks']))
        self.assertTrue(all('wayloom.stack' in row['message'] for row in report['checks'][1:]))

