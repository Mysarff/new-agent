"""Provider contracts with MockTransport; all keys are test-only strings."""
import json
import os
import unittest
from unittest.mock import patch

import httpx

from wayloom.travel_web import search_places, search_travel_web


AMAP_KEY = 'test-only-amap-secret-do-not-echo'
SEARCH_KEY = 'test-only-tavily-secret-do-not-echo'


class TravelWebTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'AMAP_API_KEY': AMAP_KEY, 'TAVILY_API_KEY': SEARCH_KEY})
        env.start()
        self.addCleanup(env.stop)

    def assert_no_key(self, result):
        rendered = json.dumps(result, ensure_ascii=False)
        self.assertNotIn(AMAP_KEY, rendered)
        self.assertNotIn(SEARCH_KEY, rendered)

    async def test_unconfigured_providers_do_not_make_requests_or_fake_data(self):
        def handler(request):
            self.fail('Unconfigured provider should not make a request')
        with patch.dict(os.environ, {'AMAP_API_KEY': '', 'TAVILY_API_KEY': ''}):
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                places = await search_places('博物馆', '乌鲁木齐', client)
                search = await search_travel_web('景区公告', client=client)
        for result in (places, search):
            self.assertEqual(result['status'], 'unavailable')
            self.assertFalse(result.get('evidence'))
            self.assert_no_key(result)

    async def test_arbitrary_city_query_passed_to_amap_and_safe_source_returned(self):
        def handler(request):
            self.assertEqual(request.url.host, 'restapi.amap.com')
            self.assertEqual(request.url.params['city'], '喀什')
            self.assertEqual(request.url.params['keywords'], '室内博物馆')
            self.assertEqual(request.url.params['key'], AMAP_KEY)
            self.assertEqual(request.url.params['citylimit'], 'true')
            return httpx.Response(200, json={'status': '1', 'pois': [{
                'id': 'test-poi-1', 'name': '测试博物馆', 'cityname': '喀什地区',
                'address': '测试地址', 'location': '75.98,39.47', 'extra_secret_field': AMAP_KEY,
            }]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await search_places('室内博物馆', '喀什', client)
            self.assertFalse(client.is_closed)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['places'][0]['id'], 'test-poi-1')
        self.assertNotIn('extra_secret_field', result['places'][0])
        self.assertEqual(result['evidence'][0]['metadata']['provenance'], 'live_api')
        self.assert_no_key(result)

    async def test_amap_denied_and_empty_results_have_explicit_status(self):
        for payload, expected in [({'status': '0', 'infocode': '10001', 'info': AMAP_KEY}, 'error'),
                                  ({'status': '1', 'pois': []}, 'no_data')]:
            with self.subTest(expected=expected):
                async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))) as client:
                    result = await search_places('博物馆', '成都', client)
                self.assertEqual(result['status'], expected)
                self.assert_no_key(result)
                if expected == 'error':
                    self.assertFalse(result.get('evidence'))

    async def test_search_auth_is_header_only_and_official_domains_passed(self):
        def handler(request):
            self.assertEqual(request.url.host, 'api.tavily.com')
            self.assertEqual(request.headers['Authorization'], 'Bearer ' + SEARCH_KEY)
            body = json.loads(request.content)
            self.assertEqual(body['include_domains'], ['dpm.org.cn'])
            self.assertEqual(body['query'], '故宫临时闭馆公告')
            self.assertFalse(body['include_answer'])
            self.assertNotIn(SEARCH_KEY, str(request.url))
            self.assertNotIn(SEARCH_KEY, request.content.decode())
            return httpx.Response(200, json={'results': [
                {'url': 'https://www.dpm.org.cn/test-notice.html', 'title': '测试公告',
                 'content': '测试摘录' * 2000, 'published_date': '2026-09-26'},
                {'url': 'javascript:alert(1)', 'title': 'invalid', 'content': 'unsafe'},
            ]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await search_travel_web('故宫临时闭馆公告', ['dpm.org.cn'], client)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(len(result['evidence']), 1)
        item = result['evidence'][0]
        self.assertEqual(item['metadata']['provenance'], 'web_search_excerpt')
        self.assertLessEqual(len(item['data']['snippet']), 4000)
        self.assertEqual(item['data']['published_at'], '2026-09-26')
        self.assert_no_key(result)

    async def test_empty_search_and_invalid_domains_do_not_invent_evidence(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={'results': []})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await search_travel_web('无匹配公告', client=client)
            self.assertEqual(result['status'], 'no_data')
            self.assertEqual(result['evidence'], [])
            for domains in [['https://dpm.org.cn'], ['dpm.org.cn/path'], ['user:password@dpm.org.cn']]:
                with self.subTest(domains=domains):
                    with self.assertRaises(ValueError):
                        await search_travel_web('公告', domains, client)
        self.assertEqual(len(requests), 1)

    async def test_http_errors_and_timeout_return_safe_failure_not_key_bearing_exception(self):
        for provider in ('places', 'search'):
            for status in (401, 429, 503, 'timeout'):
                with self.subTest(provider=provider, status=status):
                    def handler(request):
                        if status == 'timeout':
                            raise httpx.ReadTimeout('fixture error: ' + AMAP_KEY, request=request)
                        return httpx.Response(status, text=AMAP_KEY + SEARCH_KEY)
                    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                        result = await (search_places('博物馆', '西安', client) if provider == 'places'
                                        else search_travel_web('官方公告', client=client))
                    self.assertEqual(result['status'], 'error')
                    self.assertFalse(result.get('evidence'))
                    self.assert_no_key(result)

    async def test_malformed_provider_payload_is_rejected_without_evidence(self):
        for provider, payload in [('places', []), ('places', {'status': '1', 'pois': ['wrong']}),
                                  ('search', []), ('search', {'results': {'unexpected': 'object'}}),
                                  ('search', {'results': [{'url': 'https://example.test/', 'content': {'wrong': 1}}]})]:
            with self.subTest(provider=provider, payload=payload):
                async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))) as client:
                    result = await (search_places('博物馆', '西安', client) if provider == 'places'
                                    else search_travel_web('官方公告', client=client))
                self.assertEqual(result['status'], 'error')
                self.assertFalse(result.get('evidence'))
                self.assert_no_key(result)

    async def test_search_urls_require_hostname_and_cannot_embed_credentials(self):
        urls = ['https:broken', 'https://user:password@example.test/notice',
                'https://www.dpm.org.cn/safe-test-notice.html']
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
            'results': [{'url': url, 'title': '测试', 'content': '测试'} for url in urls]}))) as client:
            result = await search_travel_web('官方公告', client=client)
        self.assertEqual(result['status'], 'success')
        self.assertEqual([entry['source'] for entry in result['evidence']], [urls[-1]])


if __name__ == '__main__':
    unittest.main()
