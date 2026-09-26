"""Offline UI smoke tests; every writable path belongs to a temporary directory."""
import copy
from contextlib import closing
import json
import os
from datetime import date, timedelta
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from streamlit.testing.v1 import AppTest

from SmartVoyage.knowledge import TravelKnowledge


ROOT = Path(__file__).resolve().parents[1]
SIMULATION_LABEL = '开启本地模拟票务，理解这不会购买真实车票'


class TravelUITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        docs = self.directory / 'knowledge'
        docs.mkdir()
        (docs / 'test-rule.json').write_text(json.dumps({
            'id': 'ui-rule-fixture', 'title': '测试电池携带资料',
            'text': '这是一份界面测试资料。携带测试电池时，请先检查额定容量和设备标识。',
            'source': 'https://example.test/ui-rule', 'source_type': 'user_supplied_unverified',
            'regions': ['CN'], 'tags': ['电池', '携带'],
        }, ensure_ascii=False), encoding='utf-8')
        self.config = json.loads((ROOT / 'config' / 'travel.json').read_text(encoding='utf-8'))
        self.config.update(knowledge_dir=str(docs), index_path=str(self.directory / 'index.json'),
                           booking_db=str(self.directory / 'demo.sqlite3'))
        self.network_calls = []

        def no_network(*args, **kwargs):
            self.network_calls.append(True)
            raise AssertionError('UI smoke tests must never call an external API')

        self.tool = AsyncMock(side_effect=self.offline_tool)
        patches = [
            # Patch before app.py imports these functions: .env is never loaded.
            patch('SmartVoyage.config.load_config', return_value=self.config),
            patch('SmartVoyage.config.integrations', return_value={
                'model': False, 'weather': True, 'embeddings': False,
                'places': False, 'live_search': False, 'tickets': False}),
            patch('SmartVoyage.runtime.direct_tool', self.tool),
            patch('httpx.Client.request', side_effect=no_network),
            patch('httpx.AsyncClient.request', side_effect=no_network),
            patch.dict(os.environ, {'SMARTVOYAGE_A2A': '0'}),
        ]
        for mocked in patches:
            mocked.start()
            self.addCleanup(mocked.stop)
        self.app = AppTest.from_file(ROOT / 'app.py', default_timeout=15).run()
        self.assert_clean()

    def tearDown(self):
        self.assertFalse(self.network_calls, 'A test attempted a real network request')

    async def offline_tool(self, name, arguments):
        if name == 'travel__search_knowledge':
            return TravelKnowledge(self.config).search(**arguments)
        raise AssertionError(f'Unexpected tool call: {name}')

    def assert_clean(self):
        self.assertEqual(len(self.app.exception), 0,
                         '\n'.join(str(error.message) for error in self.app.exception))

    def widget(self, kind, label):
        widgets = [element for element in getattr(self.app, kind) if element.label == label]
        self.assertEqual(len(widgets), 1, f'Expected one {kind} labeled {label!r}')
        return widgets[0]

    def click(self, label):
        self.widget('button', label).click().run()
        self.assert_clean()

    def create_route(self):
        self.widget('checkbox', SIMULATION_LABEL).check().run()
        self.assert_clean()
        self.widget('text_input', '模拟出发地').set_value('乌鲁木齐')
        self.widget('text_input', '模拟目的地').set_value('喀什')
        self.travel_day = date.today() + timedelta(days=9)
        self.widget('date_input', '模拟出行日期').set_value(self.travel_day)
        self.click('为这条路线创建并查询模拟数据')

    def test_initial_tabs_disable_unconfigured_chat_without_network(self):
        self.assertEqual([tab.label for tab in self.app.tabs],
                         ['💬 旅行对话', '🌦️ 实时天气', '📚 知识与公告', '🎫 票务演练'])
        self.assertTrue(self.widget('button', '发送给旅行助手').disabled)
        self.assertFalse(self.widget('button', '查询地点').disabled)
        self.assertFalse(self.widget('button', '检索知识与公告').disabled)
        self.assertFalse(self.widget('checkbox', SIMULATION_LABEL).value)
        self.tool.assert_not_awaited()
        self.assertTrue(Path(self.config['index_path']).exists())
        with closing(sqlite3.connect(self.config['booking_db'])) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM demo_tickets').fetchone()[0], 0)

    def test_weather_requires_explicit_candidate_then_renders_tool_result(self):
        candidates = [
            {'location_id': 101, 'label': '测试城市 · 测试地区甲', 'timezone': 'Asia/Shanghai'},
            {'location_id': 202, 'label': '测试城市 · 测试地区乙', 'timezone': 'Asia/Shanghai'},
        ]
        day = date.today()

        async def tool(name, arguments):
            if name == 'travel__geocode':
                self.assertEqual(arguments, {'query': '测试城市', 'country_code': 'CN'})
                return {'status': 'needs_input', 'candidates': copy.deepcopy(candidates)}
            if name == 'travel__forecast':
                self.assertEqual(arguments['location_id'], 202)
                self.assertEqual(arguments['start_date'], day.isoformat())
                self.assertEqual(arguments['end_date'], day.isoformat())
                return {'status': 'success', 'location': candidates[1], 'timezone': 'Asia/Shanghai',
                        'retrieved_at': '2026-09-26T08:00:00+00:00',
                        'source': 'https://example.test/mock-weather',
                        'daily': [{'date': day.isoformat(), 'temperature_2m_min': 12,
                                   'temperature_2m_max': 21, 'precipitation_probability_max': 45,
                                   'precipitation_sum': 1.5, 'wind_speed_10m_max': 8}]}
            self.fail('Unexpected tool call')

        self.tool.side_effect = tool
        self.widget('text_input', '城市或地点').set_value('测试城市')
        self.widget('text_input', '国家代码（可选）').set_value('CN')
        self.click('查询地点')
        self.assertEqual(self.tool.await_count, 1, 'Lookup must not automatically fetch first candidate')
        self.widget('selectbox', '确认你要去的地点').select(candidates[1])
        self.widget('date_input', '预报开始日期').set_value(day)
        self.widget('date_input', '预报结束日期').set_value(day)
        self.click('获取实时预报')
        self.assertEqual(self.tool.await_count, 2)
        self.assertEqual(self.app.session_state['weather_result']['location']['location_id'], 202)
        self.assertTrue(any('最低温 ℃' in frame.value.columns for frame in self.app.dataframe))

    def test_knowledge_search_uses_isolated_local_source_and_shows_citation(self):
        self.widget('text_input', '要查什么？').set_value('携带测试电池')
        self.widget('text_input', '适用地区（可选）').set_value('CN')
        self.click('检索知识与公告')
        result = self.app.session_state['knowledge_result']
        self.assertEqual(result['status'], 'candidates')
        self.assertEqual(result['chunks'][0]['document_id'], 'ui-rule-fixture')
        self.assertEqual(result['chunks'][0]['source'], 'https://example.test/ui-rule')
        self.assertTrue(any('测试电池携带资料' in expander.label for expander in self.app.expander))

    def test_dynamic_demo_route_requires_confirmation_and_replay_is_idempotent(self):
        self.create_route()
        tickets = self.app.session_state['demo_tickets']
        self.assertTrue(tickets)
        self.assertTrue(all(ticket['departure'] == '乌鲁木齐' and ticket['arrival'] == '喀什'
                            and ticket['date'] == self.travel_day.isoformat()
                            and ticket['source_kind'] == 'synthetic' for ticket in tickets))
        self.assertEqual(self.app.session_state['journey'].trip['destination'], '喀什')
        ticket_id, initial_remaining = tickets[0]['id'], tickets[0]['remaining']
        self.widget('number_input', '模拟数量').set_value(2)
        self.click('查看模拟报价')
        self.assertTrue(self.app.session_state['page_quote'])
        with closing(sqlite3.connect(self.config['booking_db'])) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM demo_orders').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT remaining FROM demo_tickets WHERE id=?',
                                        (ticket_id,)).fetchone()[0], initial_remaining)
        self.click('确认这笔模拟订单')
        self.click('确认这笔模拟订单')
        with closing(sqlite3.connect(self.config['booking_db'])) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM demo_orders').fetchone()[0], 1)
            self.assertEqual(db.execute('SELECT remaining FROM demo_tickets WHERE id=?',
                                        (ticket_id,)).fetchone()[0], initial_remaining - 2)
        self.tool.assert_not_awaited()

    def test_disabling_simulation_clears_old_candidate_and_quote(self):
        self.create_route()
        self.click('查看模拟报价')
        self.assertTrue(self.app.session_state['page_quote'])
        self.widget('checkbox', SIMULATION_LABEL).uncheck().run()
        self.assert_clean()
        self.assertFalse(self.app.session_state['journey'].demo_enabled)
        self.assertEqual(self.app.session_state['journey'].candidates, [])
        self.assertEqual(self.app.session_state['demo_tickets'], [])
        self.assertIsNone(self.app.session_state['page_quote'])
        self.assertFalse(any(button.label == '确认这笔模拟订单' for button in self.app.button))


if __name__ == '__main__':
    unittest.main()
