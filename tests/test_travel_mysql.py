"""Opt-in real MySQL checks. Never run against a user's business database."""
from concurrent.futures import ThreadPoolExecutor
import asyncio
import os
from pathlib import Path
import re
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import test_travel_tickets as sqlite_tests
from wayloom.booking import MySQLBookingStore, MySQLSettings, _Connection, TABLES
from wayloom.database import migrate_sqlite
from wayloom.tickets import DemoBookingStore
from wayloom.config import load_config


@unittest.skipUnless(os.getenv('WAYLOOM_TEST_MYSQL') == '1', 'requires isolated MySQL test database')
class MySQLBookingTests(sqlite_tests.DemoBookingTests):
    @classmethod
    def setUpClass(cls):
        cls.settings = MySQLSettings.from_env()
        if not re.fullmatch(r'wayloom_test(?:_[a-f0-9]+)?', cls.settings.database):
            raise RuntimeError('Refusing to modify a non-test MySQL database')
        MySQLBookingStore(cls.settings).initialize()

    def setUp(self):
        self.store = MySQLBookingStore(self.settings)
        with self.store.connection(write=True) as db:
            for table in reversed(TABLES):
                db.execute(f'DELETE FROM {table}')

    def tearDown(self):
        pass

    def test_configured_quantity_limit_and_invalid_quantities(self):
        self.store = MySQLBookingStore(self.settings, max_quantity=2)
        ticket = self.seed()
        for quantity in (True, 0, -1, 3, 1.2, '1'):
            with self.assertRaises(ValueError):
                self.store.prepare(ticket['id'], quantity)
        self.assertEqual(self.store.prepare(ticket['id'], 2)['status'], 'success')

    def race(self, calls):
        barrier = threading.Barrier(len(calls))
        def call(args):
            barrier.wait(timeout=10)
            try:
                return self.store.confirm(*args)
            except ValueError:
                return {'status': 'request_conflict'}
        with ThreadPoolExecutor(max_workers=len(calls)) as executor:
            return list(executor.map(call, calls))

    def test_twelve_duplicate_clicks_make_one_order(self):
        ticket, quote = self.prepare(quantity=2)
        results = self.race([(quote['quote_id'], 'same-click')] * 12)
        self.assertTrue(all(result['status'] == 'success' for result in results))
        self.assertEqual(len({result['order_id'] for result in results}), 1)
        self.assertEqual(sum(not result['replayed'] for result in results), 1)
        self.assertEqual(self.store.query(**sqlite_tests.QUERY)['tickets'][0]['remaining'], ticket['remaining'] - 2)

    def test_same_quote_different_request_keys_make_one_order(self):
        ticket, quote = self.prepare(quantity=2)
        results = self.race([(quote['quote_id'], f'click-{i}') for i in range(12)])
        self.assertEqual(len({result['order_id'] for result in results}), 1)
        counts = self.store.health()['counts']
        self.assertEqual(counts['demo_orders'], 1)
        self.assertEqual(counts['demo_request_keys'], 12)

    def test_same_request_different_quotes_rolls_back_losing_stock(self):
        ticket = self.seed()
        quotes = [self.store.prepare(ticket['id'], 1)['quote'] for _ in range(8)]
        results = self.race([(quote['quote_id'], 'one-request') for quote in quotes])
        self.assertEqual([result['status'] for result in results].count('success'), 1)
        self.assertEqual([result['status'] for result in results].count('request_conflict'), 7)
        self.assertEqual(self.store.query(**sqlite_tests.QUERY)['tickets'][0]['remaining'], ticket['remaining'] - 1)

    def test_distinct_quotes_twelve_buyers_do_not_oversell(self):
        ticket = self.seed()
        with self.store.connection(write=True) as db:
            db.execute('UPDATE demo_tickets SET remaining=5 WHERE id=?', (ticket['id'],))
        quotes = [self.store.prepare(ticket['id'], 1)['quote'] for _ in range(12)]
        results = self.race([(quote['quote_id'], f'buyer-{i}') for i, quote in enumerate(quotes)])
        self.assertEqual([result['status'] for result in results].count('success'), 5)
        self.assertEqual([result['status'] for result in results].count('no_data'), 7)
        self.assertEqual(self.store.query(**sqlite_tests.QUERY)['tickets'][0]['remaining'], 0)
        self.assertEqual(self.store.health()['counts']['demo_orders'], 5)

    def test_failure_after_stock_update_rolls_back_everything(self):
        ticket, quote = self.prepare()
        original = _Connection.execute
        def fail_after_update(db, statement, values=()):
            if statement.startswith('INSERT INTO demo_orders'):
                raise RuntimeError('injected interruption')
            return original(db, statement, values)
        with patch.object(_Connection, 'execute', fail_after_update):
            with self.assertRaises(RuntimeError):
                self.store.confirm(quote['quote_id'], 'interrupted')
        self.assertEqual(self.store.health()['counts']['demo_orders'], 0)
        self.assertEqual(self.store.query(**sqlite_tests.QUERY)['tickets'][0]['remaining'], ticket['remaining'])
        self.assertEqual(self.store.confirm(quote['quote_id'], 'interrupted')['status'], 'success')

    def test_quote_expiring_while_waiting_for_stock_is_rejected(self):
        self.store = MySQLBookingStore(self.settings, quote_ttl_seconds=2)
        ticket, quote = self.prepare()
        waiting = threading.Event()
        original = _Connection.execute
        def observe_lock(db, statement, values=()):
            if statement == 'SELECT * FROM demo_tickets WHERE id=? FOR UPDATE':
                waiting.set()
            return original(db, statement, values)
        with ThreadPoolExecutor(max_workers=1) as executor:
            with self.store.connection(write=True) as db:
                db.execute('SELECT id FROM demo_tickets WHERE id=? FOR UPDATE', (ticket['id'],))
                with patch.object(_Connection, 'execute', observe_lock):
                    pending = executor.submit(self.store.confirm, quote['quote_id'], 'wait-expiry')
                    self.assertTrue(waiting.wait(timeout=1.5))
                    time.sleep(max(0, quote['expires_at'] - time.time()) + 0.05)
            result = pending.result(timeout=5)
        self.assertEqual(result['status'], 'expired')
        self.assertEqual(self.store.health()['counts']['demo_orders'], 0)
        self.assertEqual(self.store.query(**sqlite_tests.QUERY)['tickets'][0]['remaining'], ticket['remaining'])

    def test_request_keys_are_case_sensitive_and_initialization_preserves_orders(self):
        ticket, quote = self.prepare()
        first = self.store.confirm(quote['quote_id'], 'Click')
        second_quote = self.store.prepare(ticket['id'], 1)['quote']
        second = self.store.confirm(second_quote['quote_id'], 'click')
        self.assertNotEqual(first['order_id'], second['order_id'])
        self.store.initialize()
        self.assertEqual(self.store.health()['counts']['demo_orders'], 2)

    def test_read_only_migration_preserves_replay_and_refuses_nonempty_target(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'legacy.sqlite3'
            legacy = DemoBookingStore(source)
            legacy.seed_demo('北京', '西安', '2031-04-23')
            ticket = legacy.query('train', '北京', '西安', '2031-04-23')['tickets'][0]
            quote = legacy.prepare(ticket['id'], 2)['quote']
            order = legacy.confirm(quote['quote_id'], 'legacy-click')
            before = source.read_bytes()
            result = migrate_sqlite(source, self.store)
            self.assertEqual(result['counts']['demo_orders'], 1)
            replay = self.store.confirm(quote['quote_id'], 'legacy-click')
            self.assertEqual(replay['order_id'], order['order_id'])
            self.assertTrue(replay['replayed'])
            self.assertEqual(self.store.query('train', '北京', '西安', '2031-04-23')['tickets'][0]['remaining'], 10)
            self.assertEqual(source.read_bytes(), before)
            with self.assertRaises(ValueError):
                migrate_sqlite(source, self.store)

    def test_import_failure_rolls_back_all_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'legacy.sqlite3'
            legacy = DemoBookingStore(source)
            legacy.seed_demo('北京', '西安', '2031-04-23')
            ticket = legacy.query('train', '北京', '西安', '2031-04-23')['tickets'][0]
            legacy.prepare(ticket['id'], 1)
            original = _Connection.execute
            def fail_import(db, statement, values=()):
                if statement.startswith('INSERT INTO demo_quotes'):
                    raise RuntimeError('injected interruption')
                return original(db, statement, values)
            with patch.object(_Connection, 'execute', fail_import):
                with self.assertRaises(RuntimeError):
                    migrate_sqlite(source, self.store)
            self.assertEqual(sum(self.store.health()['counts'].values()), 0)

    def test_real_mcp_subprocess_uses_mysql_and_exposes_no_confirmation(self):
        from wayloom.mcp_client import ToolHub
        ticket = self.seed()
        config = load_config()
        async def run():
            async with ToolHub(config) as hub:
                self.assertFalse(any('confirm' in name for name in hub.tools))
                result = await hub.call('travel__query_demo_tickets', sqlite_tests.QUERY)
                self.assertEqual(result['tickets'][0]['id'], ticket['id'])
                prepared = await hub.call('travel__prepare_demo_booking', {'ticket_id': ticket['id'], 'quantity': 2})
                self.assertEqual(prepared['status'], 'success')
        asyncio.run(run())
        self.assertEqual(self.store.health()['counts']['demo_quotes'], 1)
        self.assertEqual(self.store.health()['counts']['demo_orders'], 0)

    def test_streamlit_confirmation_replays_against_real_mysql(self):
        from streamlit.testing.v1 import AppTest
        config = load_config()
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            config['index_path'] = str(Path(directory) / 'index.json')
            with patch('wayloom.config.load_config', return_value=config), patch('wayloom.config.integrations',
                    return_value={'model': False, 'weather': True, 'embeddings': False,
                                  'places': False, 'live_search': False, 'tickets': False}):
                app = AppTest.from_file(root / 'app.py', default_timeout=20).run()
                def widget(kind, label):
                    return next(item for item in getattr(app, kind) if item.label == label)
                widget('checkbox', '开启本地模拟票务，理解这不会购买真实车票').check().run()
                widget('button', '为这条路线创建并查询模拟数据').click().run()
                widget('button', '查看模拟报价').click().run()
                self.assertEqual(self.store.health()['counts']['demo_orders'], 0)
                widget('button', '确认这笔模拟订单').click().run()
                widget('button', '确认这笔模拟订单').click().run()
                self.assertFalse(app.exception)
                self.assertEqual(self.store.health()['counts']['demo_orders'], 1)
