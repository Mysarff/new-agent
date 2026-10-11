import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dotenv import dotenv_values
import pymysql

from wayloom.booking import booking_action, booking_store, BookingUnavailable, MySQLBookingStore, MySQLSettings
from wayloom.config import load_config
from wayloom.database import setup_local


class DatabaseConfigurationTests(unittest.TestCase):
    def test_setup_preserves_keys_passwords_and_root_secret_on_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.env').write_text('WAYLOOM_API_KEY=existing-key\n', encoding='utf-8')
            setup_local(root)
            first = dotenv_values(root / '.env')
            secret = (root / 'var/mysql-root-password').read_bytes()
            setup_local(root)
            self.assertEqual(first, dotenv_values(root / '.env'))
            self.assertEqual(first['WAYLOOM_API_KEY'], 'existing-key')
            self.assertGreater(len(first['WAYLOOM_MYSQL_PASSWORD']), 30)
            self.assertEqual(secret, (root / 'var/mysql-root-password').read_bytes())

    def test_process_environment_wins_over_dotenv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.env').write_text('WAYLOOM_MYSQL_HOST=localhost\n', encoding='utf-8')
            original = Path(__file__).resolve().parents[1] / 'config/travel.json'
            with patch('wayloom.config.ROOT', root), patch.dict(os.environ, {
                    'WAYLOOM_CONFIG': str(original), 'WAYLOOM_MYSQL_HOST': 'mysql'}):
                load_config()
                self.assertEqual(MySQLSettings.from_env().host, 'mysql')

    def test_mysql_failure_does_not_fallback_or_expose_credentials(self):
        with patch.dict(os.environ, {'WAYLOOM_BOOKING_BACKEND': 'mysql',
                                    'WAYLOOM_MYSQL_PASSWORD': 'secret-for-test'}):
            with patch('pymysql.connect', side_effect=pymysql.OperationalError(2003, 'secret-for-test')):
                result = booking_action({}, 'query', 'train', '北京', '西安', '2031-04-23')
            self.assertEqual(result['status'], 'unavailable')
            self.assertNotIn('secret-for-test', str(result))
            self.assertNotIn('secret-for-test', repr(MySQLSettings.from_env()))

    def test_constructing_mysql_store_never_connects_or_creates_tables(self):
        with patch('pymysql.connect') as connect:
            MySQLBookingStore(MySQLSettings())
            connect.assert_not_called()

    def test_unrecognized_backend_is_not_silent_sqlite_fallback(self):
        with patch.dict(os.environ, {'WAYLOOM_BOOKING_BACKEND': 'typo'}):
            with self.assertRaises(BookingUnavailable):
                booking_store({})
