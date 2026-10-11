"""MySQL business storage. No credentials or database errors reach model output.

SQLite remains an explicit offline/test backend and a read-only migration source.
The runtime never silently falls back to it when MySQL is unavailable.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
import random
import time

import pymysql

from .config import env_value
from .tickets import DemoBookingStore


class BookingUnavailable(RuntimeError):
    pass


UNAVAILABLE = '票务数据库暂不可用，请先启动并初始化项目数据库；天气和知识查询仍可使用。'


@dataclass(frozen=True)
class MySQLSettings:
    host: str = '127.0.0.1'
    port: int = 3309
    database: str = 'wayloom'
    user: str = 'wayloom'
    password: str = field(default='', repr=False)

    @classmethod
    def from_env(cls):
        try:
            port = int(env_value('WAYLOOM_MYSQL_PORT', '3309'))
            if not 1 <= port <= 65535:
                raise ValueError
        except ValueError:
            raise BookingUnavailable('WAYLOOM_MYSQL_PORT 须为 1 至 65535 的整数。') from None
        return cls(host=env_value('WAYLOOM_MYSQL_HOST', '127.0.0.1'), port=port,
                   database=env_value('WAYLOOM_MYSQL_DATABASE', 'wayloom'),
                   user=env_value('WAYLOOM_MYSQL_USER', 'wayloom'),
                   password=env_value('WAYLOOM_MYSQL_PASSWORD', ''))


TABLES = {
    'demo_tickets': '''id VARCHAR(160) PRIMARY KEY, kind VARCHAR(40) NOT NULL,
        departure VARCHAR(120) NOT NULL, arrival VARCHAR(120) NOT NULL,
        date VARCHAR(10) NOT NULL, service VARCHAR(160) NOT NULL,
        seat VARCHAR(160) NOT NULL, price_cents BIGINT NOT NULL CHECK(price_cents>=0),
        currency VARCHAR(3) NOT NULL, remaining BIGINT NOT NULL CHECK(remaining>=0),
        INDEX route_lookup(kind,departure,arrival,date,price_cents,id)''',
    'demo_quotes': '''id VARCHAR(160) PRIMARY KEY, expires_at DOUBLE NOT NULL,
        payload LONGTEXT NOT NULL, used_order_id VARCHAR(160), INDEX quote_expiry(expires_at)''',
    'demo_orders': '''id VARCHAR(160) PRIMARY KEY, request_id VARCHAR(128) NOT NULL UNIQUE,
        quote_id VARCHAR(160) NOT NULL UNIQUE, payload LONGTEXT NOT NULL''',
    'demo_request_keys': '''request_id VARCHAR(128) PRIMARY KEY,
        quote_id VARCHAR(160) NOT NULL, order_id VARCHAR(160) NOT NULL''',
}


class _Row(dict):
    def __getitem__(self, key):
        return tuple(self.values())[key] if isinstance(key, int) else super().__getitem__(key)


class _Result:
    def __init__(self, rows, rowcount):
        self.rows = [_Row(row) for row in rows]
        self.rowcount = rowcount

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class _Connection:
    """Adapt only our fixed, parameterized qmark statements; never interpolates values."""
    def __init__(self, connection):
        self.raw = connection

    def execute(self, statement, values=()):
        with self.raw.cursor() as cursor:
            cursor.execute(statement.replace('?', '%s'), values)
            return _Result(cursor.fetchall() if cursor.description else [], cursor.rowcount)


class MySQLBookingStore(DemoBookingStore):
    lock_clause = ' FOR UPDATE'
    # A duplicate is a no-op, not a stock reset. Unlike INSERT IGNORE this does
    # not hide invalid data, truncation or violated CHECK constraints.
    seed_sql = ('INSERT INTO demo_tickets VALUES (?,?,?,?,?,?,?,?,?,?) '
                'ON DUPLICATE KEY UPDATE id=demo_tickets.id')

    def __init__(self, settings=None, max_quantity=5, quote_ttl_seconds=300):
        self.settings = settings or MySQLSettings.from_env()
        self._configure_limits(max_quantity, quote_ttl_seconds)
        # No connection/DDL on Streamlit reruns or on MCP tool discovery.

    @contextmanager
    def connection(self, write=False):
        db = None
        try:
            if not self.settings.password:
                raise BookingUnavailable(UNAVAILABLE)
            db = pymysql.connect(host=self.settings.host, port=self.settings.port,
                                 user=self.settings.user, password=self.settings.password,
                                 database=self.settings.database, charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor, autocommit=False,
                                 connect_timeout=3, read_timeout=10, write_timeout=10)
            adapter = _Connection(db)
            # Retry reads must see a concurrently committed idempotency key.
            adapter.execute('SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED')
            adapter.execute('SET SESSION innodb_lock_wait_timeout=5')
            yield adapter
            if write:
                db.commit()
        except Exception as exc:
            if db:
                try:
                    db.rollback()
                except pymysql.MySQLError:
                    pass
            # These failures are retried only by confirm, as whole transactions.
            if isinstance(exc, pymysql.MySQLError):
                if exc.args and exc.args[0] in (1062, 1205, 1213):
                    raise
                raise BookingUnavailable(UNAVAILABLE) from None
            raise
        finally:
            if db:
                db.close()

    def initialize(self):
        """Explicit, idempotent DDL; never initializes tickets or orders."""
        with self.connection(write=True) as db:
            for table, columns in TABLES.items():
                db.execute(f'CREATE TABLE IF NOT EXISTS {table} ({columns}) '
                           'ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin')
        return self.health()

    def health(self):
        with self.connection() as db:
            counts = {table: db.execute(f'SELECT COUNT(*) AS n FROM {table}').fetchone()['n']
                      for table in TABLES}
        return {'status': 'success', 'backend': 'mysql', 'counts': counts}

    def confirm(self, quote_id, request_id):
        # Quote and stock locks serialize competing updates to the same records.
        # UNIQUE request_id/quote_id also handles different quotes racing on one
        # request key. The losing transaction rolls back stock before retrying.
        for attempt in range(3):
            try:
                return super().confirm(quote_id, request_id)
            except (pymysql.IntegrityError, pymysql.OperationalError) as exc:
                if not exc.args or exc.args[0] not in (1062, 1205, 1213):
                    raise BookingUnavailable(UNAVAILABLE) from None
                if attempt == 2:
                    raise BookingUnavailable('模拟确认遇到并发冲突，请使用同一报价重试；不会重复扣库存。') from None
                time.sleep(random.uniform(0.01, 0.04) * (attempt + 1))


def booking_store(config):
    backend = env_value('WAYLOOM_BOOKING_BACKEND', config.get('booking_backend', 'mysql'))
    if backend == 'mysql':
        return MySQLBookingStore()
    if backend == 'sqlite':
        return DemoBookingStore(config['booking_db'])
    raise BookingUnavailable('WAYLOOM_BOOKING_BACKEND 仅支持 mysql 或显式离线 sqlite。')


def booking_action(config, action, *args, **kwargs):
    """Human UI and MCP share a safe error boundary; confirm stays UI-only."""
    if action not in ('seed_demo', 'query', 'prepare', 'confirm'):
        raise ValueError('Unknown booking action')
    try:
        return getattr(booking_store(config), action)(*args, **kwargs)
    except (BookingUnavailable, pymysql.MySQLError):
        return {'status': 'unavailable', 'source_kind': 'synthetic', 'message': UNAVAILABLE}
