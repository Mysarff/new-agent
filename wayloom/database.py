"""Explicit local database setup and offline SQLite-to-MySQL migration."""
import argparse
import json
from pathlib import Path
import secrets
import shutil
import sqlite3
import pymysql

from dotenv import dotenv_values, set_key

from .booking import BookingUnavailable, MySQLBookingStore, TABLES
from .config import ROOT, load_config


COLUMNS = {
    'demo_tickets': ('id', 'kind', 'departure', 'arrival', 'date', 'service', 'seat',
                     'price_cents', 'currency', 'remaining'),
    'demo_quotes': ('id', 'expires_at', 'payload', 'used_order_id'),
    'demo_orders': ('id', 'request_id', 'quote_id', 'payload'),
    'demo_request_keys': ('request_id', 'quote_id', 'order_id'),
}


def setup_local(root=ROOT):
    """Fill missing database settings only; do not replace API keys or passwords."""
    root = Path(root)
    env = root / '.env'
    if not env.exists():
        shutil.copyfile(root / '.env.example', env)
    current = dotenv_values(env)
    defaults = {'WAYLOOM_BOOKING_BACKEND': 'mysql', 'WAYLOOM_MYSQL_HOST': '127.0.0.1',
                'WAYLOOM_MYSQL_PORT': '3309', 'WAYLOOM_MYSQL_DATABASE': 'wayloom',
                'WAYLOOM_MYSQL_USER': 'wayloom', 'WAYLOOM_MYSQL_PASSWORD': secrets.token_urlsafe(32)}
    for key, value in defaults.items():
        if not current.get(key):
            set_key(str(env), key, value)
    secret = root / 'var' / 'mysql-root-password'
    secret.parent.mkdir(parents=True, exist_ok=True)
    if not secret.exists():
        with secret.open('x', encoding='utf-8') as file:
            file.write(secrets.token_urlsafe(48))
        secret.chmod(0o600)
    return {'status': 'success', 'message': '数据库配置已补齐；既有配置与 API 密钥保留，密码不输出。'}


def migrate_sqlite(source, target):
    """Copy a consistent read-only snapshot into an empty, initialized MySQL DB.

    Stop booking writers before running. No DDL occurs inside the import, so a
    failed import rolls back every row. The source (including its WAL) is kept.
    """
    path = Path(source).resolve(strict=True)
    connection = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute('BEGIN')
        rows = {table: [dict(row) for row in connection.execute(
            f'SELECT {",".join(columns)} FROM {table}')]
            for table, columns in COLUMNS.items()}
    finally:
        connection.close()
    # Fail closed on inconsistent legacy relations; do not import broken replays.
    tickets = {row['id'] for row in rows['demo_tickets']}
    quotes = {row['id']: row for row in rows['demo_quotes']}
    orders = {row['id']: row for row in rows['demo_orders']}
    keys = {row['request_id']: row for row in rows['demo_request_keys']}
    for row in rows['demo_quotes']:
        if json.loads(row['payload'])['ticket']['id'] not in tickets:
            raise ValueError('旧报价引用的模拟票不存在，迁移已取消。')
        if row['used_order_id'] and row['used_order_id'] not in orders:
            raise ValueError('旧报价引用的模拟订单不存在，迁移已取消。')
    for row in rows['demo_orders']:
        quote = quotes.get(row['quote_id'])
        if not quote or quote['used_order_id'] != row['id']:
            raise ValueError('旧订单与报价不一致，迁移已取消。')
        keys.setdefault(row['request_id'], {'request_id': row['request_id'],
                        'quote_id': row['quote_id'], 'order_id': row['id']})
    rows['demo_request_keys'] = list(keys.values())
    for row in rows['demo_request_keys']:
        order = orders.get(row['order_id'])
        if not order or order['quote_id'] != row['quote_id']:
            raise ValueError('旧幂等键与订单不一致，迁移已取消。')
    with target.connection(write=True) as db:
        if any(db.execute(f'SELECT COUNT(*) AS n FROM {table}').fetchone()['n'] for table in TABLES):
            raise ValueError('MySQL 已有业务数据，拒绝覆盖；请使用空的独立数据库。')
        for table, columns in COLUMNS.items():
            statement = f'INSERT INTO {table} ({",".join(columns)}) VALUES ({",".join("?" for _ in columns)})'
            for row in rows[table]:
                db.execute(statement, tuple(row[column] for column in columns))
            count = db.execute(f'SELECT COUNT(*) AS n FROM {table}').fetchone()['n']
            if count != len(rows[table]):
                raise ValueError('迁移记录数不一致，已回滚。')
    return {'status': 'success', 'counts': {table: len(items) for table, items in rows.items()},
            'message': '迁移已提交；旧 SQLite 文件保留，报价有效期不延长。'}


def main():
    parser = argparse.ArgumentParser(description='Wayloom MySQL management')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('setup', help='Generate missing local database settings; preserve existing secrets')
    sub.add_parser('init', help='Create empty tables; never seed demo data')
    sub.add_parser('status', help='Check MySQL connection and table counts')
    migration = sub.add_parser('migrate-sqlite', help='Offline import into empty MySQL; keep source')
    migration.add_argument('--source', required=True)
    args = parser.parse_args()
    try:
        if args.command == 'setup':
            result = setup_local()
        else:
            load_config()
            store = MySQLBookingStore()
            if args.command == 'init':
                result = store.initialize()
            elif args.command == 'status':
                result = store.health()
            else:
                result = migrate_sqlite(args.source, store)
    except (BookingUnavailable, ValueError, OSError, sqlite3.Error, pymysql.MySQLError):
        # Driver errors can contain credentials or imported payloads; show no raw traceback.
        parser.exit(1, '数据库操作未完成。检查 Docker、连接配置、表初始化及迁移源/空库条件；原库不会被覆盖。\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
