"""Run real integration tests in a disposable schema in this project's container.

Does not modify the wayloom business schema. Docker Desktop must be running.
The root secret stays in the container; no credential is passed on a command line.
"""
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wayloom.config import load_config
from wayloom.booking import MySQLSettings


def main():
    load_config()
    settings = MySQLSettings.from_env()
    if not re.fullmatch(r'[A-Za-z0-9_]+', settings.user):
        raise SystemExit('Test helper requires a simple MySQL username')
    database = 'wayloom_test_' + uuid.uuid4().hex[:12]
    admin = ['docker', 'compose', 'exec', '-T', 'mysql', 'sh', '-c',
             'export MYSQL_PWD="$(cat "$MYSQL_ROOT_PASSWORD_FILE")"; exec mysql -u root']
    def sql(statement):
        subprocess.run(admin, input=statement, text=True, encoding='utf-8', cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL)
    sql(f'CREATE DATABASE `{database}`;')
    granted = False
    try:
        sql(f"GRANT ALL ON `{database}`.* TO '{settings.user}'@'%';")
        granted = True
        environment = {**os.environ, 'WAYLOOM_TEST_MYSQL': '1', 'WAYLOOM_MYSQL_DATABASE': database,
                       'WAYLOOM_BOOKING_BACKEND': 'mysql', 'PYTHONUTF8': '1'}
        result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
                                 '-p', 'test_travel_mysql.py', '-v'], cwd=ROOT, env=environment)
        return result.returncode
    finally:
        # This name is generated above, never read from user configuration.
        try:
            if granted:
                sql(f"REVOKE ALL PRIVILEGES ON `{database}`.* FROM '{settings.user}'@'%';")
        finally:
            sql(f'DROP DATABASE `{database}`;')


if __name__ == '__main__':
    raise SystemExit(main())
