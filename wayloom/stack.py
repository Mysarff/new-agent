"""Start the travel UI and four independent A2A specialists; clean up owned children."""
import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from .config import ROOT, load_config
from .knowledge import ingest


def stop_children(children):
    for child in children:
        if child.poll() is None:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True,
                               creationflags=subprocess.CREATE_NO_WINDOW, timeout=10, check=False)
            else:
                child.terminate()
    for child in children:
        try:
            child.wait(timeout=8)
        except subprocess.TimeoutExpired:
            child.kill()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8510)
    parser.add_argument('--no-ui', action='store_true')
    args = parser.parse_args()
    config = load_config()
    if not Path(config['index_path']).exists():
        ingest(config)
    ports = [a['port'] for a in config['agents']] + ([] if args.no_ui else [args.port])
    for port in ports:
        with socket.socket() as sock:
            try:
                sock.bind(('127.0.0.1', port))
            except OSError:
                parser.error(f'端口{port}已被使用，请先停止对应的旧服务。')
    env = {**os.environ, 'WAYLOOM_A2A': '1', 'PYTHONUTF8': '1', 'PYTHONUNBUFFERED': '1'}
    commands = [['-m', 'wayloom.a2a', '--agent', agent['id']] for agent in config['agents']]
    if not args.no_ui:
        commands.append(['-m', 'streamlit', 'run', 'app.py', '--server.address', '127.0.0.1',
                         '--server.port', str(args.port), '--server.headless', 'true', '--browser.gatherUsageStats', 'false'])
    children = []
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        for command in commands:
            children.append(subprocess.Popen([sys.executable, *command], cwd=ROOT, env=env,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0))
        print(f'Wayloom: http://127.0.0.1:{args.port} ; Ctrl+C to stop', flush=True)
        while all(child.poll() is None for child in children):
            time.sleep(0.5)
        stopped = [(commands[i], child.returncode) for i, child in enumerate(children) if child.poll() is not None]
        raise RuntimeError(f'A child service stopped unexpectedly: {stopped}')
    except KeyboardInterrupt:
        pass
    finally:
        stop_children(children)


if __name__ == '__main__':
    main()
