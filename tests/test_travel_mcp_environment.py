"""Embedding configuration survives filtered stdio environments without API calls."""
import asyncio
import copy
import json
import os
import subprocess
import tempfile
import textwrap
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from wayloom.mcp_client import stdio_environment
from wayloom.config import load_config
from wayloom.knowledge import ingest
from wayloom.model import Embeddings


class TravelMCPEnvironmentTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.server = json.loads((root / 'config/travel.json').read_text())['mcp_servers'][0]
        self.dialogue = {'WAYLOOM_BASE_URL': 'https://mock.invalid/v1/',
                         'WAYLOOM_API_KEY': 'fake-dialogue-key',
                         'WAYLOOM_EMBEDDING_MODEL': 'fake-embedding'}

    def test_environment_only_child_matches_ingestion_embedder(self):
        with patch.dict(os.environ, self.dialogue, clear=True):
            parent = Embeddings()
            environment = stdio_environment(self.server)
        self.assertNotIn('WAYLOOM_API_KEY', environment)
        self.assertNotIn('WAYLOOM_BASE_URL', environment)
        # Real subprocess without loading .env, matching an env-only container.
        result = subprocess.run([sys.executable, '-c',
            'import json; from wayloom.model import Embeddings; e=Embeddings(); '
            'print(json.dumps([e.identity, e.key]))'], env=environment,
            check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(result.stdout), [parent.identity, parent.key])

    def test_explicit_embedding_settings_take_precedence(self):
        settings = dict(self.dialogue, WAYLOOM_EMBEDDING_BASE_URL='https://embedding.invalid/v1',
                        WAYLOOM_EMBEDDING_API_KEY='fake-embedding-key')
        with patch.dict(os.environ, settings, clear=True):
            environment = stdio_environment(self.server)
        self.assertEqual(environment['WAYLOOM_EMBEDDING_BASE_URL'], settings['WAYLOOM_EMBEDDING_BASE_URL'])
        self.assertEqual(environment['WAYLOOM_EMBEDDING_API_KEY'], 'fake-embedding-key')
        self.assertNotIn('fake-dialogue-key', environment.values())

    def test_legacy_fallback_and_empty_embedding_override(self):
        legacy = {'SMARTVOYAGE_BASE_URL': 'https://mock.invalid/v1',
                  'SMARTVOYAGE_API_KEY': 'fake-legacy-key',
                  'SMARTVOYAGE_EMBEDDING_MODEL': 'fake-model',
                  'WAYLOOM_EMBEDDING_BASE_URL': '', 'WAYLOOM_EMBEDDING_API_KEY': ''}
        with patch.dict(os.environ, legacy, clear=True):
            parent = Embeddings()
            environment = stdio_environment(self.server)
        with patch.dict(os.environ, environment, clear=True):
            child = Embeddings()
            self.assertEqual((child.identity, child.key), (parent.identity, parent.key))

    def test_other_servers_never_implicitly_receive_dialogue_credentials(self):
        for overrides in [{'args': ['-m', 'third_party']}, {'command': 'third-party'},
                          {'args': ['-m', 'wayloom.tools', '--extra']}]:
            with self.subTest(overrides=overrides), patch.dict(os.environ, self.dialogue, clear=True):
                environment = stdio_environment(dict(self.server, **overrides))
                self.assertNotIn('WAYLOOM_MCP_RESOLVED_EMBEDDINGS', environment)
                self.assertNotIn('WAYLOOM_API_KEY', environment)
                self.assertNotIn('WAYLOOM_EMBEDDING_API_KEY', environment)
                self.assertNotIn('WAYLOOM_EMBEDDING_BASE_URL', environment)

    def test_pass_env_opt_in_is_still_required(self):
        server = copy.deepcopy(self.server)
        server['pass_env'] = []
        with patch.dict(os.environ, self.dialogue, clear=True):
            environment = stdio_environment(server)
        self.assertFalse(any(key.startswith('WAYLOOM_EMBEDDING_') for key in environment))
        self.assertNotIn('WAYLOOM_MCP_RESOLVED_EMBEDDINGS', environment)


    def test_real_server_preserves_mixed_dotenv_embedding_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'docs').mkdir()
            (root / 'docs/rule.txt').write_text('existing travel rule', encoding='utf-8')
            (root / 'config.json').write_text(json.dumps({
                'agents': [], 'mcp_servers': [], 'knowledge_dir': 'docs',
                'index_path': 'index.json', 'booking_db': 'booking.sqlite3'}))
            # Supported partial .env: dialogue credentials are supplied only by
            # the container environment, while blank embeddings request fallback.
            (root / '.env').write_text('WAYLOOM_EMBEDDING_MODEL=fake-embedding\n'
                                      'WAYLOOM_EMBEDDING_BASE_URL=\n'
                                      'WAYLOOM_EMBEDDING_API_KEY=\n')
            settings = dict(self.dialogue, WAYLOOM_CONFIG=str(root / 'config.json'))
            with patch.dict(os.environ, settings, clear=True), patch('wayloom.config.ROOT', root):
                config = load_config()
                parent = Embeddings()
                with patch.object(Embeddings, 'encode', return_value=[[1.0, 0.0]]):
                    ingest(config, parent)
                environment = stdio_environment(self.server)
            code = textwrap.dedent("""
                import asyncio, json, sys
                from pathlib import Path
                from unittest.mock import patch
                import wayloom.config as config
                from wayloom.model import Embeddings
                from wayloom.tools import build_server
                config.ROOT = Path(sys.argv[1])
                seen = []
                def encode(self, texts):
                    seen.append([self.identity, self.key])
                    return [[1.0, 0.0] for _ in texts]
                with patch.object(Embeddings, 'encode', encode):
                    server = build_server()
                    asyncio.run(server.call_tool('search_knowledge', {'query': 'travel'}))
                print(json.dumps(seen))
            """)
            child = subprocess.run([sys.executable, '-c', code, str(root)], env=environment,
                                   capture_output=True, text=True)
            self.assertEqual(child.returncode, 0, child.stderr)
            self.assertEqual(json.loads(child.stdout), [[parent.identity, parent.key]])

    def test_standalone_server_keeps_dotenv_precedence(self):
        from wayloom.tools import build_server

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text(json.dumps({
                'agents': [], 'mcp_servers': [], 'knowledge_dir': 'docs',
                'index_path': 'index.json', 'booking_db': 'booking.sqlite3'}))
            (root / '.env').write_text('WAYLOOM_EMBEDDING_MODEL=dotenv-model\n'
                                      'WAYLOOM_EMBEDDING_BASE_URL=https://dotenv.invalid/v1\n'
                                      'WAYLOOM_EMBEDDING_API_KEY=fake-dotenv-key\n')
            settings = dict(self.dialogue, WAYLOOM_CONFIG=str(root / 'config.json'),
                            WAYLOOM_EMBEDDING_BASE_URL='https://env.invalid/v1',
                            WAYLOOM_EMBEDDING_API_KEY='fake-env-key')
            with patch.dict(os.environ, settings, clear=True), patch('wayloom.config.ROOT', root), \
                    patch('wayloom.tools.TravelKnowledge') as knowledge:
                knowledge.return_value.search.return_value = {'status': 'no_evidence'}
                server = build_server()
                asyncio.run(server.call_tool('search_knowledge', {'query': 'travel'}))
                embedding = knowledge.call_args.args[1]
                self.assertEqual(embedding.identity, 'https://dotenv.invalid/v1|dotenv-model')
                self.assertEqual(embedding.key, 'fake-dotenv-key')
