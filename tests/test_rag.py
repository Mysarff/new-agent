import json
import tempfile
import unittest
from pathlib import Path

from opsatlas.config import load_config
from opsatlas.rag import Retriever, build_index, split_document, tokens


class RagTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.knowledge = self.directory / 'docs'
        self.knowledge.mkdir()
        self.config = dict(load_config(), knowledge_dir=str(self.knowledge),
                           index_path=str(self.directory / 'index.json'), chunk_size=100, chunk_overlap=20)

    def document(self, name, text):
        (self.knowledge / name).write_text(text, encoding='utf-8')

    def test_chinese_and_english_tokens(self):
        self.assertIn('连接', tokens('PostgreSQL 连接池'))
        self.assertIn('postgresql', tokens('PostgreSQL 连接池'))

    def test_long_sections_preserve_offsets_and_coverage(self):
        body = '# 标题\n\n## 章节\n' + '记录中文与 English 信息。\n' * 80
        chunks = split_document(body, 'doc.md', 100, 20)
        covered = set()
        for chunk in chunks:
            self.assertEqual(body[chunk['start']:chunk['end']], chunk['text'])
            self.assertLessEqual(len(chunk['text']), 100)
            covered.update(range(chunk['start'], chunk['end']))
        self.assertEqual(covered, set(range(len(body))))
        self.assertEqual(chunks, split_document(body, 'doc.md', 100, 20))

    def test_invalid_chunk_settings(self):
        for size, overlap in [(80, 80), (50, 10), (100, -1)]:
            with self.assertRaises(ValueError):
                split_document('abc', 'doc', size, overlap)

    def test_deleted_and_modified_documents_need_rebuild(self):
        self.document('first.md', '# 连接池\n数据库连接超时')
        build_index(self.config)
        retriever = Retriever(self.config)
        self.document('first.md', '# 新文档\n缓存问题')
        with self.assertRaisesRegex(ValueError, 'changed'):
            retriever.search('连接池')
        build_index(self.config)
        (self.knowledge / 'first.md').unlink()
        self.document('second.md', '# Queue\n消费者吞吐')
        build_index(self.config)
        self.assertTrue(all(c['source'] == 'second.md' for c in Retriever(self.config).chunks))

    def test_empty_index_does_not_overwrite_existing(self):
        self.document('one.md', '# 数据库\n连接')
        build_index(self.config)
        before = Path(self.config['index_path']).read_bytes()
        (self.knowledge / 'one.md').unlink()
        with self.assertRaises(ValueError):
            build_index(self.config)
        self.assertEqual(before, Path(self.config['index_path']).read_bytes())

    def test_relevance_no_match_and_metadata(self):
        self.document('db.md', '<!-- metadata: {"provenance":"synthetic"} -->\n# 数据库\n连接池耗尽与长事务')
        self.document('queue.md', '# Queue\n消息队列消费积压')
        build_index(self.config)
        retriever = Retriever(self.config)
        result = retriever.search('连接池耗尽')
        self.assertEqual(result['chunks'][0]['source'], 'db.md')
        self.assertEqual(result['chunks'][0]['metadata']['provenance'], 'synthetic')
        self.assertEqual(retriever.search('zxqv98765')['status'], 'no_evidence')
        with self.assertRaises(ValueError):
            retriever.search('连接池', 0)

    def test_dense_roundtrip_and_identity_guard(self):
        class EmbeddingFixture:
            identity = 'test-fixture:v1'
            def encode(self, texts):
                return [[1, 0] if 'database' in t else [0, 1] for t in texts]
        self.document('db.md', '# database\nconnection')
        self.document('queue.md', '# queue\nbacklog')
        build_index(self.config, EmbeddingFixture())
        result = Retriever(self.config, EmbeddingFixture()).search('database')
        self.assertEqual(result['mode'], 'hybrid_bm25_dense_rrf')
        self.assertEqual(result['chunks'][0]['source'], 'db.md')
        with self.assertRaisesRegex(ValueError, 'provider'):
            Retriever(self.config).search('database')

    def test_failed_embeddings_preserve_previous_index(self):
        class BadEmbedding:
            identity = 'bad'
            def encode(self, texts):
                return [[float('nan')]] * len(texts)
        self.document('db.md', '# database\nconnection')
        build_index(self.config)
        before = Path(self.config['index_path']).read_bytes()
        with self.assertRaises(ValueError):
            build_index(self.config, BadEmbedding())
        self.assertEqual(before, Path(self.config['index_path']).read_bytes())
