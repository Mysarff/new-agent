"""Upload validation must not poison the existing corpus or index."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from wayloom.knowledge import TravelKnowledge, fingerprint, ingest, save_upload


class TravelUploadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.docs = self.root / 'docs'
        self.docs.mkdir()
        (self.docs / 'rule.txt').write_text('existing travel rule', encoding='utf-8')
        self.config = {'knowledge_dir': str(self.docs), 'index_path': str(self.root / 'index.json')}
        ingest(self.config)
        self.files = fingerprint(self.docs)
        self.index = Path(self.config['index_path']).read_bytes()

    def assert_old_retrieval_works(self):
        self.assertEqual(fingerprint(self.docs), self.files)
        self.assertEqual(Path(self.config['index_path']).read_bytes(), self.index)
        self.assertEqual(TravelKnowledge(self.config).search('existing')['status'], 'candidates')
        self.assertEqual(list(self.docs.glob('.travel-upload-*')), [])

    def test_invalid_json_documents_leave_corpus_and_retrieval_unchanged(self):
        bad = [{'title': 'test'}, {'text': ''}, {'text': '  '}, None, 'text', 3, [],
               [{'text': 'valid'}, {'title': 'invalid second document'}],
               {'text': 'ok', 'title': 3}, {'text': 'ok', 'source': []},
               {'text': 'ok', 'valid_from': '2026-02-30'},
               {'text': 'ok', 'valid_from': '2026-10-01', 'valid_to': '2026-09-01'},
               {'text': 'ok', 'regions': 'CN'}, {'text': 'ok', 'tags': [1]},
               {'text': 'ok', 'extra': float('nan')}]
        for document in bad:
            with self.subTest(document=document), self.assertRaises(ValueError):
                save_upload(self.config, 'input.json', json.dumps(document).encode())
            self.assert_old_retrieval_works()

    def test_invalid_encoding_size_and_type_leave_retrieval_unchanged(self):
        for name, content in [('x.txt', b'\xff'), ('x.txt', b'x' * 2_000_001),
                              ('x.csv', b'text'), ('x.json', b'{'), ('x.md', b' ')]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                save_upload(self.config, name, content)
            self.assert_old_retrieval_works()

    def test_disk_failure_leaves_retrieval_unchanged(self):
        for operation in ('os.replace', 'os.fsync'):
            with patch('wayloom.knowledge.' + operation, side_effect=OSError('mock disk failure')):
                with self.assertRaises(OSError):
                    save_upload(self.config, 'valid.txt', b'valid travel information')
            self.assert_old_retrieval_works()

    def test_valid_batch_is_unverified_unique_and_ingestible(self):
        raw = [{'text': 'new travel rule', 'source_type': 'official_summary', 'id': 'forged',
                'checked_at': '2026-09-30', 'regions': ['CN']},
               {'text': 'another travel rule', 'id': 'forged'}]
        path = save_upload(self.config, '../../input.json', json.dumps(raw).encode())
        self.assertEqual(path.parent, self.docs)
        records = json.loads(path.read_text())
        self.assertEqual(len({record['id'] for record in records}), 2)
        self.assertTrue(all(record['source_type'] == 'user_supplied_unverified' for record in records))
        self.assertEqual(records[0]['checked_at'], '2026-09-30')
        with self.assertRaisesRegex(ValueError, 'changed'):
            TravelKnowledge(self.config).search('travel')
        self.assertEqual(ingest(self.config)['documents'], 3)
        self.assertEqual(TravelKnowledge(self.config).search('travel')['status'], 'candidates')

    def test_text_and_markdown_uploads_support_bom(self):
        for filename in ['notes.txt', 'notes.md']:
            save_upload(self.config, filename, '\ufeff旅行资料'.encode('utf-8'))
        self.assertEqual(ingest(self.config)['documents'], 3)
