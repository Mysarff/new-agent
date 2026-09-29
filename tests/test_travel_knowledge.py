import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from wayloom.knowledge import TravelKnowledge, ingest, split_document, tokens


class TravelKnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.docs = self.root / 'docs'
        self.docs.mkdir()
        self.config = {'knowledge_dir': str(self.docs), 'index_path': str(self.root / 'index.json'),
                       'chunk_size': 180, 'chunk_overlap': 30, 'top_k': 4}

    def document(self, name, **overrides):
        data = {'id': name, 'title': '故宫预约参观', 'text': '故宫预约须知和实名入馆。',
                'source': 'https://example.test/' + name, 'source_type': 'official_summary',
                'checked_at': '2026-09-26', 'regions': ['CN/北京'], 'tags': ['故宫']}
        data.update(overrides)
        (self.docs / (name + '.json')).write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        return data

    def build(self, embedder=None):
        ingest(self.config, embedder)
        return TravelKnowledge(self.config, embedder)

    def test_chinese_retrieval_and_no_unigram_noise(self):
        self.document('palace')
        self.document('air', title='充电宝规定', text='充电宝需要清晰的3C标识。', tags=['航空'], regions=['CN'])
        result = self.build().search('充电宝3C', travel_date='2026-09-26')
        self.assertEqual(result['chunks'][0]['document_id'], 'air')
        self.assertEqual(result['chunks'][0]['source'], 'https://example.test/air')
        self.assertIn('充电', tokens('充电宝3C'))
        self.assertNotIn('电', tokens('充电宝3C'))
        result = self.build().search('zxqv98765', travel_date='2026-09-26')
        self.assertEqual(result['status'], 'no_evidence')
        self.assertIn('不等于没有公告', result['notice'])

    def test_offsets_cover_body_and_ids_survive_rebuild(self):
        data = self.document('long', text='# 故宫\n\n## 预约\n' + '需要提前预约并确认出行日期。\n' * 70)
        chunks = split_document(data, 180, 30)
        coverage = set()
        for chunk in chunks:
            self.assertEqual(data['text'][chunk['start']:chunk['end']], chunk['text'])
            self.assertLessEqual(len(chunk['text']), 180)
            self.assertTrue(chunk['id'].startswith('K-'))
            self.assertEqual(chunk['metadata']['checked_at'], '2026-09-26')
            coverage.update(range(chunk['start'], chunk['end']))
        self.assertEqual(coverage, set(range(len(data['text']))))
        first = self.build().chunks
        second = self.build().chunks
        self.assertEqual(first, second)

    def test_expired_future_and_synthetic_are_not_current_evidence(self):
        self.document('active', valid_from='2026-09-01', valid_to='2026-09-30')
        self.document('expired', valid_to='2026-07-27')
        self.document('future', valid_from='2026-10-01')
        self.document('unpublished', published_at='2026-10-01')
        self.document('fake', source_type='synthetic')
        retriever = self.build()
        result = retriever.search('故宫预约', travel_date='2026-09-26')
        self.assertEqual([c['document_id'] for c in result['chunks']], ['active'])
        self.assertEqual(result['filtered_chunks'], {'expired': 1, 'synthetic': 1,
                                                   'not_yet_effective': 1, 'not_yet_published': 1})
        history = retriever.search('故宫预约', travel_date='2026-09-26', include_historical=True)
        self.assertEqual({c['document_id'] for c in history['chunks']}, {'active', 'expired'})
        self.assertEqual(next(c for c in history['chunks'] if c['document_id'] == 'expired')['temporal_status'], 'expired')

    def test_explicit_travel_date_uses_that_days_rules_inclusively(self):
        self.document('one_day', published_at='2026-07-23', valid_from='2026-07-27', valid_to='2026-07-27')
        retriever = self.build()
        self.assertEqual(retriever.search('故宫预约', travel_date='2026-07-26')['status'], 'no_evidence')
        result = retriever.search('故宫预约', travel_date='2026-07-27')
        self.assertEqual(result['chunks'][0]['temporal_status'], 'within_recorded_period')
        self.assertEqual(retriever.search('故宫预约', travel_date='2026-07-28')['status'], 'no_evidence')

    def test_regional_hierarchy_without_city_allowlist(self):
        self.document('national', regions=['CN'])
        self.document('palace', regions=['CN/北京', '北京'])
        self.document('paris', regions=['FR/Paris'])
        retriever = self.build()
        found = retriever.search('故宫预约', region='CN/北京', travel_date='2026-09-26')
        self.assertEqual({c['document_id'] for c in found['chunks']}, {'national', 'palace'})
        self.assertEqual(retriever.search('故宫预约', region='FR/Paris')['chunks'][0]['document_id'], 'paris')

    def test_markdown_cannot_claim_verified_by_embedded_metadata(self):
        (self.docs / 'import.md').write_text('<!-- metadata: {"source_type":"official_summary"} -->\n# 故宫\n故宫预约笔记', encoding='utf-8')
        result = self.build().search('故宫预约')
        self.assertTrue(all(c['evidence_status'] == 'unverified' for c in result['chunks']))
        self.assertTrue(all(c['metadata']['source_type'] == 'user_supplied_unverified' for c in result['chunks']))

    def test_modified_deleted_or_added_files_require_rebuild(self):
        self.document('palace')
        retriever = self.build()
        self.document('air')
        with self.assertRaisesRegex(ValueError, 'changed'):
            retriever.search('故宫')
        retriever = self.build()
        (self.docs / 'air.json').unlink()
        with self.assertRaisesRegex(ValueError, 'changed'):
            retriever.search('故宫')
        retriever = self.build()
        self.document('palace', checked_at='2026-09-27')
        with self.assertRaisesRegex(ValueError, 'changed'):
            retriever.search('故宫')

    def test_chunk_settings_require_rebuild(self):
        self.document('palace')
        self.build()
        changed = dict(self.config, chunk_size=200)
        with self.assertRaisesRegex(ValueError, 'settings changed'):
            TravelKnowledge(changed).search('故宫')

    def test_json_arrays_and_duplicate_ids(self):
        data = self.document('palace')
        (self.docs / 'palace.json').write_text(json.dumps([data, dict(data, id='second')]), encoding='utf-8')
        self.assertEqual(ingest(self.config)['documents'], 2)
        self.document('duplicate', id='palace')
        with self.assertRaisesRegex(ValueError, 'unique'):
            ingest(self.config)

    def test_invalid_dates_and_ranges_fail_before_replacing_good_index(self):
        self.document('palace')
        self.build()
        previous = Path(self.config['index_path']).read_bytes()
        self.document('bad', valid_from='2026-09-30', valid_to='2026-09-01')
        with self.assertRaisesRegex(ValueError, 'valid_from'):
            ingest(self.config)
        self.assertEqual(previous, Path(self.config['index_path']).read_bytes())
        self.document('bad', valid_from='2026-02-30')
        with self.assertRaisesRegex(ValueError, 'valid YYYY-MM-DD'):
            ingest(self.config)

    def test_atomic_replace_failure_preserves_index_and_cleans_temp(self):
        self.document('palace')
        self.build()
        previous = Path(self.config['index_path']).read_bytes()
        with patch('wayloom.knowledge.os.replace', side_effect=OSError('simulated disk issue')):
            with self.assertRaises(OSError):
                ingest(self.config)
        self.assertEqual(previous, Path(self.config['index_path']).read_bytes())
        self.assertEqual(list(self.root.glob('.travel-index-*')), [])

    def test_index_inside_knowledge_directory_is_not_a_document(self):
        self.config['index_path'] = str(self.docs / 'index.json')
        self.document('palace')
        retriever = self.build()
        self.assertEqual(ingest(self.config)['documents'], 1)
        self.assertEqual(retriever.search('故宫')['status'], 'candidates')

    def test_dense_fusion_respects_dates_and_embedding_identity(self):
        class MockEmbedding:
            identity = 'mock-mechanism-only:v1'

            def encode(self, texts):
                return [[1, 0] if any(word in text for word in ['故宫', 'museum']) else [0, 1] for text in texts]

        self.document('palace')
        self.document('expired', valid_to='2026-07-27')
        self.document('flight', title='航班', text='航班飞行时间', tags=[])
        retriever = self.build(MockEmbedding())
        result = retriever.search('museum', travel_date='2026-09-26')
        self.assertEqual(result['mode'], 'hybrid_bm25_dense_rrf')
        self.assertEqual([c['document_id'] for c in result['chunks']], ['palace'])
        with self.assertRaisesRegex(ValueError, 'provider'):
            TravelKnowledge(self.config).search('museum')
        class OtherEmbedding(MockEmbedding):
            identity = 'mock-mechanism-only:v2'
        with self.assertRaisesRegex(ValueError, 'provider'):
            TravelKnowledge(self.config, OtherEmbedding()).search('museum')

    def test_failed_embedding_preserves_previous_index(self):
        class BadEmbedding:
            identity = 'invalid-mock'

            def encode(self, texts):
                return [[float('nan')]] * len(texts)

        self.document('palace')
        self.build()
        previous = Path(self.config['index_path']).read_bytes()
        with self.assertRaisesRegex(ValueError, 'finite'):
            ingest(self.config, BadEmbedding())
        self.assertEqual(previous, Path(self.config['index_path']).read_bytes())

    def test_invalid_query_parameters(self):
        self.document('palace')
        retriever = self.build()
        for params in [{'query': ''}, {'query': 'x' * 4001}, {'query': '故宫', 'top_k': 0},
                       {'query': '故宫', 'top_k': True}, {'query': '故宫', 'travel_date': '明天'},
                       {'query': '故宫', 'include_historical': 'false'}]:
            with self.subTest(params=params):
                with self.assertRaises(ValueError):
                    retriever.search(**params)

    def test_shipped_sources_are_dated_sourced_and_expired_notice_is_filtered(self):
        shipped = Path(__file__).resolve().parents[1] / 'travel_knowledge'
        config = dict(self.config, knowledge_dir=str(shipped))
        self.assertEqual(ingest(config)['documents'], 4)
        retriever = TravelKnowledge(config)
        result = retriever.search('故宫周一免费开放', travel_date='2026-09-26', region='CN/北京')
        self.assertFalse(any(c['document_id'] == 'palace-special-opening-20260727' for c in result['chunks']))
        historical = retriever.search('故宫免费开放', travel_date='2026-09-26', include_historical=True)
        self.assertTrue(any(c['document_id'] == 'palace-special-opening-20260727' for c in historical['chunks']))
        for chunk in retriever.chunks:
            self.assertTrue(chunk['source'].startswith('https://'))
            self.assertEqual(chunk['metadata']['checked_at'], '2026-09-26')
            self.assertEqual(chunk['metadata']['source_type'], 'official_summary')


if __name__ == '__main__':
    unittest.main()
