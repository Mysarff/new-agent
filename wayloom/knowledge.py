"""Source-bearing travel retrieval with date filtering and optional dense RRF.

Documents are local evidence snapshots, never a live announcement feed. JSON
metadata is supplied by the curator; ``official_summary`` is not independently
authenticated by this module. Markdown/TXT imports remain unverified.
"""
import hashlib
import json
import math
import os
import re
import tempfile
import uuid
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np


INDEX_VERSION = 1
DOCUMENT_SUFFIXES = {'.json', '.md', '.txt'}
DATE_FIELDS = ('published_at', 'checked_at', 'valid_from', 'valid_to')


def tokens(text):
    """Latin words and Chinese bigrams; avoid matches on common single glyphs."""
    terms = re.findall(r'[a-z0-9_./-]+', text.casefold())
    for phrase in re.findall(r'[\u4e00-\u9fff]+', text):
        terms.extend(phrase[i:i + 2] for i in range(len(phrase) - 1))
        if len(phrase) == 1:
            terms.append(phrase)
    return terms


def _day(value, name):
    if value is None or value == '':
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise ValueError(f'{name} must be YYYY-MM-DD')
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f'{name} must be a valid YYYY-MM-DD date') from exc


def _settings(config):
    size, overlap = config.get('chunk_size', 650), config.get('chunk_overlap', 100)
    if (type(size) is not int or type(overlap) is not int
            or size < 80 or not 0 <= overlap < size):
        raise ValueError('Require chunk_size >= 80 and 0 <= chunk_overlap < chunk_size')
    return size, overlap


def fingerprint(directory, index_path=None):
    directory = Path(directory)
    excluded = Path(index_path).resolve() if index_path else None
    if not directory.is_dir():
        raise ValueError('Knowledge directory does not exist')
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob('*'))
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in DOCUMENT_SUFFIXES
        and path.resolve() != excluded and not any(part.startswith('.') for part in path.relative_to(directory).parts)
    }


def _document(raw, local_path):
    if not isinstance(raw, dict) or not isinstance(raw.get('text'), str) or not raw['text'].strip():
        raise ValueError(f'{local_path}: document requires nonempty text')
    record = dict(raw)
    record.setdefault('id', local_path)
    record.setdefault('title', local_path)
    record.setdefault('source', local_path)
    record.setdefault('source_type', 'user_supplied_unverified')
    for key in ('id', 'title', 'source', 'source_type'):
        if not isinstance(record[key], str) or not record[key].strip():
            raise ValueError(f'{local_path}: {key} must be a nonempty string')
    for key in DATE_FIELDS:
        parsed = _day(record.get(key), key)
        record[key] = parsed.isoformat() if parsed else None
    if record['valid_from'] and record['valid_to'] and record['valid_from'] > record['valid_to']:
        raise ValueError(f'{local_path}: valid_from must be <= valid_to')
    for key in ('regions', 'tags'):
        record.setdefault(key, [])
        if not isinstance(record[key], list) or any(not isinstance(v, str) for v in record[key]):
            raise ValueError(f'{local_path}: {key} must be a list of strings')
    record['local_path'] = local_path
    return record


def save_upload(config, filename, content):
    """Validate the entire upload before atomically publishing any knowledge file.

    Reuse ingestion's document schema so a rejected upload cannot invalidate the
    current index. Successful imports still require an explicit index rebuild.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in DOCUMENT_SUFFIXES or len(content) > 2_000_000:
        raise ValueError('Unsupported upload type or size')
    text = content.decode('utf-8-sig')
    raw = json.loads(text) if suffix == '.json' else {'title': Path(filename).name, 'text': text}
    items = raw if isinstance(raw, list) else [raw]
    if not items:
        raise ValueError('Upload must contain at least one document')
    target = Path(config['knowledge_dir']) / ('upload-' + uuid.uuid4().hex + '.json')
    documents = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError('Upload documents must be objects')
        document = dict(item, id='upload-' + uuid.uuid4().hex,
                        source_type='user_supplied_unverified')
        documents.append(_document(document, target.name))
    # Serialize before touching the directory, including unknown metadata fields.
    payload = json.dumps(documents, ensure_ascii=False, indent=2, allow_nan=False)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=target.parent,
                                         prefix='.travel-upload-', delete=False) as handle:
            temp = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
    finally:
        if temp and Path(temp).exists():
            Path(temp).unlink()
    return target


def _load_documents(directory, files):
    documents = []
    for filename in files:
        text = (Path(directory) / filename).read_text(encoding='utf-8-sig')
        if filename.lower().endswith('.json'):
            raw = json.loads(text)
            items = raw if isinstance(raw, list) else [raw]
            documents.extend(_document(item, filename) for item in items)
        else:
            title = next((line.lstrip('# ').strip() for line in text.splitlines()
                          if line.startswith('# ')), filename)
            # Imported prose cannot elevate its trust using an embedded comment.
            documents.append(_document({'text': text, 'title': title}, filename))
    ids = [d['id'] for d in documents]
    if len(set(ids)) != len(ids):
        raise ValueError('Knowledge document IDs must be unique')
    return documents


def split_document(document, size=650, overlap=100):
    """Character offsets refer to the exact JSON ``text`` or imported file text."""
    _settings({'chunk_size': size, 'chunk_overlap': overlap})
    body = document['text']
    metadata = {key: value for key, value in document.items() if key not in ('text', 'title', 'source')}
    boundaries = sorted(set([0, len(body)] + [m.start() for m in re.finditer(r'^#{1,6} ', body, re.M)]))
    heading, chunks = document['title'], []
    for left, right in zip(boundaries, boundaries[1:]):
        if body[left:right].startswith('#'):
            heading = body[left:right].splitlines()[0].lstrip('# ').strip()
        start = left
        while start < right:
            end = min(start + size, right)
            if end < right:
                newline = body.rfind('\n', start + size // 2, end)
                if newline > start:
                    end = newline + 1
            text = body[start:end]
            if text.strip():
                signature = json.dumps([document['id'], document['source'], start, end, text], ensure_ascii=False)
                identity = hashlib.sha256(signature.encode('utf-8')).hexdigest()[:20]
                chunks.append({'id': 'K-' + identity, 'document_id': document['id'],
                               'title': document['title'], 'heading': heading,
                               'source': document['source'], 'text': text,
                               'start': start, 'end': end, 'metadata': metadata})
            if end == right:
                break
            start = max(start + 1, end - overlap)
    return chunks


def _normalize(vectors):
    matrix = np.asarray(vectors, dtype=float)
    if (matrix.ndim != 2 or not matrix.shape[1] or not np.isfinite(matrix).all()
            or np.any(np.linalg.norm(matrix, axis=1) == 0)):
        raise ValueError('Embedding vectors must be finite, nonzero and rectangular')
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def ingest(config, embedder=None):
    size, overlap = _settings(config)
    files = fingerprint(config['knowledge_dir'], config['index_path'])
    documents = _load_documents(config['knowledge_dir'], files)
    chunks = [chunk for doc in documents for chunk in split_document(doc, size, overlap)]
    if not chunks:
        raise ValueError('No JSON/Markdown/TXT knowledge documents found')
    data = {'version': INDEX_VERSION, 'created_at': datetime.now(timezone.utc).isoformat(),
            'files': files, 'chunk_size': size, 'chunk_overlap': overlap,
            'documents': len(documents), 'chunks': chunks}
    if embedder is not None:
        vectors = _normalize(embedder.encode([c['title'] + '\n' + c['text'] for c in chunks]))
        if len(vectors) != len(chunks):
            raise ValueError('Embedding count mismatch')
        data.update(embedding_identity=embedder.identity, vectors=vectors.tolist())
    if fingerprint(config['knowledge_dir'], config['index_path']) != files:
        raise ValueError('Knowledge files changed during ingestion; rebuild index')
    target = Path(config['index_path'])
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=target.parent,
                                         prefix='.travel-index-', delete=False) as handle:
            temp = handle.name
            json.dump(data, handle, ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
    finally:
        if temp and Path(temp).exists():
            Path(temp).unlink()
    return {'documents': len(documents), 'chunks': len(chunks),
            'mode': 'hybrid_bm25_dense_rrf' if embedder is not None else 'bm25',
            'index_path': str(target), 'created_at': data['created_at']}


def _temporal_status(metadata, target):
    if metadata.get('published_at') and metadata['published_at'] > target:
        return 'not_yet_published'
    if metadata.get('valid_from') and metadata['valid_from'] > target:
        return 'not_yet_effective'
    if metadata.get('valid_to') and metadata['valid_to'] < target:
        return 'expired'
    if metadata.get('valid_from') or metadata.get('valid_to'):
        return 'within_recorded_period'
    return 'validity_not_specified'


def _region_matches(regions, region):
    # Hierarchies are caller-supplied, e.g. CN/北京. No city allowlist or guessed country.
    if not region or not regions or '*' in regions:
        return True
    needle = region.strip().casefold().strip('/')
    return any(needle == item.strip().casefold().strip('/')
               or needle.startswith(item.strip().casefold().strip('/') + '/') for item in regions)


class TravelKnowledge:
    def __init__(self, config, embedder=None):
        self.config, self.embedder = dict(config), embedder
        self.data = json.loads(Path(config['index_path']).read_text(encoding='utf-8'))
        if self.data.get('version') != INDEX_VERSION:
            raise ValueError('Unsupported knowledge index version; rebuild index')
        self.chunks = self.data['chunks']
        self.counts = [Counter(tokens(' '.join([c['title'], c['heading'], c['text'],
                                              ' '.join(c['metadata'].get('tags', []))]))) for c in self.chunks]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average = sum(self.lengths) / max(1, len(self.lengths)) or 1
        self.df = Counter(term for counts in self.counts for term in counts)

    def search(self, query, travel_date=None, region=None, top_k=5, include_historical=False):
        if not isinstance(query, str) or not query.strip() or len(query) > 4000:
            raise ValueError('Query must contain 1..4000 characters')
        if top_k is None:
            top_k = self.config.get('top_k', 5)
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError('top_k must be 1..20')
        if type(include_historical) is not bool:
            raise ValueError('include_historical must be boolean')
        if region is not None and (not isinstance(region, str) or len(region) > 200):
            raise ValueError('region must be a string of at most 200 characters')
        target = (_day(travel_date, 'travel_date') or date.today()).isoformat()
        if fingerprint(self.config['knowledge_dir'], self.config['index_path']) != self.data['files']:
            raise ValueError('Knowledge files changed; rebuild index before querying')
        if _settings(self.config) != (self.data['chunk_size'], self.data['chunk_overlap']):
            raise ValueError('Chunk settings changed; rebuild index')
        eligibility, excluded = {}, Counter()
        for index, chunk in enumerate(self.chunks):
            meta = chunk['metadata']
            status = _temporal_status(meta, target)
            if meta.get('source_type') == 'synthetic' and not self.config.get('include_synthetic', False):
                excluded['synthetic'] += 1
            elif status in ('not_yet_effective', 'not_yet_published'):
                excluded[status] += 1
            elif status == 'expired' and not include_historical:
                excluded['expired'] += 1
            elif not _region_matches(meta.get('regions', []), region):
                excluded['other_region'] += 1
            else:
                eligibility[index] = status
        scores = {}
        for index in eligibility:
            counts, score = self.counts[index], 0.0
            for term in set(tokens(query)):
                frequency = counts[term]
                if frequency:
                    idf = math.log(1 + (len(self.chunks) - self.df[term] + 0.5) / (self.df[term] + 0.5))
                    score += idf * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * self.lengths[index] / self.average))
            if score > 0:
                scores[index] = score
        lexical = sorted(scores, key=scores.get, reverse=True)[:30]
        fused = {index: 1 / (60 + rank) for rank, index in enumerate(lexical, 1)}
        mode, similarities = 'bm25', {}
        if 'vectors' in self.data:
            if self.embedder is None or self.embedder.identity != self.data['embedding_identity']:
                raise ValueError('Embedding provider changed or absent; use indexed provider or rebuild')
            encoded = _normalize(self.embedder.encode([query]))
            if len(encoded) != 1:
                raise ValueError('Query embedding count mismatch')
            matrix = _normalize(self.data['vectors'])
            if matrix.shape != (len(self.chunks), encoded.shape[1]):
                raise ValueError('Embedding dimensions/count changed; rebuild index')
            values = matrix @ encoded[0]
            threshold = float(self.config.get('dense_min_similarity', 0.25))
            if not math.isfinite(threshold) or not -1 <= threshold <= 1:
                raise ValueError('dense_min_similarity must be -1..1')
            similarities = {index: float(values[index]) for index in eligibility if values[index] >= threshold}
            dense = sorted(similarities, key=similarities.get, reverse=True)[:30]
            for rank, index in enumerate(dense, 1):
                fused[index] = fused.get(index, 0) + 1 / (60 + rank)
            mode = 'hybrid_bm25_dense_rrf'
        ranking = sorted(fused, key=fused.get, reverse=True)[:top_k]
        found = []
        for index in ranking:
            chunk = self.chunks[index]
            found.append(dict(chunk, score=round(fused[index], 6),
                              lexical_score=round(scores.get(index, 0), 6),
                              dense_similarity=similarities.get(index), temporal_status=eligibility[index],
                              evidence_status=('unverified' if chunk['metadata']['source_type'] == 'user_supplied_unverified'
                                               else chunk['metadata']['source_type'])))
        return {'mode': mode, 'status': 'candidates' if found else 'no_evidence',
                'query_date': target, 'region': region, 'include_historical': include_historical,
                'index_created_at': self.data['created_at'], 'filtered_chunks': dict(excluded),
                'notice': ('检索结果是本地资料候选证据，分数不是事实置信度；核验日期不代表持续有效。'
                           '需核对来源、适用地区和行程日期，出行前复核官方最新信息。'
                           '未检索到证据不等于没有公告；历史资料不能当作当前公告。'),
                'chunks': found}
