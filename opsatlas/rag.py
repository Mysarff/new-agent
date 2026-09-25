"""Heading-aware chunks, source offsets, BM25 and optional dense RRF retrieval."""
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def tokens(text):
    words = re.findall(r'[a-z0-9_./-]+', text.lower())
    for sequence in re.findall(r'[\u4e00-\u9fff]+', text):
        words.extend(sequence)
        words.extend(sequence[i:i + 2] for i in range(len(sequence) - 1))
    return words


def fingerprint(directory):
    files = sorted(p for p in Path(directory).rglob('*') if p.suffix.lower() in ('.md', '.txt') and p.is_file())
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def split_document(text, source, size=650, overlap=100):
    if not 0 <= overlap < size or size < 80:
        raise ValueError('Require chunk size >= 80 and 0 <= overlap < size')
    title = next((line.lstrip('# ').strip() for line in text.splitlines() if line.startswith('# ')), source)
    metadata_match = re.search(r'<!-- metadata: (.*?) -->', text)
    metadata = json.loads(metadata_match[1]) if metadata_match else {'provenance': 'user_supplied_unverified'}
    boundaries = sorted(set([0, len(text)] + [m.start() for m in re.finditer(r'^#{1,6} ', text, re.M)]))
    heading, chunks = title, []
    for left, right in zip(boundaries, boundaries[1:]):
        section = text[left:right]
        if section.startswith('#'):
            heading = section.splitlines()[0].lstrip('# ').strip()
        start = left
        while start < right:
            end = min(start + size, right)
            if end < right:
                newline = text.rfind('\n', start + size // 2, end)
                if newline > start:
                    end = newline + 1
            body = text[start:end]
            if body.strip() and not body.strip().startswith('<!-- metadata:'):
                identity = hashlib.sha256(f'{source}:{start}:{end}:{body}'.encode()).hexdigest()[:16]
                chunks.append({'id': 'K' + identity, 'source': source, 'title': title, 'heading': heading,
                               'start': start, 'end': end, 'text': body, 'metadata': metadata})
            if end == right:
                break
            start = max(start + 1, end - overlap)
    return chunks


def normalize(vectors):
    matrix = np.asarray(vectors, dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or np.any(np.linalg.norm(matrix, axis=1) == 0):
        raise ValueError('Embedding vectors must be finite, nonzero and rectangular')
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def build_index(config, embedder=None):
    directory = Path(config['knowledge_dir'])
    files = fingerprint(directory)
    chunks = []
    for source in files:
        text = (directory / source).read_text(encoding='utf-8')
        chunks.extend(split_document(text, source, config['chunk_size'], config['chunk_overlap']))
    if not chunks:
        raise ValueError('No Markdown/TXT knowledge documents found')
    data = {'version': 1, 'created_at': datetime.now(timezone.utc).isoformat(), 'files': files,
            'chunk_size': config['chunk_size'], 'chunk_overlap': config['chunk_overlap'], 'chunks': chunks}
    if embedder:
        vectors = normalize(embedder.encode([c['title'] + '\n' + c['text'] for c in chunks]))
        if len(vectors) != len(chunks):
            raise ValueError('Embedding count mismatch')
        data.update(embedding_identity=embedder.identity, vectors=vectors.tolist())
    target = Path(config['index_path'])
    target.parent.mkdir(parents=True, exist_ok=True)
    # Same-directory atomic replacement keeps a good index if generation fails.
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=target.parent, delete=False) as handle:
        temp = handle.name
        json.dump(data, handle, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, target)
    return {'documents': len(files), 'chunks': len(chunks), 'mode': 'hybrid' if embedder else 'bm25'}


class Retriever:
    def __init__(self, config, embedder=None):
        self.config, self.embedder = config, embedder
        self.data = json.loads(Path(config['index_path']).read_text(encoding='utf-8'))
        self.chunks = self.data['chunks']
        self.counts = [Counter(tokens(c['title'] + ' ' + c['heading'] + ' ' + c['text'])) for c in self.chunks]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average = sum(self.lengths) / max(1, len(self.lengths))
        self.df = Counter(term for counts in self.counts for term in counts)

    def search(self, query, top_k=None):
        if not query.strip() or len(query) > 4000:
            raise ValueError('Query must contain 1..4000 characters')
        top_k = self.config['top_k'] if top_k is None else top_k
        if not 1 <= top_k <= 20:
            raise ValueError('top_k must be 1..20')
        if fingerprint(self.config['knowledge_dir']) != self.data['files']:
            raise ValueError('Knowledge files changed; rebuild index before querying')
        scores = []
        for i, counts in enumerate(self.counts):
            score = 0.0
            for term in set(tokens(query)):
                frequency = counts[term]
                if frequency:
                    idf = math.log(1 + (len(self.chunks) - self.df[term] + 0.5) / (self.df[term] + 0.5))
                    score += idf * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * self.lengths[i] / self.average))
            scores.append(score)
        lexical = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: scores[i], reverse=True)
        ranking, mode = lexical, 'bm25'
        fused = {i: 1 / (60 + rank) for rank, i in enumerate(lexical[:30], 1)}
        if 'vectors' in self.data:
            if not self.embedder or self.embedder.identity != self.data['embedding_identity']:
                raise ValueError('Embedding provider changed or absent; use the indexed provider or rebuild')
            vector = normalize(self.embedder.encode([query]))[0]
            matrix = normalize(self.data['vectors'])
            if matrix.shape[1] != len(vector):
                raise ValueError('Embedding dimensions changed; rebuild index')
            similarities = matrix @ vector
            dense = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:30]
            for rank, i in enumerate(dense, 1):
                fused[i] = fused.get(i, 0) + 1 / (60 + rank)
            ranking = sorted(fused, key=fused.get, reverse=True)
            mode = 'hybrid_bm25_dense_rrf'
        return {'mode': mode, 'status': 'candidates' if ranking else 'no_evidence',
                'notice': '检索命中仅是候选证据；分数不是事实置信度。模拟资料不可当真实生产规范。',
                'chunks': [dict(self.chunks[i], score=round(fused.get(i, 0), 6)) for i in ranking[:top_k]]}
