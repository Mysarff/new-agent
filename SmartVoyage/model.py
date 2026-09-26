"""Provider-neutral Chat Completions and embeddings adapters; no fake model fallback."""
import os

import httpx


class ChatModel:
    def __init__(self):
        self.model = os.getenv('SMARTVOYAGE_MODEL', '')
        self.key = os.getenv('SMARTVOYAGE_API_KEY', '')
        self.base = os.getenv('SMARTVOYAGE_BASE_URL', '').rstrip('/')
        if not self.model or not self.key or not self.base:
            raise ValueError('请在 .env 配置 SMARTVOYAGE_MODEL、SMARTVOYAGE_BASE_URL 和 SMARTVOYAGE_API_KEY。')

    async def complete(self, messages, tools, tool_choice='auto'):
        payload = {'model': self.model, 'messages': messages}
        if tools:
            payload.update(tools=tools, tool_choice=tool_choice)
        async with httpx.AsyncClient(timeout=90, follow_redirects=False) as client:
            response = await client.post(self.base + '/chat/completions', json=payload,
                                         headers={'Authorization': f'Bearer {self.key}'})
            response.raise_for_status()
            data = response.json()
        message = data['choices'][0]['message']
        return {k: v for k, v in message.items() if k in ('role', 'content', 'tool_calls')}


class Embeddings:
    def __init__(self):
        self.model = os.getenv('SMARTVOYAGE_EMBEDDING_MODEL', '')
        self.base = os.getenv('SMARTVOYAGE_EMBEDDING_BASE_URL') or os.getenv('SMARTVOYAGE_BASE_URL', '')
        self.key = os.getenv('SMARTVOYAGE_EMBEDDING_API_KEY') or os.getenv('SMARTVOYAGE_API_KEY', '')

    @property
    def identity(self):
        return self.base.rstrip('/') + '|' + self.model

    def encode(self, texts):
        if not self.model or not self.key:
            raise ValueError('Embedding provider is not configured')
        vectors = []
        with httpx.Client(timeout=60, follow_redirects=False) as client:
            for start in range(0, len(texts), 32):
                batch = texts[start:start + 32]
                response = client.post(self.base.rstrip('/') + '/embeddings',
                                       headers={'Authorization': f'Bearer {self.key}'},
                                       json={'model': self.model, 'input': batch})
                response.raise_for_status()
                rows = sorted(response.json()['data'], key=lambda x: x['index'])
                if [row['index'] for row in rows] != list(range(len(batch))):
                    raise ValueError('Embedding response has missing/duplicate indices')
                vectors.extend(row['embedding'] for row in rows)
        return vectors
