"""Provider-neutral Chat Completions and embeddings adapters; no fake model fallback."""
import os

from .config import env_value

import httpx

USE_ENV = object()


def thinking_setting(value):
    if value is None or value == '':
        return None
    if type(value) is bool:
        return value
    if isinstance(value, str) and value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    raise ValueError('WAYLOOM_ENABLE_THINKING must be empty, true or false')


class ChatModel:
    def __init__(self, enable_thinking=USE_ENV):
        self.model = env_value('WAYLOOM_MODEL', '')
        self.key = env_value('WAYLOOM_API_KEY', '')
        self.base = env_value('WAYLOOM_BASE_URL', '').rstrip('/')
        self.enable_thinking = thinking_setting(env_value('WAYLOOM_ENABLE_THINKING', '')
                                                if enable_thinking is USE_ENV else enable_thinking)
        if not self.model or not self.key or not self.base:
            raise ValueError('请在 .env 配置 WAYLOOM_MODEL、WAYLOOM_BASE_URL 和 WAYLOOM_API_KEY。')

    async def complete(self, messages, tools, tool_choice='auto'):
        payload = {'model': self.model, 'messages': messages}
        if self.enable_thinking is not None:
            payload['enable_thinking'] = self.enable_thinking
        if tools:
            payload.update(tools=tools, tool_choice=tool_choice)
        async with httpx.AsyncClient(timeout=90, follow_redirects=False) as client:
            response = await client.post(self.base + '/chat/completions', json=payload,
                                         headers={'Authorization': f'Bearer {self.key}'})
            response.raise_for_status()
            data = response.json()
        message = data['choices'][0]['message']
        result = {k: v for k, v in message.items() if k in ('role', 'content', 'tool_calls')}
        result['_usage'] = data.get('usage') if isinstance(data.get('usage'), dict) else {}
        details = result['_usage'].get('completion_tokens_details')
        if isinstance(details, dict) and type(details.get('reasoning_tokens')) is int:
            result['_usage']['reasoning_tokens'] = details['reasoning_tokens']
        return result


class Embeddings:
    def __init__(self):
        self.model = env_value('WAYLOOM_EMBEDDING_MODEL', '')
        self.base = env_value('WAYLOOM_EMBEDDING_BASE_URL') or env_value('WAYLOOM_BASE_URL', '')
        self.key = env_value('WAYLOOM_EMBEDDING_API_KEY') or env_value('WAYLOOM_API_KEY', '')

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
