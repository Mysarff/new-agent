"""SmartVoyage's python-a2a 0.5.4 Task envelope over real HTTP JSON-RPC."""
import argparse
import asyncio
import json
import uuid

import httpx
from fastapi import FastAPI, Request
from python_a2a import AgentCard, AgentSkill, Message, MessageRole, Task, TaskState, TaskStatus, TextContent

from .config import load_config
from .engine import Engine, RunState, TravelSession
from .mcp_client import ToolHub
from .model import ChatModel


def endpoint(agent):
    return agent.get('endpoint') or f"http://127.0.0.1:{agent['port']}"


class RemoteAgents:
    def __init__(self, config):
        self.config = config

    async def call(self, agent, query, history, context, evidence, budget):
        if budget < 1:
            raise ValueError('No remaining tool budget')
        task_id = str(uuid.uuid4())
        content = {'query': query, 'history': history, 'context': context, 'evidence': evidence, 'budget': budget,
                   'compact_handoffs': self.config.get('compact_handoffs', False)}
        if 'enable_thinking' in self.config:
            content['enable_thinking'] = self.config['enable_thinking']
        task = Task(id=task_id, message=Message(role=MessageRole.USER,
                    content=TextContent(text=json.dumps(content, ensure_ascii=False))).to_dict())
        async with httpx.AsyncClient(timeout=self.config['a2a_timeout'], follow_redirects=False) as client:
            card = await client.get(endpoint(agent) + '/.well-known/agent.json')
            card.raise_for_status()
            if card.json().get('name') != agent['id']:
                raise ValueError('Unexpected AgentCard')
            response = await client.post(endpoint(agent) + '/tasks/send', json={
                'jsonrpc': '2.0', 'id': task_id, 'method': 'tasks/send', 'params': task.to_dict()})
            response.raise_for_status()
            raw = response.json()
        if raw.get('id') != task_id or 'error' in raw:
            raise ValueError('A2A task failed')
        returned = Task.from_dict(raw['result'])
        if returned.id != task_id or getattr(returned.status.state, 'value', returned.status.state) != 'completed':
            raise ValueError('A2A task not complete')
        result = returned.metadata['business']
        if not 0 <= result.get('tool_calls', 0) <= budget:
            raise ValueError('A2A budget mismatch')
        result['trace'].insert(0, {'actor': agent['id'], 'event': 'A2A tasks/send', 'status': 'success'})
        return result


def create_app(agent_id, model=None):
    config = load_config()
    agent = next(a for a in config['agents'] if a['id'] == agent_id)
    app = FastAPI(title=agent['name'])

    @app.get('/health')
    def health():
        return {'status': 'ok', 'agent': agent_id}

    @app.get('/.well-known/agent.json')
    def card():
        return AgentCard(name=agent_id, description=agent['description'], url=endpoint(agent), version='1.0',
                          skills=[AgentSkill(id=agent_id, name=agent['name'], description=agent['description'])]).to_dict()

    @app.post('/tasks/send')
    async def send(request: Request):
        body = {}
        try:
            body = await request.json()
            if not isinstance(body, dict) or body.get('jsonrpc') != '2.0' or body.get('method') != 'tasks/send':
                raise ValueError('Unsupported method')
            task = Task.from_dict(body['params'])
            if body.get('id') != task.id:
                raise ValueError('Task ID mismatch')
            payload = json.loads(task.message['content']['text'])
            if not 1 <= len(payload['query']) <= 6000 or not 1 <= payload['budget'] <= config['max_tool_calls']:
                raise ValueError('Invalid task')
            context = payload.get('context', {})
            compact = payload.get('compact_handoffs', config.get('compact_handoffs', False))
            if type(compact) is not bool:
                raise ValueError('compact_handoffs must be boolean')
            if 'enable_thinking' in payload and payload['enable_thinking'] is not None and type(payload['enable_thinking']) is not bool:
                raise ValueError('enable_thinking must be null or boolean')
            if type(context.get('demo_enabled', False)) is not bool:
                raise ValueError('demo_enabled must be boolean')
            history = payload.get('history', [])
            if (not isinstance(history, list) or len(history) > 14 or any(
                    not isinstance(m, dict) or m.get('role') not in ('user', 'assistant')
                    or not isinstance(m.get('content'), str) or len(m['content']) > 6000 for m in history)):
                raise ValueError('Invalid history')
            candidates = context.get('recent_ticket_candidates', [])
            if not isinstance(candidates, list) or any(not isinstance(c, dict) for c in candidates):
                raise ValueError('Invalid candidates')
            session = TravelSession(demo_enabled=context.get('demo_enabled', False))
            session.update(context.get('trip', {}))
            session.candidates = candidates
            state = RunState(evidence={item['id']: item for item in payload.get('evidence', [])})
            async with ToolHub(config) as hub:
                request_model = model or (ChatModel(enable_thinking=payload['enable_thinking'])
                    if 'enable_thinking' in payload else ChatModel())
                engine = Engine(request_model, hub, dict(config, max_tool_calls=payload['budget'], compact_handoffs=compact), session)
                answer = await asyncio.wait_for(engine.specialist(agent, payload['query'], history, state),
                                                config['a2a_timeout'] - 5)
            result = {'answer': answer, 'evidence': list(state.evidence.values()), 'trace': state.trace,
                      'warnings': state.warnings, 'tool_calls': state.calls, 'candidates': session.candidates,
                      'quotes': session.quotes}
            task.status = TaskStatus(state=TaskState.COMPLETED)
            task.artifacts = [{'parts': [{'type': 'text', 'text': answer}]}]
            task.metadata = {'business': result}
            return {'jsonrpc': '2.0', 'id': task.id, 'result': task.to_dict()}
        except Exception:
            return {'jsonrpc': '2.0', 'id': body.get('id') if isinstance(body, dict) else None,
                    'error': {'code': -32000, 'message': 'Agent 请求未完成'}}
    return app


if __name__ == '__main__':
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', required=True)
    args = parser.parse_args()
    config = load_config()
    agent = next(a for a in config['agents'] if a['id'] == args.agent)
    uvicorn.run(create_app(args.agent), host='127.0.0.1', port=agent['port'], log_level='warning')
