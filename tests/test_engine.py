import json
import unittest

from opsatlas.config import load_config
from opsatlas.engine import Engine


def invoke(name, arguments):
    return {'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': 'call1', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments)}}]}


class ScriptedModel:
    """Protocol fixture only; not shipped as an application model or intelligence demo."""
    def __init__(self, messages):
        self.messages, self.inputs = iter(messages), []
    async def complete(self, messages, tools):
        self.inputs.append((list(messages), tools))
        return next(self.messages)


class HubFixture:
    def __init__(self):
        self.calls = []
    def available(self, patterns):
        if 'knowledge__*' not in patterns:
            return []
        return [{'type': 'function', 'function': {'name': 'knowledge__search_knowledge', 'description': 'Search',
                'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}}}]
    async def call(self, name, args):
        self.calls.append((name, args))
        return {'status': 'candidates', 'chunks': [{'id': 'K123', 'source': 'db.md', 'text': '连接池需要检查',
                'metadata': {'provenance': 'synthetic'}}]}


class EngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_delegate_retrieve_cite_and_history(self):
        model = ScriptedModel([invoke('delegate__runbook', {'task': '查连接池'}),
                               invoke('knowledge__search_knowledge', {'query': '连接池'}),
                               {'role': 'assistant', 'content': '模拟资料 [K123]'},
                               {'role': 'assistant', 'content': '请检查连接池。模拟资料 [K123]'}])
        hub = HubFixture()
        result = await Engine(model, hub, load_config()).run('接着查', [{'role': 'user', 'content': '数据库超时'}])
        self.assertEqual(len(result['citations']), 1)
        self.assertEqual(result['tool_calls'], 2)
        self.assertEqual(hub.calls[0][1], {'query': '连接池'})
        self.assertTrue(any('虚构' in w for w in result['warnings']))
        self.assertTrue(any(m.get('content') == '数据库超时' for m in model.inputs[1][0]))

    async def test_unknown_tool_never_executes(self):
        hub = HubFixture()
        model = ScriptedModel([invoke('delete_everything', {}), {'role': 'assistant', 'content': '不支持'}])
        result = await Engine(model, hub, load_config()).run('test')
        self.assertEqual(hub.calls, [])
        self.assertEqual(result['trace'][1]['status'], 'error')

    async def test_invalid_arguments_do_not_delegate(self):
        model = ScriptedModel([invoke('delegate__runbook', {'other': 'x'}), {'role': 'assistant', 'content': '请补充'}])
        result = await Engine(model, HubFixture(), load_config()).run('test')
        self.assertEqual(result['delegations'], 0)

    async def test_tool_permissions(self):
        hub = HubFixture()
        model = ScriptedModel([invoke('delegate__researcher', {'task': 'test'}),
                               invoke('knowledge__search_knowledge', {'query': 'secret'}),
                               {'role': 'assistant', 'content': '不可访问'}, {'role': 'assistant', 'content': '不可访问'}])
        await Engine(model, hub, load_config()).run('test')
        self.assertEqual(hub.calls, [])

    async def test_invalid_citation_and_no_evidence(self):
        result = await Engine(ScriptedModel([{'role': 'assistant', 'content': '有依据 [Kfake]'}]),
                              HubFixture(), load_config()).run('test')
        self.assertEqual(result['citations'], [])
        self.assertNotIn('[Kfake]', result['answer'])
        self.assertTrue(result['warnings'])

    async def test_call_budget_stops_repeated_calls(self):
        model = ScriptedModel([invoke('delegate__runbook', {'task': 'test'})] * 20)
        config = dict(load_config(), max_tool_calls=2)
        result = await Engine(model, HubFixture(), config).run('test')
        self.assertEqual(result['tool_calls'], 2)
        self.assertIn('预算', result['answer'])

    async def test_configuration_can_add_agent_without_router_changes(self):
        config = load_config()
        config['agents'].append({'id': 'new_specialist', 'description': 'New domain', 'prompt': 'Test', 'tools': []})
        model = ScriptedModel([invoke('delegate__new_specialist', {'task': 'hello'}),
                               {'role': 'assistant', 'content': 'new'}, {'role': 'assistant', 'content': 'done'}])
        result = await Engine(model, HubFixture(), config).run('test')
        self.assertEqual(result['delegations'], 1)
        self.assertIn('new_specialist', [e['actor'] for e in result['trace']])

    async def test_model_failure_is_visible(self):
        result = await Engine(ScriptedModel([]), HubFixture(), load_config()).run('test')
        self.assertIn('模型调用未完成', result['answer'])

    async def test_evidence_is_per_run(self):
        model = ScriptedModel([{'role': 'assistant', 'content': 'hi'}, {'role': 'assistant', 'content': '[K123]'}])
        engine = Engine(model, HubFixture(), load_config())
        await engine.run('first')
        second = await engine.run('second')
        self.assertFalse(second['citations'])
