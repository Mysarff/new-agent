"""Execution contracts and negative controls, not simulated model accuracy."""
import copy
import unittest

from wayloom.engine import Engine, TravelSession
from wayloom.planning import compact_result, forecast_summary, overconfident_weather
from tests.test_travel_engine import ScriptedModel, configuration, full_context, invoke
from tests.test_travel_engine import HubFixture, answer


FORECAST = {'id': 'Wforecast', 'source': 'https://weather.test/forecast',
    'metadata': {'kind': 'weather_forecast'}, 'data': {'location': {'label': '测试城市'}, 'daily': [
        {'date': '2030-09-08', 'weather_description': '阴天', 'temperature_2m_min': 12.6,
         'temperature_2m_max': 23.2, 'precipitation_probability_max': 0,
         'precipitation_sum': 0.0, 'wind_speed_10m_max': 8.8}]}}
PLACES = {'id': 'Wplaces', 'source': 'https://places.test/search', 'data': {'places': [
    {'name': '测试馆', 'address': '测试街1号', 'type': '博物馆'}]}}
KNOWLEDGE = {'id': 'Krule', 'source': 'https://rules.test/', 'text': '测试规则',
             'metadata': {'checked_at': '2030-09-01'}}


def task(identifier='t1', agent='weather', depends=None, condition=None):
    return dict(id=identifier, task='此任务的用户需求', agent=agent, depends_on=depends or [], condition=condition)


def section(identifier='t1', answer='当前预报及建议', ids=None, status='answered'):
    return dict(task_id=identifier, answer=answer, evidence_ids=ids or [], status=status)


def plan(tasks):
    return invoke('plan_trip', dict(context=full_context(destination='测试城市', start_date='2030-09-08', end_date='2030-09-08'), tasks=tasks))


def final(sections, missing=None):
    return invoke('submit_answer', dict(sections=sections, missing_requests=missing or []))


class Remote:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    async def call(self, agent, query, history, context, evidence, budget, **kwargs):
        self.calls.append((agent['id'], query, copy.deepcopy(context), copy.deepcopy(evidence), budget))
        if self.fail and agent['id'] == 'weather':
            raise ConnectionError('private-provider-host')
        item = FORECAST if agent['id'] == 'weather' else KNOWLEDGE if agent['id'] == 'knowledge' else PLACES
        return dict(answer='测试交接 [' + item['id'] + ']', evidence=[copy.deepcopy(item)], warnings=[],
                    trace=[{'actor': agent['id'], 'event': 'tool', 'tool': 'fixture_lookup', 'status': 'success'}], tool_calls=1)


class PlannedExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_specialist_cannot_skip_required_lookup_due_to_dry_forecast(self):
        from wayloom.engine import RunState
        config = configuration(require_task_evidence=True)
        hub = HubFixture({'status': 'success', 'evidence': [PLACES]})
        hub.catalog.append(hub.tool('travel__search_places', {'city': {'type': 'string'}, 'query': {'type': 'string'}}, ['city', 'query']))
        model = ScriptedModel([answer('没有雨，无需安排备选'),
            invoke('travel__search_places', {'city': '测试城市', 'query': '博物馆'}), answer('雨天地点候选 [Wplaces]')])
        engine = Engine(model, hub, config)
        state = RunState(evidence={'Wforecast': copy.deepcopy(FORECAST)})
        result = await engine.specialist(engine.agents['itinerary'], '准备雨天备选', [], state)
        self.assertEqual(len(hub.calls), 1)
        self.assertIn('雨天地点候选', result)
        self.assertTrue(any(t.get('event') == 'evidence_required' for t in state.trace))

    async def run_case(self, outputs, remote=None, **config):
        remote = remote or Remote()
        model = ScriptedModel(outputs, auto_context=False)
        result = await Engine(model, None, configuration(planned_execution=True, compact_handoffs=True, **config), remote=remote).run('查天气；如果下雨也给出备选')
        return result, model, remote

    async def test_conditional_task_always_runs_and_forecast_numbers_are_not_model_invented(self):
        tasks = [task(), task('t2', 'itinerary', ['t1'], '即使目前无雨，也给出雨天备选')]
        result, model, remote = await self.run_case([plan(tasks), final([
            section(answer='错误温度 99℃', ids=['Wforecast']),
            section('t2', '若下雨，可核对测试馆的室内条件和开放情况。', ['Wplaces'])])])
        self.assertEqual([row[0] for row in remote.calls], ['weather', 'itinerary'])
        self.assertEqual(remote.calls[1][3][0]['id'], 'Wforecast')
        self.assertIn('若下雨', result['answer'])
        self.assertIn('备选方案（即使目前无雨，也给出雨天备选）', result['answer'])
        self.assertIn('12.6℃～23.2℃', result['answer'])
        self.assertNotIn('99℃', result['answer'])
        self.assertIn('0% 也不保证无雨', result['answer'])
        self.assertEqual(result['answer_review'], 'completed')
        self.assertEqual(len(model.inputs), 2)  # plan + final, excluding remote workers

    async def test_invalid_plan_cannot_execute_or_change_session(self):
        for tasks in ([task(depends=['t2']), task('t2', 'itinerary')],
                      [task(), task()], [task(agent='unknown')]):
            result, _, remote = await self.run_case([plan(tasks), plan(tasks)])
            self.assertEqual(remote.calls, [])
            self.assertEqual(result['context']['trip'], {})
            self.assertIn('未能确认', result['answer'])

    async def test_missing_section_is_repaired_without_repeating_tool_queries(self):
        tasks = [task(), task('t2', 'itinerary', ['t1'])]
        result, model, remote = await self.run_case([plan(tasks), final([section(ids=['Wforecast'])]),
            final([section(ids=['Wforecast']), section('t2', '雨天备选', ['Wplaces'])])])
        self.assertEqual(len(remote.calls), 2)
        self.assertEqual(len(model.inputs), 3)
        self.assertIn('雨天备选', result['answer'])
        self.assertTrue(any(row.get('status') == 'retry' for row in result['trace']))

    async def test_final_review_can_recover_a_task_omitted_from_plan(self):
        result, _, remote = await self.run_case([plan([task()]), final([section(ids=['Wforecast'])],
            [{'task': '用户还要求雨天备选，需查场所', 'agent': 'itinerary'}]), final([
                section(ids=['Wforecast']), section('t2', '下雨时的备选', ['Wplaces'])])])
        self.assertEqual([row[0] for row in remote.calls], ['weather', 'itinerary'])
        self.assertIn('下雨时的备选', result['answer'])
        self.assertTrue(any(row.get('event') == 'coverage_repair' for row in result['trace']))

    async def test_overconfident_advice_cannot_escape_a_failed_repair(self):
        bad = final([section(answer='明天不会下雨，无需防雨准备', ids=['Wforecast'])])
        result, _, remote = await self.run_case([plan([task()]), bad, bad])
        self.assertNotIn('不会下雨', result['answer'])
        self.assertIn('尚未通过', result['answer'])
        self.assertEqual(result['answer_review'], 'incomplete')
        self.assertEqual(len(remote.calls), 1)
        self.assertTrue(result['retrieved_evidence'])
        self.assertIn('12.6℃～23.2℃', result['answer'])
        self.assertIn('本轮尚未全部完成', result['answer'])

    async def test_prepared_contingency_cannot_be_labelled_not_triggered(self):
        tasks = [task(), task('t2', 'itinerary', ['t1'], '假设下雨，预先准备')]
        bad = final([section(ids=['Wforecast']), section('t2', '雨天备选暂不触发', ['Wplaces'], 'unavailable')])
        good = final([section(ids=['Wforecast']), section('t2', '若下雨，可核对测试馆是否开放后前往。', ['Wplaces'])])
        result, _, remote = await self.run_case([plan(tasks), bad, good])
        self.assertNotIn('暂不触发', result['answer'])
        self.assertIn('若下雨', result['answer'])
        self.assertEqual(len(remote.calls), 2)

    async def test_failed_agent_cannot_be_marked_answered(self):
        result, _, _ = await self.run_case([plan([task()]), final([section(answer='伪造已完成')])], Remote(fail=True))
        self.assertNotIn('伪造已完成', result['answer'])
        self.assertIn('未完成', result['answer'])
        self.assertNotIn('private-provider-host', str(result))

    async def test_unknown_sources_fail_and_task_budget_remains_bounded(self):
        bad = final([section(ids=['Winvented'])])
        result, _, _ = await self.run_case([plan([task()]), bad, bad])
        self.assertNotIn('[Winvented]', result['answer'])
        tasks = [task(), task('t2', 'itinerary')]
        result, _, remote = await self.run_case([plan(tasks), final([
            section(ids=['Wforecast']), section('t2', '声称完成')])], max_delegations=1)
        self.assertEqual(len(remote.calls), 1)
        self.assertNotIn('声称完成', result['answer'])

    async def test_knowledge_checked_date_survives_composition(self):
        result, _, _ = await self.run_case([plan([task(agent='knowledge')]), final([
            section(answer='规则摘要', ids=['Krule'])])])
        self.assertIn('资料核验记录：2030-09-01', result['answer'])

    async def test_failed_existing_task_is_not_redelegated_as_missing_request(self):
        result, _, remote = await self.run_case([plan([task()]), final([
            section(answer='专家连接未完成', status='unavailable')],
            [{'task': task()['task'], 'agent': 'weather'}])], Remote(fail=True))
        self.assertEqual(len(remote.calls), 1)
        self.assertIn('未完成', result['answer'])

    async def test_plan_accepts_lossless_numeric_string_without_extra_model_call(self):
        initial = invoke('plan_trip', dict(context=full_context(travelers='2'), tasks=[task(agent=None)]))
        result, model, _ = await self.run_case([initial, final([section(answer='请补充地点')])])
        self.assertEqual(result['context']['trip']['travelers'], 2)
        self.assertEqual(len(model.inputs), 2)

    async def test_other_task_evidence_cannot_substitute_for_places_citation(self):
        tasks = [task(), task('t2', 'itinerary')]
        bad = final([section(ids=['Wforecast']), section('t2', '地点候选', ['Wforecast'])])
        result, _, _ = await self.run_case([plan(tasks), bad, bad])
        self.assertEqual(result['answer_review'], 'incomplete')

    def test_all_forecasts_in_multi_city_task_must_be_retained(self):
        from wayloom.engine import RunState
        from wayloom.planning import validate_answer
        other = copy.deepcopy(FORECAST)
        other['id'] = 'Wother'
        state = RunState(evidence={'Wforecast': FORECAST, 'Wother': other}, handoffs=[{
            'task_id': 't1', 'execution_status': 'success', 'evidence_ids': ['Wforecast', 'Wother']}])
        with self.assertRaises(ValueError):
            validate_answer({'sections': [section(ids=['Wforecast'])]}, [task()], state)
        validate_answer({'sections': [section(ids=['Wforecast', 'Wother'])]}, [task()], state)

    def test_compaction_removes_only_duplicate_data_and_preserves_source_metadata(self):
        result = dict(status='success', daily=FORECAST['data']['daily'], evidence=[copy.deepcopy(FORECAST)], notice='请复核')
        original = copy.deepcopy(result)
        compact = compact_result(result)
        self.assertNotIn('daily', compact)
        self.assertEqual(compact['evidence'], original['evidence'])
        self.assertEqual(compact['notice'], '请复核')
        self.assertEqual(result, original)

    def test_weather_guard_allows_negated_claims_and_unknown_values(self):
        for safe in ('不能保证不会下雨', '0%也不保证无雨', '建议备好雨具'):
            self.assertFalse(overconfident_weather(safe))
        for bad in ('无需防雨准备', '不需要带伞', '肯定不会下雨', '保证无雨'):
            self.assertTrue(overconfident_weather(bad))
        evidence = copy.deepcopy(FORECAST)
        evidence['data']['daily'][0]['precipitation_probability_max'] = None
        self.assertIn('暂无', forecast_summary([evidence]))

    def test_schema_feedback_identifies_constraint_without_leaking_values(self):
        from jsonschema import ValidationError, validate
        from wayloom.planning import validation_issue
        try:
            validate({'status': 'private-value'}, {'properties': {'status': {'enum': ['answered']}}})
        except ValidationError as exc:
            issue = validation_issue(exc)
        self.assertIn('status', issue)
        self.assertIn('enum', issue)
        self.assertNotIn('private-value', issue)
