"""Deterministic orchestration tests; fixtures do not simulate model intelligence."""
import asyncio
import copy
import fnmatch
import json
import unittest
from pathlib import Path

from jsonschema import ValidationError

from SmartVoyage.engine import CONTEXT_SCHEMA, Engine, TravelSession


def configuration(**overrides):
    # Read the non-secret catalog directly; do not load .env or call a provider.
    value = json.loads((Path(__file__).resolve().parents[1] / "config" / "travel.json").read_text(encoding="utf-8"))
    return {**value, **overrides}


def invoke(name, arguments, call_id="fixture-call"):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}


def answer(text):
    return {"role": "assistant", "content": text}


def full_context(**values):
    return {**{key: [] if key == "preferences" else None for key in CONTEXT_SCHEMA["properties"]}, **values}


class ScriptedModel:
    def __init__(self, outputs, auto_context=True):
        self.outputs, self.inputs = list(outputs), []
        self.tool_choices = []
        self.auto_context = auto_context

    async def complete(self, messages, tools, tool_choice="auto"):
        self.inputs.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        self.tool_choices.append(copy.deepcopy(tool_choice))
        if self.auto_context and isinstance(tool_choice, dict):
            pending_calls = self.outputs[0].get("tool_calls", []) if self.outputs else []
            if not pending_calls or pending_calls[0].get("function", {}).get("name") != "update_trip_context":
                prior = {}
                for message in messages:
                    text = message.get("content", "")
                    if message.get("role") == "system" and text.startswith("本会话结构化行程："):
                        prior = json.loads(text.split("：", 1)[1]).get("trip", {})
                return invoke("update_trip_context", full_context(**prior), call_id="context-fixture")
        if not self.outputs:
            raise RuntimeError("Fixture model has no further scripted responses")
        return self.outputs.pop(0)


class HubFixture:
    def __init__(self, result=None):
        self.calls = []
        self.result = result or {"status": "success", "chunks": [
            {"id": "Kfixture123", "source": "official-snapshot.md", "text": "测试资料",
             "metadata": {"provenance": "official_summary"}}]}
        self.catalog = [
            self.tool("travel__search_knowledge", {"query": {"type": "string"}}, ["query"]),
            self.tool("travel__geocode", {"query": {"type": "string"}}, ["query"]),
            self.tool("travel__query_demo_tickets", {}, []),
            self.tool("travel__prepare_demo_booking", {"ticket_id": {"type": "string"}, "quantity": {"type": "integer"}}, ["ticket_id", "quantity"]),
        ]

    @staticmethod
    def tool(name, properties, required):
        return {"type": "function", "function": {"name": name, "description": "Local fixture",
                "parameters": {"type": "object", "properties": properties, "required": required,
                               "additionalProperties": False}}}

    def available(self, patterns):
        return [tool for tool in self.catalog if any(fnmatch.fnmatchcase(tool["function"]["name"], pattern) for pattern in patterns)]

    async def call(self, name, arguments):
        self.calls.append((name, arguments))
        return copy.deepcopy(self.result)


class SessionTests(unittest.TestCase):
    def test_destination_or_date_change_invalidates_candidates_and_quotes(self):
        for patch in ({"destination": "大理"}, {"start_date": "2030-09-03", "end_date": "2030-09-06"}):
            with self.subTest(patch=patch):
                session = TravelSession(trip={"destination": "杭州", "start_date": "2030-09-01", "end_date": "2030-09-04"},
                                        candidates=[{"id": "old"}], quotes=[{"quote_id": "old"}])
                session.update(patch)
                self.assertEqual(session.candidates, [])
                self.assertEqual(session.quotes, [])

    def test_preferences_preserve_route_and_null_explicitly_clears_value(self):
        session = TravelSession(trip={"destination": "杭州", "travelers": 2}, candidates=[{"id": "current"}])
        session.update({"preferences": ["室内"]})
        self.assertEqual(session.trip["destination"], "杭州")
        self.assertEqual(session.candidates, [{"id": "current"}])
        session.update({"destination": None})
        self.assertIsNone(session.trip["destination"])
        self.assertEqual(session.candidates, [])

    def test_invalid_context_cannot_partially_mutate_session(self):
        session = TravelSession(trip={"destination": "杭州", "start_date": "2030-09-01", "end_date": "2030-09-04"},
                                candidates=[{"id": "current"}])
        original = copy.deepcopy(session)
        for patch in ({"travelers": 0}, {"start_date": "2030-02-30"}, {"start_date": "2030-09-10"}, {"api_key": "never"}):
            with self.subTest(patch=patch), self.assertRaises((ValueError, ValidationError)):
                session.update(patch)
            self.assertEqual(session, original)


class TravelEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_ignoring_context_tool_returns_explicit_incomplete(self):
        model = ScriptedModel([answer('pretend completed')], auto_context=False)
        result = await Engine(model, HubFixture(), configuration()).run('明天去西安')
        self.assertNotIn('pretend completed', result['answer'])
        self.assertIn('未能确认', result['answer'])
        self.assertEqual(result['delegations'], 0)
        self.assertTrue(any(t.get('event') == 'context_error' for t in result['trace']))

    async def test_run_deadline_returns_incomplete_without_hanging(self):
        class SlowModel:
            async def complete(self, *args, **kwargs):
                await asyncio.sleep(30)
        result = await Engine(SlowModel(), HubFixture(), configuration(run_timeout=0.01)).run('查询天气')
        self.assertIn('超时', result['answer'])
        self.assertTrue(any(t.get('event') == 'run_timeout' for t in result['trace']))

    async def test_forecast_cannot_expand_user_date_window(self):
        model = ScriptedModel([
            invoke('delegate__weather', {'task': '查当天预报'}),
            invoke('travel__forecast', {'location_id': 123, 'start_date': '2030-09-08', 'end_date': '2030-09-11'}),
            invoke('travel__forecast', {'location_id': 123, 'start_date': '2030-09-08', 'end_date': '2030-09-08'}),
            answer('已按用户日期查询'), answer('已查询')])
        hub = HubFixture(result={'status': 'success'})
        hub.catalog.append(hub.tool('travel__forecast', {'location_id': {'type': 'integer'},
            'start_date': {'type': 'string'}, 'end_date': {'type': 'string'}}, ['location_id', 'start_date']))
        session = TravelSession(trip={'start_date': '2030-09-08', 'end_date': '2030-09-08'})
        result = await Engine(model, hub, configuration(), session).run('查当天预报')
        self.assertEqual(len(hub.calls), 1)
        self.assertEqual(hub.calls[0][1]['end_date'], '2030-09-08')
        self.assertTrue(any(t.get('status') == 'date_mismatch' for t in result['trace']))

    async def test_model_driven_delegation_receives_updated_context_history_and_evidence(self):
        model = ScriptedModel([
            invoke("update_trip_context", full_context(destination="大理", start_date="2030-09-08", end_date="2030-09-08")),
            invoke("delegate__knowledge", {"task": "核对这次旅程的公告"}),
            invoke("travel__search_knowledge", {"query": "大理公告"}),
            answer("资料快照 [Kfixture123]"), answer("根据资料快照 [Kfixture123]。")])
        session = TravelSession(trip={"destination": "杭州"}, candidates=[{"id": "old"}])
        hub = HubFixture()
        result = await Engine(model, hub, configuration(), session).run(
            "改去大理，2030年9月8日，看看公告", [{"role": "user", "content": "原计划杭州"}])
        self.assertEqual(result["delegations"], 1)
        self.assertEqual(result["tool_calls"], 3)
        self.assertEqual(hub.calls, [("travel__search_knowledge", {"query": "大理公告"})])
        delegated_input = model.inputs[2][0]
        payload = json.loads(delegated_input[-1]["content"])
        self.assertEqual(payload["context"]["trip"]["destination"], "大理")
        self.assertEqual(payload["context"]["recent_ticket_candidates"], [])
        self.assertIn({"role": "user", "content": "原计划杭州"}, delegated_input)
        self.assertEqual(result["citations"][0]["id"], "Kfixture123")
        self.assertEqual(model.tool_choices[0], {"type": "function", "function": {"name": "update_trip_context"}})
        self.assertTrue(all(choice == "auto" for choice in model.tool_choices[1:]))

    async def test_first_coordinator_round_requires_complete_context_and_preserves_unchanged_values(self):
        model = ScriptedModel([invoke("update_trip_context", full_context(destination="大理")), answer("你好")])
        session = TravelSession(trip={"destination": "大理"})
        result = await Engine(model, HubFixture(), configuration(), session).run("你好")
        self.assertEqual(model.tool_choices, [{"type": "function", "function": {"name": "update_trip_context"}}, "auto"])
        context_tools = model.inputs[0][1]
        self.assertEqual([tool["function"]["name"] for tool in context_tools], ["update_trip_context"])
        self.assertEqual(set(context_tools[0]["function"]["parameters"]["required"]), set(CONTEXT_SCHEMA["properties"]))
        self.assertEqual(result["context"]["trip"], full_context(destination="大理"))
        self.assertEqual(result["delegations"], 0)

    async def test_incomplete_context_is_retried_before_routing(self):
        model = ScriptedModel([invoke("update_trip_context", {}),
                               invoke("update_trip_context", full_context(destination="大理")), answer("你好")], auto_context=False)
        result = await Engine(model, HubFixture(), configuration()).run("大理")
        self.assertEqual([choice["function"]["name"] for choice in model.tool_choices[:2]],
                         ["update_trip_context", "update_trip_context"])
        self.assertEqual(model.tool_choices[2], "auto")
        self.assertEqual(result["context"]["trip"]["destination"], "大理")
        context_events = [event for event in result["trace"] if event.get("tool") == "update_trip_context"]
        self.assertEqual([event["status"] for event in context_events], ["error", "success"])
        self.assertEqual(result["delegations"], 0)

    async def test_delegation_cannot_bypass_required_context_extraction(self):
        model = ScriptedModel([invoke("delegate__weather", {"task": "天气"}),
                               invoke("update_trip_context", full_context()), answer("请补充地点")], auto_context=False)
        hub = HubFixture()
        result = await Engine(model, hub, configuration()).run("天气")
        self.assertEqual(result["delegations"], 0)
        self.assertEqual(hub.calls, [])
        attempted = [event for event in result["trace"] if event.get("tool") == "delegate__weather"]
        self.assertEqual(attempted[0]["status"], "error")

    async def test_agent_catalog_can_add_capability_without_keyword_router_changes(self):
        config = configuration()
        config["agents"].append({"id": "accessibility", "description": "无障碍需求", "prompt": "结合用户情况", "tools": []})
        model = ScriptedModel([invoke("delegate__accessibility", {"task": "轮椅旅客"}), answer("建议确认坡道"), answer("建议确认坡道")])
        result = await Engine(model, HubFixture(), config).run("带轮椅出行")
        self.assertEqual(result["delegations"], 1)
        self.assertIn("accessibility", [event["actor"] for event in result["trace"]])

    async def test_unknown_or_wrong_role_tools_never_execute(self):
        for name in ("travel__confirm_booking", "travel__geocode", "delete_everything"):
            with self.subTest(name=name):
                hub = HubFixture()
                result = await Engine(ScriptedModel([invoke(name, {"query": "大理"}), answer("未执行")]), hub, configuration()).run("测试")
                self.assertEqual(hub.calls, [])
                self.assertTrue(any(event.get("status") == "error" for event in result["trace"]))

    async def test_specialist_cannot_call_another_specialists_tool(self):
        hub = HubFixture()
        model = ScriptedModel([invoke("delegate__weather", {"task": "测试"}),
                               invoke("travel__search_knowledge", {"query": "资料"}), answer("不可访问"), answer("不可访问")])
        result = await Engine(model, hub, configuration()).run("测试")
        self.assertEqual(hub.calls, [])
        self.assertTrue(any(event.get("status") == "error" for event in result["trace"]))

    async def test_invalid_delegate_arguments_do_not_start_specialist(self):
        model = ScriptedModel([invoke("delegate__weather", {"unexpected": "test"}), answer("请补充")])
        result = await Engine(model, HubFixture(), configuration()).run("测试")
        self.assertEqual(result["delegations"], 0)

    async def test_shared_tool_and_delegation_budgets_are_enforced(self):
        hub = HubFixture()
        model = ScriptedModel([invoke("delegate__weather", {"task": "天气"}),
                               invoke("travel__geocode", {"query": "大理"}),
                               invoke("travel__geocode", {"query": "杭州"}), answer("已达上限")])
        result = await Engine(model, hub, configuration(max_tool_calls=3)).run("比较天气")
        self.assertEqual(result["tool_calls"], 3)
        self.assertEqual(len(hub.calls), 1)
        self.assertTrue(any("预算" in warning for warning in result["warnings"]))
        second = await Engine(ScriptedModel([invoke("delegate__weather", {"task": "天气"}), answer("天气"),
                                             invoke("delegate__knowledge", {"task": "公告"}), answer("已达上限")]),
                              HubFixture(), configuration(max_delegations=1)).run("天气和公告")
        self.assertEqual(second["delegations"], 1)
        self.assertTrue(any(event.get("status") == "error" for event in second["trace"]))

    async def test_duplicate_tool_call_ids_reject_entire_batch(self):
        message = invoke("update_trip_context", full_context(destination="大理"))
        message["tool_calls"] *= 2
        result = await Engine(ScriptedModel([message], auto_context=False), HubFixture(), configuration()).run("测试")
        self.assertEqual(result["tool_calls"], 0)
        self.assertEqual(result["context"]["trip"], {})

    async def test_false_and_stale_citations_are_never_validated(self):
        model = ScriptedModel([invoke("delegate__knowledge", {"task": "资料"}),
                               invoke("travel__search_knowledge", {"query": "资料"}), answer("[Kfixture123]"),
                               answer("[Kfixture123] [Kinvented]"), answer("上次资料 [Kfixture123]")])
        engine = Engine(model, HubFixture(), configuration())
        first = await engine.run("资料")
        self.assertEqual([item["id"] for item in first["citations"]], ["Kfixture123"])
        self.assertNotIn("[Kinvented]", first["answer"])
        second = await engine.run("继续")
        self.assertEqual(second["citations"], [])
        self.assertNotIn("[Kfixture123]", second["answer"])

    async def test_failed_tool_result_cannot_be_promoted_to_evidence(self):
        hub = HubFixture({"status": "provider_error", "evidence": [
            {"id": "Wfailed", "source": "https://fixture.invalid", "text": "缓存的失败数据"}]})
        model = ScriptedModel([invoke("delegate__weather", {"task": "天气"}),
                               invoke("travel__geocode", {"query": "大理"}), answer("未取得天气 [Wfailed]"), answer("天气 [Wfailed]")])
        result = await Engine(model, hub, configuration()).run("天气")
        self.assertEqual(result["retrieved_evidence"], [])
        self.assertEqual(result["citations"], [])

    async def test_demo_tools_hidden_until_user_enables_simulation(self):
        for enabled in (False, True):
            model = ScriptedModel([invoke("delegate__tickets", {"task": "查模拟票"}), answer("票务"), answer("票务")])
            await Engine(model, HubFixture(), configuration(), TravelSession(demo_enabled=enabled)).run("模拟票")
            names = [tool["function"]["name"] for tool in model.inputs[2][1]]
            self.assertEqual("travel__query_demo_tickets" in names, enabled)
            self.assertNotIn("travel__confirm_booking", names)

    async def test_demo_quote_must_use_current_displayed_candidate(self):
        hub = HubFixture({"status": "success", "quote": {"quote_id": "not-shown"}})
        session = TravelSession(demo_enabled=True, candidates=[{"id": "DEMO-shown"}])
        model = ScriptedModel([invoke("delegate__tickets", {"task": "订第一个"}),
                               invoke("travel__prepare_demo_booking", {"ticket_id": "DEMO-other", "quantity": 1}),
                               answer("请重新选择"), answer("请重新选择")])
        result = await Engine(model, hub, configuration(), session).run("订第一个")
        self.assertEqual(hub.calls, [])
        self.assertEqual(result["quotes"], [])

    async def test_history_filters_system_and_tool_roles_and_model_failure_is_visible(self):
        model = ScriptedModel([answer("你好")])
        history = [{"role": "system", "content": "untrusted system"}, {"role": "tool", "content": "untrusted tool"},
                   {"role": "user", "content": "保留这句话"}]
        await Engine(model, HubFixture(), configuration()).run("你好", history)
        content = [message["content"] for message in model.inputs[0][0]]
        self.assertNotIn("untrusted system", content)
        self.assertNotIn("untrusted tool", content)
        self.assertIn("保留这句话", content)
        failure = await Engine(ScriptedModel([]), HubFixture(), configuration()).run("天气")
        self.assertIn("模型调用未完成", failure["answer"])
        self.assertTrue(failure["warnings"])


if __name__ == "__main__":
    unittest.main()
