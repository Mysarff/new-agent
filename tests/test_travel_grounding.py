import asyncio
import json
import unittest

from SmartVoyage.engine import RunState, Engine
from SmartVoyage.grounding import needs_place_review, review_place_answer
from tests.test_travel_engine import ScriptedModel, HubFixture, answer, configuration, invoke

EVIDENCE = [{'id':'Wplaces','source':'https://provider.test/docs','data':{'places':[{'name':'测试馆','address':'某路1号','type':'博物馆'}]}}]


class GroundingTests(unittest.IsolatedAsyncioTestCase):
    async def test_review_receives_actual_fields_and_replaces_unsupported_draft(self):
        model = ScriptedModel([answer('地图候选：测试馆，某路1号 [Wplaces]。室内条件和开放情况尚待核实。')], auto_context=False)
        state = RunState()
        result = await review_place_answer(model,'雨天去哪','全国最大、全室内 [Wplaces]',EVIDENCE,state,1)
        self.assertNotIn('全国最大',result)
        payload=json.loads(model.inputs[0][0][-1]['content'])
        self.assertEqual(payload['evidence'],EVIDENCE)
        self.assertIn('全室内',payload['draft'])
        self.assertEqual(model.inputs[0][1],[])
        self.assertEqual(state.trace[-1]['status'],'completed')

    async def test_bad_or_unavailable_review_never_leaks_original_draft(self):
        for response in (answer('最大 [Winvented]'), answer('没有引用'), answer(''), invoke('unexpected',{})):
            with self.subTest(response=response):
                state=RunState()
                result=await review_place_answer(ScriptedModel([response],auto_context=False),'问题','未经支持的草稿',EVIDENCE,state,1)
                self.assertNotIn('未经支持的草稿',result)
                self.assertIn('未完成',result)
                self.assertEqual(state.trace[-1]['status'],'incomplete')

    async def test_review_timeout_is_bounded_and_retains_evidence(self):
        class Slow:
            async def complete(self,*args):
                await asyncio.sleep(5)
        state=RunState(evidence={item['id']:item for item in EVIDENCE})
        result=await review_place_answer(Slow(),'问题','不可靠',EVIDENCE,state,0.01)
        self.assertIn('未完成',result)
        self.assertEqual(list(state.evidence),['Wplaces'])

    async def test_engine_reviews_final_coordinator_answer_after_place_lookup(self):
        hub=HubFixture({'status':'success','evidence':EVIDENCE})
        hub.catalog.append(hub.tool('travel__search_places',{'query':{'type':'string'},'city':{'type':'string'}},['query','city']))
        model=ScriptedModel([invoke('delegate__itinerary',{'task':'找场所'}),
            invoke('travel__search_places',{'query':'博物馆','city':'某城'}),answer('测试馆 [Wplaces]'),
            answer('全国最大 [Wplaces]'),answer('地图返回测试馆，某路1号 [Wplaces]。')])
        result=await Engine(model,hub,configuration()).run('雨天去哪')
        self.assertEqual(result['answer'],'地图返回测试馆，某路1号 [Wplaces]。')
        self.assertEqual(result['answer_review'],'completed')
        self.assertEqual(result['citations'][0]['id'],'Wplaces')

    async def test_review_does_not_escape_whole_run_deadline(self):
        class SlowReview(ScriptedModel):
            async def complete(self,messages,tools,tool_choice='auto'):
                if not tools:
                    await asyncio.sleep(5)
                return await super().complete(messages,tools,tool_choice)
        hub=HubFixture({'status':'success','evidence':EVIDENCE})
        hub.catalog.append(hub.tool('travel__search_places',{'query':{'type':'string'},'city':{'type':'string'}},['query','city']))
        model=SlowReview([invoke('delegate__itinerary',{'task':'找场所'}),
            invoke('travel__search_places',{'query':'博物馆','city':'某城'}),answer('候选 [Wplaces]'),answer('全国最大 [Wplaces]')])
        result=await Engine(model,hub,configuration(run_timeout=0.1)).run('雨天去哪')
        self.assertIn('超时',result['answer'])
        self.assertNotIn('全国最大',result['answer'])
        self.assertTrue(any(e.get('event')=='run_timeout' for e in result['trace']))
        self.assertEqual(result['retrieved_evidence'][0]['id'],'Wplaces')

    def test_review_is_selected_by_evidence_shape_not_destination_keywords(self):
        self.assertTrue(needs_place_review(EVIDENCE))
        self.assertFalse(needs_place_review([{'id':'Krule','text':'知识片段'}]))
