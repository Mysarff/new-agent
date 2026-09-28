import copy
import json
import time
import unittest
from datetime import date

from SmartVoyage.benchmark import DEFAULT_CASES, resolve_dates, score_execution, summarize
from SmartVoyage.engine import Engine
from SmartVoyage.metrics import model_event, summarize_trace
from tests.test_travel_engine import HubFixture, ScriptedModel, answer, configuration, invoke
from tests.test_travel_grounding import EVIDENCE


class BenchmarkTests(unittest.TestCase):
    def test_airport_does_not_pass_city_forecast_check(self):
        case = {'agents':['weather'], 'weather':{'names':['西安'], 'country':'CN', 'date':'2030-01-01'}}
        item = {'id':'Wforecast', 'metadata':{'kind':'weather_forecast'}, 'data':{
            'location':{'name':'西安西关机场','country_code':'CN'}, 'daily':[{'date':'2030-01-01'}]}}
        result = {'answer':'天气 [Wforecast]', 'retrieved_evidence':[item], 'trace':[
            {'event':'tool','tool':'delegate__weather','status':'success'}]}
        self.assertFalse(score_execution(case,result)['cited_weather_matches_target'])
        item['data']['location']['name']='西安'
        self.assertTrue(score_execution(case,result)['cited_weather_matches_target'])

    def test_failures_stay_in_latency_and_success_denominators(self):
        records = [dict(case_id='a', repeat=1, profile='baseline', status='passed', elapsed_seconds=10,
                        checks={'routing_exact':True},metrics={'model_calls':3,'total_tokens':100}),
                   dict(case_id='a', repeat=1, profile='compact', status='failed', elapsed_seconds=240,
                        checks={'routing_exact':True},metrics={'model_calls':4},trace=[{'event':'run_timeout'}])]
        result=summarize(records)
        self.assertEqual(result['profiles']['compact']['latency_p95_seconds'],240)
        self.assertEqual(result['profiles']['compact']['timeout_rate'],1)
        self.assertIsNone(result['profiles']['compact']['mean_total_tokens'])
        self.assertIsNone(result['profiles']['baseline']['semantic_accuracy'])
        self.assertEqual(result['paired_regressions'],['a'])
        self.assertEqual(result['paired_delta_mean_seconds'],230)

    def test_model_usage_is_removed_from_history_and_missing_is_not_zero(self):
        message={'content':'ok','_usage':{'prompt_tokens':5,'completion_tokens':2,'total_tokens':7,'secret':'never'}}
        event=model_event('weather','model',time.monotonic(),message)
        self.assertNotIn('_usage',message)
        self.assertNotIn('secret',event['usage'])
        summary=summarize_trace([event,{'actor':'coordinator','event':'tool','elapsed_ms':99999},
            {'actor':'weather','event':'tool','elapsed_ms':10}],100)
        self.assertEqual(summary['total_tokens'],7)
        self.assertEqual(summary['leaf_tool_elapsed_ms'],10)
        summary=summarize_trace([event,{'event':'model_error','elapsed_ms':30}],100)
        self.assertIsNone(summary['total_tokens'])
        self.assertEqual(summary['model_calls'],2)

    def test_manual_accuracy_requires_all_attempts_reviewed_and_keeps_failures(self):
        rows=[dict(case_id=str(i),repeat=1,profile='baseline',status=status,elapsed_seconds=1)
              for i,status in enumerate(('passed','failed'))]
        rows[0]['manual_review']={'status':'completed','reviewer':'human-1','answer_correct':True,
            'covers_user_request':True,'claim_count':2,'supported_claim_count':2}
        self.assertIsNone(summarize(rows)['profiles']['baseline']['semantic_accuracy'])
        rows[1]['manual_review']={'status':'completed','reviewer':'human-1','answer_correct':False,
            'covers_user_request':False,'claim_count':2,'supported_claim_count':1}
        summary=summarize(rows)['profiles']['baseline']
        self.assertEqual(summary['semantic_accuracy'],.5)
        self.assertEqual(summary['request_coverage_rate'],.5)
        self.assertEqual(summary['supported_claim_rate'],.75)

    def test_dataset_resolves_dates_and_keeps_original_immutable(self):
        dataset=json.loads(DEFAULT_CASES.read_text(encoding='utf-8'))
        original=copy.deepcopy(dataset)
        resolved=resolve_dates(dataset,date(2030,12,31))
        self.assertEqual(resolved['cases'][0]['context']['start_date'],'2031-01-01')
        self.assertEqual(dataset,original)
        self.assertEqual(len({case['id'] for case in dataset['cases']}),len(dataset['cases']))

    def test_unsupported_citation_and_round_limit_fail(self):
        checks=score_execution({'agents':[]}, {'answer':'成功 [Winvented]',
            'trace':[{'event':'round_limit'}]})
        self.assertFalse(checks['citation_ids_valid'])
        self.assertFalse(checks['no_incomplete_event'])


class CompactHandoffTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_evidence_review_catches_coordinator_bypassing_itinerary(self):
        hub=HubFixture({'status':'candidates','chunks':[{'id':'Krule','source':'https://example.org/rule','text':'规则事实'}]})
        model=ScriptedModel([invoke('delegate__knowledge',{'task':'查规则'}),
            invoke('travel__search_knowledge',{'query':'规则'}),answer('规则 [Krule]'),
            answer('未经查询的场所都能去 [Krule]'),answer('只能核实规则事实 [Krule]；尚未查询场所。')])
        result=await Engine(model,hub,configuration(review_all_evidence=True)).run('查规则和场所')
        self.assertEqual(result['answer_review'],'completed')
        self.assertNotIn('都能去',result['answer'])

    async def test_itinerary_without_lookup_still_requires_source_review(self):
        model=ScriptedModel([invoke('delegate__itinerary',{'task':'雨天安排'}),
            answer('未查过的场馆全室内'),answer('未查过的场馆全室内'),answer('没有查到场馆证据')])
        result=await Engine(model,self.hub(),configuration()).run('雨天安排')
        self.assertNotIn('未查过的场馆',result['answer'])
        self.assertEqual(result['answer_review'],'incomplete')

    async def test_query_failure_without_evidence_cannot_become_ungrounded_rules(self):
        hub=HubFixture({'status':'tool_error','message':'failed'})
        model=ScriptedModel([invoke('delegate__knowledge',{'task':'查规则'}),
            invoke('travel__search_knowledge',{'query':'规则'}),answer('不可靠规定'),answer('不可靠规定')])
        result=await Engine(model,hub,configuration()).run('查规则')
        self.assertNotIn('不可靠规定',result['answer'])
        self.assertIn('未取得可核验资料',result['answer'])

    def hub(self):
        hub=HubFixture({'status':'success','evidence':EVIDENCE})
        hub.catalog.append(hub.tool('travel__search_places',{'query':{'type':'string'},'city':{'type':'string'}},['query','city']))
        return hub

    async def test_compact_finalization_retains_evidence_review(self):
        model=ScriptedModel([invoke('delegate__itinerary',{'task':'查场馆'}),
            invoke('travel__search_places',{'query':'馆','city':'测试城'}),answer('测试馆 [Wplaces]'),
            invoke('finalize_answer',{}),answer('地图候选测试馆 [Wplaces]；室内条件未核实。')])
        result=await Engine(model,self.hub(),configuration(compact_handoffs=True)).run('找场馆')
        self.assertEqual(result['answer_review'],'completed')
        self.assertTrue(any(row['event']=='finalize_handoffs' for row in result['trace']))
        review_input=json.loads(model.inputs[-1][0][-1]['content'])
        self.assertEqual(review_input['tool_context']['handoffs'][0]['report'],'测试馆 [Wplaces]')
        self.assertEqual(review_input['evidence'],EVIDENCE)
        self.assertEqual(model.inputs[-1][1],[])
        self.assertEqual(result['metrics']['model_calls'],6)

    async def test_compact_cannot_finalize_before_evidence_or_skip_failed_review(self):
        model=ScriptedModel([invoke('finalize_answer',{}),answer('请补充地点')])
        result=await Engine(model,self.hub(),configuration(compact_handoffs=True)).run('去哪')
        self.assertFalse(any(row['event']=='finalize_handoffs' for row in result['trace']))
        model=ScriptedModel([invoke('delegate__itinerary',{'task':'查场馆'}),
            invoke('travel__search_places',{'query':'馆','city':'测试城'}),answer('不可靠草稿 [Wplaces]'),
            invoke('finalize_answer',{}),answer('无引用不合格')])
        result=await Engine(model,self.hub(),configuration(compact_handoffs=True)).run('去哪')
        self.assertEqual(result['answer_review'],'incomplete')
        self.assertNotIn('不可靠草稿',result['answer'])

    async def test_baseline_does_not_offer_finalization_tool(self):
        model=ScriptedModel([invoke('delegate__itinerary',{'task':'查场馆'}),
            invoke('travel__search_places',{'query':'馆','city':'测试城'}),answer('候选 [Wplaces]'),
            answer('最终草稿 [Wplaces]'),answer('来源复核后 [Wplaces]')])
        result=await Engine(model,self.hub(),configuration(compact_handoffs=False)).run('去哪')
        self.assertEqual(result['answer_review'],'completed')
        self.assertFalse(any(t['function']['name']=='finalize_answer' for _,tools in model.inputs for t in tools))
