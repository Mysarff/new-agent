import json
import os
import unittest
from unittest.mock import patch

import httpx

from SmartVoyage.model import ChatModel, thinking_setting
from SmartVoyage.tools import ISODate
from pydantic import TypeAdapter, ValidationError


class ModelOptionsTests(unittest.IsolatedAsyncioTestCase):
    async def test_optional_thinking_and_usage_are_transmitted_without_reasoning_text(self):
        sent=[]
        def handler(request):
            sent.append(json.loads(request.content))
            return httpx.Response(200,json={'choices':[{'message':{'role':'assistant','content':'回答',
                'reasoning_content':'private reasoning'}}], 'usage':{'prompt_tokens':10,'completion_tokens':3,
                'total_tokens':13,'completion_tokens_details':{'reasoning_tokens':1}}})
        client=httpx.AsyncClient
        with patch.dict(os.environ,{'SMARTVOYAGE_MODEL':'test-model','SMARTVOYAGE_API_KEY':'local-test',
            'SMARTVOYAGE_BASE_URL':'https://model.test/v1','SMARTVOYAGE_ENABLE_THINKING':'false'}), \
            patch('SmartVoyage.model.httpx.AsyncClient',side_effect=lambda **kw:client(transport=httpx.MockTransport(handler),**kw)):
            response=await ChatModel().complete([{'role':'user','content':'天气'}],[])
            await ChatModel(enable_thinking=None).complete([{'role':'user','content':'天气'}],[])
            await ChatModel(enable_thinking=True).complete([{'role':'user','content':'天气'}],[])
        self.assertIs(sent[0]['enable_thinking'],False)
        self.assertNotIn('enable_thinking',sent[1])
        self.assertIs(sent[2]['enable_thinking'],True)
        self.assertEqual(response['_usage']['reasoning_tokens'],1)
        self.assertNotIn('reasoning_content',response)

    def test_bad_thinking_setting_fails_visibly(self):
        for value in (0,1,{},'sometimes'):
            with self.assertRaises(ValueError):
                thinking_setting(value)
        self.assertIsNone(thinking_setting(''))
        self.assertIs(thinking_setting('false'),False)

    def test_knowledge_optional_date_schema_rejects_string_none(self):
        adapter=TypeAdapter(ISODate | None)
        self.assertIsNone(adapter.validate_python(None))
        self.assertEqual(adapter.validate_python('2030-09-08'),'2030-09-08')
        with self.assertRaises(ValidationError):
            adapter.validate_python('None')
