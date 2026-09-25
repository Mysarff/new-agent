import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from opsatlas.config import ROOT


class UiTests(unittest.TestCase):
    def test_retrieval_and_clear_conversation(self):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
        self.assertFalse(app.exception)
        app.chat_input[0].set_value('数据库连接池耗尽').run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.chat_message), 2)
        self.assertTrue(any('候选原文' in item.value for item in app.markdown))
        app.button[0].click().run()
        self.assertEqual(len(app.chat_message), 0)
