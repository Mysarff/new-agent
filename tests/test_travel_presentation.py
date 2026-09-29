import unittest

from wayloom.presentation import format_answer, readable_units, safe_source_url, source_catalog, trip_overview


class PresentationTests(unittest.TestCase):
    def test_screenshot_celsius_regression_preserves_numbers(self):
        original = r'气温在 \(18.1^\circ\text{C} \sim 20.6^\circ\text{C}\) 之间 [Wweather]。'
        self.assertEqual(readable_units(original), '气温在 18.1℃ ～ 20.6℃ 之间 [Wweather]。')
        self.assertEqual(readable_units(r'$-2.5^{\circ}\mathrm{C}$'), '-2.5℃')

    def test_unrelated_math_and_existing_celsius_unchanged(self):
        text = r'普通公式 $x^2$、\(a+b\)，气温18℃～21℃，概率80%。'
        self.assertEqual(readable_units(text), text)

    def test_citations_are_numbered_by_first_use_with_unique_turn_anchors(self):
        items = [{'id': 'Wgeo', 'title': '位置'}, {'id': 'Wweather', 'title': '天气'}, {'id': 'Krule', 'title': '规则'}]
        answer = '天气 [Wweather]，规则 [Krule]，再提天气 [Wweather]。'
        catalog = source_catalog(items, answer)
        self.assertEqual([item['id'] for item in catalog], ['Wweather','Krule','Wgeo'])
        first = format_answer(answer, catalog, 'reply-1')
        second = format_answer(answer, catalog, 'reply-3')
        self.assertEqual(first.count('[来源1](#source-reply-1-1)'), 2)
        self.assertIn('#source-reply-3-1', second)
        self.assertNotIn('Wweather', first)

    def test_unknown_citation_cannot_link_to_unrelated_source(self):
        self.assertEqual(format_answer('事实 [Winvented]', [], 'reply-1'), '事实 〔来源未核验〕')
        with self.assertRaises(ValueError):
            format_answer('text', [], 'x"><script>')

    def test_source_urls_reject_credentials_scripts_and_encode_markdown_delimiters(self):
        for value in ('javascript:alert(1)', 'https://user:password@site.test', 'file:///local', 'https://site.test/\nx'):
            self.assertIsNone(safe_source_url(value))
        self.assertEqual(safe_source_url('https://site.test/a(b)'), 'https://site.test/a%28b%29')

    def test_trip_single_day_empty_preferences_and_unknown_count_read_naturally(self):
        summary = dict(trip_overview({'destination':'西安','start_date':'2026-09-28','end_date':'2026-09-28','preferences':[]}))
        self.assertEqual(summary['出行日期'], '2026-09-28（当天）')
        self.assertEqual(summary['旅行偏好'], '未提供')
        self.assertEqual(summary['同行人数'], '未提供')
        self.assertNotIn('开始日期', summary)

    def test_multi_day_and_named_preferences_are_preserved(self):
        summary = dict(trip_overview({'start_date':'2026-09-28','end_date':'2026-09-30','travelers':2,'preferences':['安静','无障碍']}))
        self.assertEqual(summary['出行日期'], '2026-09-28 至 2026-09-30')
        self.assertEqual(summary['同行人数'], '2人')
        self.assertEqual(summary['旅行偏好'], '安静、无障碍')
