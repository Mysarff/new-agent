import copy
from datetime import datetime, timedelta, timezone
import unittest

import httpx

from wayloom.weather import DAILY_VARIABLES, forecast, geocode


# Unit-test fixtures, deliberately isolated from production and never a fallback.
PLACE = {"id": 1790630, "name": "西安", "latitude": 34.25833, "longitude": 108.92861,
         "timezone": "Asia/Shanghai", "country_code": "CN", "country": "中国", "admin1": "陕西"}
NOW = datetime(2026, 9, 25, 16, 30, tzinfo=timezone.utc)  # Sep 26 in Xi'an.


def daily_payload(start="2026-09-26", end="2026-09-26"):
    start_day = datetime.fromisoformat(start).date()
    count = (datetime.fromisoformat(end).date() - start_day).days + 1
    dates = [(start_day + timedelta(days=i)).isoformat() for i in range(count)]
    units = dict(zip(DAILY_VARIABLES, ["wmo code", "°C", "°C", "%", "mm", "mm", "km/h"]))
    daily = {"time": dates, **{key: [value] * count for key, value in zip(DAILY_VARIABLES, [3, 27, 18, 20, 0.1, 0.1, 10])}}
    return {"daily": daily, "daily_units": units, "timezone": "Asia/Shanghai"}


class WeatherTests(unittest.IsolatedAsyncioTestCase):
    async def test_ambiguous_places_are_not_silently_selected(self):
        other = {**PLACE, "id": 1790631, "admin1": "湖南", "latitude": 28.46}
        def handler(request):
            self.assertEqual(request.url.params["name"], "西安")
            self.assertEqual(request.url.params["countryCode"], "CN")
            return httpx.Response(200, json={"results": [PLACE, other]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await geocode(" 西安 ", "cn", client=client)
            self.assertFalse(client.is_closed)
        self.assertEqual(result["status"], "needs_input")
        self.assertEqual(len(result["candidates"]), 2)
        self.assertNotIn("location", result)

    async def test_unique_result_and_unknown_location(self):
        for response, expected in [({"results": [PLACE]}, "success"), ({}, "no_data")]:
            with self.subTest(expected=expected):
                async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=response))) as client:
                    result = await geocode("unlisted-city-name", client=client)
                self.assertEqual(result["status"], expected)
                if expected == "success":
                    self.assertEqual(result["location"]["location_id"], PLACE["id"])
                    self.assertEqual(result["evidence"][0]["metadata"]["provenance"], "live_api")

    async def test_empty_query_and_bad_country_need_input_without_network(self):
        def handler(request):
            self.fail("Invalid input should not invoke an API")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            for query, country in [("", None), ("A", None), ("北京", "China")]:
                self.assertEqual((await geocode(query, country, client=client))["status"], "needs_input")

    async def test_forecast_uses_provider_coordinates_timezone_and_units(self):
        requests = []
        def handler(request):
            requests.append(request)
            if request.url.path == "/v1/get":
                self.assertEqual(request.url.params["id"], str(PLACE["id"]))
                return httpx.Response(200, json=PLACE)
            self.assertEqual(request.url.params["latitude"], str(PLACE["latitude"]))
            self.assertEqual(request.url.params["longitude"], str(PLACE["longitude"]))
            self.assertEqual(request.url.params["timezone"], "Asia/Shanghai")
            return httpx.Response(200, json=daily_payload())
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await forecast(PLACE["id"], "2026-09-26", client=client, now=NOW)
        self.assertEqual(len(requests), 2)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["daily"][0]["date"], "2026-09-26")
        self.assertEqual(result["units"]["precipitation_sum"], "mm")
        self.assertTrue(result["forecast_not_observation"])
        self.assertTrue(result["evidence"][0]["source"].startswith("https://api.open-meteo.com/"))
        self.assertIn("retrieved_at", result)

    async def test_destination_local_date_and_sixteen_day_boundary(self):
        for day, expected in [("2026-09-25", "out_of_range"), ("2026-09-26", "success"),
                              ("2026-10-11", "success"), ("2026-10-12", "out_of_range")]:
            with self.subTest(day=day):
                requests = []
                def handler(request):
                    requests.append(request)
                    if request.url.path == "/v1/get":
                        return httpx.Response(200, json=PLACE)
                    return httpx.Response(200, json=daily_payload(day, day))
                async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                    result = await forecast(PLACE["id"], day, client=client, now=NOW)
                self.assertEqual(result["status"], expected)
                self.assertEqual(result["available_range"]["end_date"], "2026-10-11")
                self.assertEqual(len(requests), 2 if expected == "success" else 1)
                if expected == "out_of_range":
                    self.assertNotIn("daily", result)
                    self.assertNotIn("evidence", result)

    async def test_bad_dates_and_model_supplied_coordinates_rejected(self):
        def handler(request):
            self.fail("Invalid input should not invoke an API")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            for location_id, start, end in [(True, "2026-09-26", None), ("34.25,108.92", "2026-09-26", None),
                                             (PLACE["id"], "2026-02-30", None), (PLACE["id"], "20260926", None),
                                             (PLACE["id"], "2026-09-27", "2026-09-26")]:
                result = await forecast(location_id, start, end, client=client, now=NOW)
                self.assertEqual(result["status"], "needs_input")

    async def test_api_failure_is_not_promoted_to_weather(self):
        for code in (400, 429, 503):
            with self.subTest(code=code):
                async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(code, json={"error": True}))) as client:
                    result = await geocode("成都", client=client)
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["http_status"], code)
                self.assertNotIn("evidence", result)
                self.assertNotIn("daily", result)
        def timeout(request):
            raise httpx.ReadTimeout("test network outage", request=request)
        async with httpx.AsyncClient(transport=httpx.MockTransport(timeout)) as client:
            result = await forecast(PLACE["id"], "2026-09-26", client=client, now=NOW)
        self.assertEqual(result["status"], "error")
        self.assertTrue(result["retryable"])

    async def test_unknown_location_id_and_mismatched_identity(self):
        for code, payload, expected in [(404, {}, "no_data"), (200, {**PLACE, "id": 123}, "error")]:
            with self.subTest(expected=expected):
                async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(code, json=payload))) as client:
                    result = await forecast(PLACE["id"], "2026-09-26", client=client, now=NOW)
                self.assertEqual(result["status"], expected)

    async def test_missing_values_remain_null_and_all_missing_is_no_data(self):
        for all_missing in (False, True):
            with self.subTest(all_missing=all_missing):
                payload = daily_payload()
                for key in (DAILY_VARIABLES if all_missing else ("precipitation_probability_max",)):
                    payload["daily"][key] = [None]
                def handler(request):
                    return httpx.Response(200, json=PLACE if request.url.path == "/v1/get" else payload)
                async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                    result = await forecast(PLACE["id"], "2026-09-26", client=client, now=NOW)
                self.assertEqual(result["status"], "no_data" if all_missing else "success")
                if not all_missing:
                    self.assertIsNone(result["daily"][0]["precipitation_probability_max"])
                    self.assertTrue(result["warnings"])

    async def test_wmo_description_distinguishes_drizzle_fog_and_missing_weather(self):
        payload = daily_payload("2026-09-26", "2026-10-01")
        payload["daily"]["weather_code"] = [51, 45, 48, 0, None, 999]

        def handler(request):
            if request.url.path == "/v1/get":
                return httpx.Response(200, json=PLACE)
            self.assertNotIn("weather_description", request.url.params["daily"])
            return httpx.Response(200, json=payload)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await forecast(PLACE["id"], "2026-09-26", "2026-10-01", client=client, now=NOW)
        self.assertEqual(result["status"], "success")
        descriptions = [row["weather_description"] for row in result["daily"]]
        self.assertIn("毛毛雨", descriptions[0])
        self.assertNotIn("雾", descriptions[0])
        self.assertEqual(descriptions[1], "雾")
        self.assertIn("雾凇", descriptions[2])
        self.assertEqual(descriptions[3], "晴空")
        self.assertIn("暂无", descriptions[4])
        self.assertIn("未知", descriptions[5])
        self.assertNotIn("晴", descriptions[4] + descriptions[5])
        self.assertIsNone(result["daily"][4]["weather_code"])
        self.assertEqual(result["evidence"][0]["data"]["daily"], result["daily"])

    async def test_malformed_or_different_date_series_rejected(self):
        mutations = [lambda p: p["daily"].update(time=["2026-09-27"]),
                     lambda p: p["daily"].update(precipitation_sum=[]),
                     lambda p: p["daily_units"].pop("precipitation_sum"),
                     lambda p: p["daily"].update(precipitation_sum=["no rain"])]
        for mutation in mutations:
            payload = copy.deepcopy(daily_payload())
            mutation(payload)
            def handler(request):
                return httpx.Response(200, json=PLACE if request.url.path == "/v1/get" else payload)
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await forecast(PLACE["id"], "2026-09-26", client=client, now=NOW)
            self.assertEqual(result["status"], "error")
            self.assertNotIn("evidence", result)


if __name__ == "__main__":
    unittest.main()
