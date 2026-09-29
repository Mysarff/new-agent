import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from wayloom.tickets import DemoBookingStore, query_tickets


QUERY = {"kind": "train", "departure": "景德镇", "arrival": "伊宁", "date": "2031-04-23"}


def provider_ticket(**overrides):
    return {"id": "supplier-123", **QUERY, "service": "供应方班次", "seat": "二等座",
            "price": 123.45, "currency": "CNY", "remaining": 3, "source": "测试供应方",
            "booking_url": "https://supplier.example/buy/123", **overrides}


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_not_configured_is_unavailable_without_fake_prices(self):
        with patch.dict("os.environ", {"WAYLOOM_TICKET_BASE_URL": ""}):
            result = await query_tickets(**QUERY)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["tickets"], [])
        self.assertNotIn("quote", result)

    async def test_parameterized_new_route_and_bearer_read_only_request(self):
        requests = []
        def handler(request):
            requests.append(request)
            self.assertEqual(dict(request.url.params), QUERY)
            self.assertEqual(request.method, "GET")
            self.assertEqual(request.headers["Authorization"], "Bearer test-secret")
            return httpx.Response(200, json={"tickets": [provider_ticket()]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch.dict("os.environ", {"WAYLOOM_TICKET_BASE_URL": "https://supplier.example/api/v1/",
                                          "WAYLOOM_TICKET_API_KEY": "test-secret"}):
                result = await query_tickets(**QUERY, client=client)
        self.assertEqual(str(requests[0].url).split("?")[0], "https://supplier.example/api/v1/tickets")
        self.assertEqual(result["tickets"][0]["source_kind"], "live")
        self.assertEqual(result["tickets"][0]["price"], 123.45)
        self.assertEqual(result["tickets"][0]["booking_url"], "https://supplier.example/buy/123")
        self.assertNotIn("order_id", result)
        self.assertNotIn("test-secret", json.dumps(result))
        self.assertEqual(len(requests), 1)

    async def test_empty_provider_result_does_not_seed_simulation(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"tickets": []}))) as client:
            result = await query_tickets(**QUERY, client=client, base_url="https://supplier.example", api_key="")
        self.assertEqual(result["status"], "no_data")
        self.assertEqual(result["tickets"], [])

    async def test_invalid_parameters_fail_before_network(self):
        with self.assertRaises(ValueError):
            await query_tickets(**{**QUERY, "date": "2031-02-30"}, base_url="")
        with self.assertRaises(ValueError):
            await query_tickets(**{**QUERY, "departure": ""}, base_url="")
        with self.assertRaises(ValueError):
            await query_tickets(**QUERY, base_url="http://supplier.example")

    async def test_schema_and_query_mismatch_are_rejected(self):
        malformed = [
            {"tickets": [provider_ticket(price=-1)]},
            {"tickets": [provider_ticket(price=True)]},
            {"tickets": [provider_ticket(remaining=-1)]},
            {"tickets": [provider_ticket(currency="人民币")]},
            {"tickets": [provider_ticket(arrival="别的城市")]},
            {"tickets": [provider_ticket(booking_url="javascript:alert(1)")]},
            {"tickets": [provider_ticket(), provider_ticket()]},
            {"tickets": [{key: value for key, value in provider_ticket().items() if key != "remaining"}]},
            {"tickets": {}},
            {"data": [provider_ticket()]},
        ]
        for payload in malformed:
            with self.subTest(payload=payload):
                async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))) as client:
                    result = await query_tickets(**QUERY, client=client, base_url="https://supplier.example")
                self.assertEqual(result["error_code"], "invalid_response")
                self.assertEqual(result["tickets"], [])

    async def test_unknown_inventory_is_explicit_null(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"tickets": [provider_ticket(remaining=None)]}))) as client:
            result = await query_tickets(**QUERY, client=client, base_url="https://supplier.example")
        self.assertIsNone(result["tickets"][0]["remaining"])

    async def test_provider_failure_and_timeout_are_explicit(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(401, text="private upstream body"))) as client:
            result = await query_tickets(**QUERY, client=client, base_url="https://supplier.example")
        self.assertEqual(result["http_status"], 401)
        self.assertNotIn("private upstream body", json.dumps(result))
        def timeout_handler(request):
            raise httpx.ReadTimeout("sensitive transport details", request=request)
        async with httpx.AsyncClient(transport=httpx.MockTransport(timeout_handler)) as client:
            result = await query_tickets(**QUERY, client=client, base_url="https://supplier.example")
        self.assertEqual(result["error_code"], "timeout")
        self.assertNotIn("sensitive transport details", json.dumps(result))

    async def test_redirects_are_not_followed(self):
        seen = []
        def handler(request):
            seen.append(request)
            return httpx.Response(302, headers={"Location": "https://another.example/tickets"})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
            result = await query_tickets(**QUERY, client=client, base_url="https://supplier.example", api_key="secret")
        self.assertEqual(len(seen), 1)
        self.assertEqual(result["http_status"], 302)


class DemoBookingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = DemoBookingStore(Path(self.temp.name) / "demo.sqlite")

    def seed(self):
        self.store.seed_demo(QUERY["departure"], QUERY["arrival"], QUERY["date"])
        return self.store.query(**QUERY)["tickets"][0]

    def prepare(self, quantity=2):
        ticket = self.seed()
        return ticket, self.store.prepare(ticket["id"], quantity)["quote"]

    def test_store_starts_empty_and_new_cities_dates_are_not_restricted(self):
        self.assertEqual(self.store.query(**QUERY)["status"], "no_data")
        self.seed()
        result = self.store.query(**QUERY)
        self.assertTrue(result["tickets"])
        self.assertEqual(result["source_kind"], "synthetic")
        for ticket in result["tickets"]:
            self.assertEqual(ticket["date"], QUERY["date"])
            self.assertEqual(ticket["departure"], QUERY["departure"])
            self.assertEqual(ticket["source_kind"], "synthetic")
            self.assertTrue(ticket["id"].startswith("DEMO-"))
        unusual = {**QUERY, "departure": "O'Hare; DROP TABLE demo_tickets;--", "date": "2034-12-31"}
        self.store.seed_demo(unusual["departure"], unusual["arrival"], unusual["date"])
        self.assertTrue(self.store.query(**unusual)["tickets"])
        self.assertTrue(self.store.query(**QUERY)["tickets"])

    def test_prepare_does_not_deduct_inventory_or_create_order(self):
        ticket, quote = self.prepare()
        self.assertEqual(self.store.query(**QUERY)["tickets"][0]["remaining"], ticket["remaining"])
        with self.store.connection() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM demo_orders").fetchone()[0], 0)
        self.assertEqual(quote["amount"], ticket["price"] * 2)
        self.assertEqual(quote["source_kind"], "synthetic")

    def test_confirmation_replay_and_reseed_never_double_deduct(self):
        ticket, quote = self.prepare()
        first = self.store.confirm(quote["quote_id"], "click-001")
        second = self.store.confirm(quote["quote_id"], "click-001")
        third = self.store.confirm(quote["quote_id"], "click-002")
        self.assertFalse(first["replayed"])
        self.assertTrue(second["replayed"])
        self.assertTrue(third["replayed"])
        self.assertEqual(first["order_id"], third["order_id"])
        self.seed()
        self.assertEqual(self.store.query(**QUERY)["tickets"][0]["remaining"], ticket["remaining"] - 2)
        with self.store.connection() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM demo_orders").fetchone()[0], 1)

    def test_request_id_cannot_be_reused_for_another_quote(self):
        ticket, first = self.prepare()
        second = self.store.prepare(ticket["id"], 1)["quote"]
        self.store.confirm(first["quote_id"], "same-click")
        with self.assertRaises(ValueError):
            self.store.confirm(second["quote_id"], "same-click")

    def test_replayed_quote_binds_new_request_id_to_original_quote(self):
        ticket, first = self.prepare()
        second = self.store.prepare(ticket["id"], 1)["quote"]
        self.store.confirm(first["quote_id"], "original-click")
        self.store.confirm(first["quote_id"], "retry-click")
        with self.assertRaises(ValueError):
            self.store.confirm(second["quote_id"], "retry-click")

    def test_price_currency_and_itinerary_changes_require_new_confirmation(self):
        ticket = self.seed()
        for field, value in (("price_cents", 88888), ("currency", "USD"), ("arrival", "另一目的地")):
            quote = self.store.prepare(ticket["id"], 1)["quote"]
            with self.store.connection(write=True) as db:
                # Field names here are fixed test constants; production uses fixed parameterized SQL.
                db.execute(f"UPDATE demo_tickets SET {field}=? WHERE id=?", (value, ticket["id"]))
            result = self.store.confirm(quote["quote_id"], "change-" + field)
            self.assertEqual(result["status"], "changed")
        with self.store.connection() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM demo_orders").fetchone()[0], 0)

    def test_inventory_is_rechecked_at_confirm(self):
        ticket, quote = self.prepare(quantity=3)
        with self.store.connection(write=True) as db:
            db.execute("UPDATE demo_tickets SET remaining=2 WHERE id=?", (ticket["id"],))
        self.assertEqual(self.store.confirm(quote["quote_id"], "stock-change")["status"], "no_data")
        self.assertEqual(self.store.query(**QUERY)["tickets"][0]["remaining"], 2)

    def test_quote_expires_at_five_minutes_without_order(self):
        ticket = self.seed()
        with patch("wayloom.tickets.time.time", return_value=1000):
            quote = self.store.prepare(ticket["id"], 1)["quote"]
        self.assertEqual(quote["expires_at"], 1300)
        with patch("wayloom.tickets.time.time", return_value=1300):
            result = self.store.confirm(quote["quote_id"], "late-click")
        self.assertEqual(result["status"], "expired")

    def test_configured_quantity_limit_and_invalid_quantities(self):
        self.store = DemoBookingStore(Path(self.temp.name) / "custom.sqlite", max_quantity=2)
        ticket = self.seed()
        for quantity in (True, 0, -1, 3, 1.2, "1"):
            with self.assertRaises(ValueError):
                self.store.prepare(ticket["id"], quantity)
        self.assertEqual(self.store.prepare(ticket["id"], 2)["status"], "success")

    def test_competing_quotes_cannot_oversell(self):
        ticket = self.seed()
        with self.store.connection(write=True) as db:
            db.execute("UPDATE demo_tickets SET remaining=3 WHERE id=?", (ticket["id"],))
        first = self.store.prepare(ticket["id"], 2)["quote"]
        second = self.store.prepare(ticket["id"], 2)["quote"]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda args: self.store.confirm(*args),
                                       [(first["quote_id"], "race-a"), (second["quote_id"], "race-b")]))
        self.assertEqual(sorted(result["status"] for result in results), ["no_data", "success"])
        self.assertEqual(self.store.query(**QUERY)["tickets"][0]["remaining"], 1)


if __name__ == "__main__":
    unittest.main()
