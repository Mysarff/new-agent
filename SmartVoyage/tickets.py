"""Read-only ticket provider and an explicitly enabled, local booking simulation.

The provider never creates orders. ``DemoBookingStore.confirm`` belongs to the
human confirmation UI and must not be exposed as an agent/MCP tool.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import date as Date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlsplit

import httpx


DEMO_SOURCE = "本地模拟数据：线路、班次、价格及库存均为虚构，无支付、无真实出票"


def _text(value, field, max_length=160):
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise ValueError(f"{field}必须为非空文本，长度不超过{max_length}")
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"{field}不能包含控制字符")
    return value.strip()


def _query_values(kind, departure, arrival, date):
    kind = _text(kind, "票种", 40)
    departure = _text(departure, "出发地", 120)
    arrival = _text(arrival, "目的地", 120)
    date = _text(date, "日期", 10)
    try:
        if Date.fromisoformat(date).isoformat() != date:
            raise ValueError
    except ValueError:
        raise ValueError("日期必须是有效的 YYYY-MM-DD 格式") from None
    return {"kind": kind, "departure": departure, "arrival": arrival, "date": date}


def _provider_url(value):
    value = _text(value, "票务接口地址", 2048).rstrip("/")
    parts = urlsplit(value)
    if (parts.scheme not in ("http", "https") or not parts.hostname
            or parts.username or parts.password or parts.query or parts.fragment):
        raise ValueError("票务接口地址须为不含认证信息、查询参数或片段的 HTTP(S) 地址")
    if parts.scheme != "https" and parts.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("真实票务接口必须使用 HTTPS；仅本机调试允许 HTTP")
    return value + "/tickets"


def _validated_ticket(row, query):
    if not isinstance(row, dict):
        raise ValueError("票务记录必须为对象")
    ticket = {}
    for field in ("id", "kind", "departure", "arrival", "date", "service", "seat", "source"):
        ticket[field] = _text(row.get(field), field, 1000 if field == "source" else 160)
    if any(ticket[field] != query[field] for field in query):
        raise ValueError("供应方返回了与查询条件不符的记录")
    price = row.get("price")
    if type(price) not in (int, float) or not math.isfinite(price) or price < 0:
        raise ValueError("price须为有限非负数")
    remaining = row.get("remaining")
    if "remaining" not in row or (remaining is not None and (type(remaining) is not int or remaining < 0)):
        raise ValueError("remaining须为非负整数，未知库存用null")
    currency = row.get("currency")
    if not isinstance(currency, str) or not re.fullmatch(r"[A-Z]{3}", currency):
        raise ValueError("currency须为三位大写货币代码")
    ticket.update(price=price, remaining=remaining, currency=currency, source_kind="live")
    if row.get("booking_url"):
        booking_url = _text(row["booking_url"], "booking_url", 2048)
        parts = urlsplit(booking_url)
        if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
            raise ValueError("booking_url须为HTTPS供应方购票链接")
        ticket["booking_url"] = booking_url
    return ticket


async def query_tickets(kind, departure, arrival, date, client=None, *,
                        base_url=None, api_key=None, timeout=15.0):
    """GET a configured supplier; absent configuration never falls back to demo data.

    ``client`` can be an injected httpx.AsyncClient. Provider failures are returned
    as explicit statuses without echoing response bodies or authentication values.
    Invalid caller parameters raise ValueError before any network request.
    """
    params = _query_values(kind, departure, arrival, date)
    base_url = os.getenv("SMARTVOYAGE_TICKET_BASE_URL", "") if base_url is None else base_url
    api_key = os.getenv("SMARTVOYAGE_TICKET_API_KEY", "") if api_key is None else api_key
    if not base_url or not base_url.strip():
        return {"status": "unavailable", "source_kind": "unavailable", "tickets": [],
                "source": "票务供应方尚未配置", "query": params,
                "message": "尚未接入真实票务接口。请配置 SMARTVOYAGE_TICKET_BASE_URL；"
                           "如供应方需要认证，再配置 SMARTVOYAGE_TICKET_API_KEY。没有查询到真实票价或库存。"}
    endpoint = _provider_url(base_url)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 120:
        raise ValueError("票务请求超时须为0至120秒之间的正数")
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + _text(api_key, "票务API密钥", 8192)
    context = {"source": endpoint, "source_kind": "live", "query": params, "tickets": []}
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient()
    try:
        # Disable redirects so bearer credentials are not sent to a different endpoint.
        response = await client.get(endpoint, params=params, headers=headers,
                                    timeout=timeout, follow_redirects=False)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("tickets"), list):
            raise ValueError("响应必须包含tickets数组")
        if len(payload["tickets"]) > 500:
            raise ValueError("供应方单次记录超过500条")
        tickets = [_validated_ticket(row, params) for row in payload["tickets"]]
        if len({ticket["id"] for ticket in tickets}) != len(tickets):
            raise ValueError("供应方返回了重复票号")
        return {**context, "status": "success" if tickets else "no_data", "tickets": tickets,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "message": "供应方查询结果；价格和库存以供应方最终确认页面为准。本项目不会自动下单或出票。"}
    except httpx.TimeoutException:
        return {**context, "status": "provider_error", "error_code": "timeout",
                "message": "票务供应方请求超时，未取得可用票价或库存。"}
    except httpx.HTTPStatusError as exc:
        return {**context, "status": "provider_error", "error_code": "http_error",
                "http_status": exc.response.status_code,
                "message": "票务供应方拒绝或未完成查询，请检查接口权限和服务状态。"}
    except httpx.RequestError:
        return {**context, "status": "provider_error", "error_code": "network_error",
                "message": "无法连接票务供应方，未取得可用票价或库存。"}
    except (ValueError, TypeError, OverflowError):
        return {**context, "status": "provider_error", "error_code": "invalid_response",
                "message": "票务供应方返回的数据不符合接口约定，已拒绝使用；请检查字段、币种及查询条件。"}
    finally:
        if own_client:
            await client.aclose()


class DemoBookingStore:
    """Local synthetic data. Construction creates tables, never tickets or orders."""

    def __init__(self, path, max_quantity=5, quote_ttl_seconds=300):
        if type(max_quantity) is not int or not 1 <= max_quantity <= 1000:
            raise ValueError("max_quantity必须为1至1000的整数")
        if type(quote_ttl_seconds) is not int or not 1 <= quote_ttl_seconds <= 300:
            raise ValueError("模拟报价有效期必须为1至300秒")
        self.path = str(path)
        self.max_quantity = max_quantity
        self.quote_ttl_seconds = quote_ttl_seconds
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS demo_tickets (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, departure TEXT NOT NULL,
                    arrival TEXT NOT NULL, date TEXT NOT NULL, service TEXT NOT NULL,
                    seat TEXT NOT NULL, price_cents INTEGER NOT NULL CHECK(price_cents>=0),
                    currency TEXT NOT NULL, remaining INTEGER NOT NULL CHECK(remaining>=0));
                CREATE TABLE IF NOT EXISTS demo_quotes (
                    id TEXT PRIMARY KEY, expires_at REAL NOT NULL, payload TEXT NOT NULL,
                    used_order_id TEXT);
                CREATE TABLE IF NOT EXISTS demo_orders (
                    id TEXT PRIMARY KEY, request_id TEXT NOT NULL UNIQUE,
                    quote_id TEXT NOT NULL UNIQUE, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS demo_request_keys (
                    request_id TEXT PRIMARY KEY, quote_id TEXT NOT NULL, order_id TEXT NOT NULL);
                INSERT OR IGNORE INTO demo_request_keys(request_id,quote_id,order_id)
                    SELECT request_id,quote_id,id FROM demo_orders;
            """)

    @contextmanager
    def connection(self, write=False):
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            if write:
                db.commit()
        except Exception:
            if write:
                db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _ticket(row):
        ticket = dict(row)
        ticket["price"] = float(Decimal(ticket.pop("price_cents")) / 100)
        return {**ticket, "source": DEMO_SOURCE, "source_kind": "synthetic"}

    @staticmethod
    def _result(status, **values):
        return {"status": status, "source_kind": "synthetic", "source": DEMO_SOURCE, **values}

    def seed_demo(self, departure, arrival, date):
        """Call only after the user clicks to enable a simulation for this route.

        Fixed fake prices exercise the booking flow; cities/dates are user input.
        Repeat calls use stable IDs and never replenish already consumed inventory.
        """
        params = _query_values("train", departure, arrival, date)
        route = json.dumps([params["departure"], params["arrival"], params["date"]], ensure_ascii=False)
        route_id = hashlib.sha256(route.encode()).hexdigest()[:20].upper()
        templates = [
            ("train", "模拟列车-A", "模拟二等座", 19900, 12),
            ("train", "模拟列车-B", "模拟一等座", 29900, 6),
            ("flight", "模拟航班-A", "模拟经济舱", 59900, 8),
            ("attraction", "模拟景区入场", "模拟成人票", 9900, 20),
            ("concert", "模拟演出", "模拟看台", 19900, 10),
        ]
        with self.connection(write=True) as db:
            for index, (kind, service, seat, price_cents, remaining) in enumerate(templates):
                db.execute("INSERT OR IGNORE INTO demo_tickets VALUES (?,?,?,?,?,?,?,?,?,?)",
                           (f"DEMO-{route_id}-{index}", kind, params["departure"], params["arrival"],
                            params["date"], service, seat, price_cents, "CNY", remaining))
        return self._result("success", message="已为指定城市与日期开启本地模拟演练；所有票务数据均为虚构。",
                            departure=params["departure"], arrival=params["arrival"], date=params["date"])

    def query(self, kind, departure, arrival, date):
        params = _query_values(kind, departure, arrival, date)
        with self.connection() as db:
            rows = db.execute("SELECT * FROM demo_tickets WHERE kind=? AND departure=? AND arrival=? "
                              "AND date=? ORDER BY price_cents,id LIMIT 50",
                              (params["kind"], params["departure"], params["arrival"], params["date"])).fetchall()
        tickets = [self._ticket(row) for row in rows]
        return self._result("success" if tickets else "no_data", tickets=tickets, data=tickets, query=params,
                            message="仅查询已由用户开启的模拟数据；不代表真实票价、班次或库存。")

    def prepare(self, ticket_id, quantity):
        ticket_id = _text(ticket_id, "模拟票号")
        if type(quantity) is not int or not 1 <= quantity <= self.max_quantity:
            raise ValueError(f"数量必须为1至{self.max_quantity}的整数")
        with self.connection(write=True) as db:
            row = db.execute("SELECT * FROM demo_tickets WHERE id=?", (ticket_id,)).fetchone()
            if not row or row["remaining"] < quantity:
                return self._result("no_data", message="模拟票号不存在或库存不足；未创建订单。")
            quote_id = "DEMO-QUOTE-" + uuid.uuid4().hex
            expires_at = time.time() + self.quote_ttl_seconds
            quote = {"quote_id": quote_id, "ticket": self._ticket(row), "quantity": quantity,
                     "amount": float(Decimal(row["price_cents"] * quantity) / 100), "currency": row["currency"],
                     "expires_at": expires_at, "source_kind": "synthetic", "source": DEMO_SOURCE}
            db.execute("INSERT INTO demo_quotes(id,expires_at,payload) VALUES (?,?,?)",
                       (quote_id, expires_at, json.dumps(quote, ensure_ascii=False)))
            return self._result("success", quote=quote, requires_confirmation=True,
                                message="请在页面核对并单独确认模拟报价。报价最长5分钟有效，未锁库存、未扣款、未出票。")

    def confirm(self, quote_id, request_id):
        """Human-facing UI action only; transactionally validates then simulates an order."""
        quote_id = _text(quote_id, "模拟报价单ID")
        request_id = _text(request_id, "请求ID", 128)
        with self.connection(write=True) as db:
            previous = db.execute("SELECT k.quote_id,o.payload FROM demo_request_keys k "
                                  "JOIN demo_orders o ON o.id=k.order_id WHERE k.request_id=?",
                                  (request_id,)).fetchone()
            if previous:
                if previous["quote_id"] != quote_id:
                    raise ValueError("相同请求ID不能用于不同报价单")
                return {**json.loads(previous["payload"]), "replayed": True}
            record = db.execute("SELECT * FROM demo_quotes WHERE id=?", (quote_id,)).fetchone()
            if not record:
                return self._result("no_data", message="模拟报价单不存在，请重新查询并确认；未创建订单。")
            if record["used_order_id"]:
                order = db.execute("SELECT payload FROM demo_orders WHERE id=?", (record["used_order_id"],)).fetchone()
                db.execute("INSERT INTO demo_request_keys VALUES (?,?,?)",
                           (request_id, quote_id, record["used_order_id"]))
                return {**json.loads(order["payload"]), "replayed": True}
            if record["expires_at"] <= time.time():
                return self._result("expired", message="模拟报价单已过期，请重新查询并确认；未创建订单。")
            quote = json.loads(record["payload"])
            ticket, quantity = quote["ticket"], quote["quantity"]
            row = db.execute("SELECT * FROM demo_tickets WHERE id=?", (ticket["id"],)).fetchone()
            if not row or row["remaining"] < quantity:
                return self._result("no_data", message="模拟库存不足或票已移除，请重新查询；未创建订单。")
            current = self._ticket(row)
            fields = ("price", "kind", "departure", "arrival", "date", "service", "seat", "currency")
            if any(current[field] != ticket[field] for field in fields):
                return self._result("changed", message="模拟价格或行程已变化，请重新查询并确认；未创建订单。")
            order_id = "SIM-" + uuid.uuid4().hex[:16].upper()
            result = self._result("success", order_id=order_id, quote_id=quote_id, ticket_id=ticket["id"],
                                  ticket=ticket, quantity=quantity, amount=quote["amount"], currency=row["currency"],
                                  created_at=datetime.now(timezone.utc).isoformat(),
                                  message="本地模拟订单已创建，无支付、无真实出票。")
            changed = db.execute("UPDATE demo_tickets SET remaining=remaining-? WHERE id=? AND remaining>=?",
                                 (quantity, ticket["id"], quantity)).rowcount
            if changed != 1:
                raise RuntimeError("模拟库存更新失败")
            db.execute("INSERT INTO demo_orders VALUES (?,?,?,?)",
                       (order_id, request_id, quote_id, json.dumps(result, ensure_ascii=False)))
            db.execute("INSERT INTO demo_request_keys VALUES (?,?,?)", (request_id, quote_id, order_id))
            db.execute("UPDATE demo_quotes SET used_order_id=? WHERE id=?", (order_id, quote_id))
            return {**result, "replayed": False}
