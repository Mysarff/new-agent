"""Live, worldwide weather lookup with explicit place and date boundaries.

The only geographic authority is Open-Meteo's GeoNames-backed API. There is
no city table, fabricated forecast, or fallback to historical averages.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
import re
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
LOCATION_URL = "https://geocoding-api.open-meteo.com/v1/get"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_DAYS = 16
DAILY_VARIABLES = (
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_probability_max",
    "precipitation_sum",
    "rain_sum",
    "wind_speed_10m_max",
)
# Standard-code interpretation, not a city-specific forecast or model inference.
# Source: https://open-meteo.com/en/docs#weathervariables (verified 2026-09-26).
WMO_DESCRIPTIONS = {
    0: "晴空",
    1: "大部晴朗",
    2: "局部多云",
    3: "阴天",
    45: "雾",
    48: "雾凇雾",
    51: "轻微毛毛雨",
    53: "中等毛毛雨",
    55: "浓密毛毛雨",
    56: "轻微冻毛毛雨",
    57: "浓密冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "轻度冻雨",
    67: "强冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "米雪（雪粒）",
    80: "轻微阵雨",
    81: "中等阵雨",
    82: "猛烈阵雨",
    85: "轻微阵雪",
    86: "强阵雪",
    95: "雷暴",
    96: "雷暴伴轻度冰雹",
    97: "强雷暴",
    99: "雷暴伴强冰雹",
}


class _InvalidData(ValueError):
    """The upstream payload cannot safely be treated as weather evidence."""


@asynccontextmanager
async def _client_scope(client: httpx.AsyncClient | None):
    if client is not None:
        yield client
    else:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(25.0, connect=10.0),
            headers={"User-Agent": "SmartVoyage/2.0 (travel-weather)"},
            follow_redirects=False,
        ) as owned:
            yield owned


async def _request(client: httpx.AsyncClient, url: str, params: dict) -> tuple[dict, str]:
    response = await client.get(url, params=params)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("error"):
        raise _InvalidData("Open-Meteo did not return a successful JSON object")
    return payload, str(response.request.url)


def _retrieved_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _evidence(source: str, title: str, data: dict, retrieved_at: str, kind: str) -> list[dict]:
    digest = hashlib.sha256(
        json.dumps([source, data, retrieved_at], ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return [{
        "id": "W" + digest,
        "source": source,
        "title": title,
        "data": data,
        "metadata": {"provenance": "live_api", "kind": kind, "provider": "Open-Meteo"},
        "retrieved_at": retrieved_at,
    }]


def _failure(exc: Exception, source: str) -> dict:
    result = {"status": "error", "source": source, "retrieved_at": _retrieved_at()}
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        result.update(message=f"天气数据服务返回 HTTP {code}，未取得有效数据。", http_status=code,
                      retryable=code == 429 or code >= 500)
    elif isinstance(exc, httpx.RequestError):
        result.update(message="无法连接天气数据服务或请求超时，请稍后重试。", retryable=True)
    else:
        result.update(message="天气服务返回的数据格式不完整，无法据此提供可靠结果。", retryable=False)
    # Never substitute demo fixtures or expose exception text / HTTP credentials.
    return result


def _location(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise _InvalidData("Invalid location")
    location_id = raw.get("id")
    if isinstance(location_id, bool) or not isinstance(location_id, int) or location_id <= 0:
        raise _InvalidData("Invalid location ID")
    latitude, longitude = raw.get("latitude"), raw.get("longitude")
    for value, bound in ((latitude, 90), (longitude, 180)):
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise _InvalidData("Invalid coordinates")
        if not math.isfinite(value) or abs(value) > bound:
            raise _InvalidData("Invalid coordinates")
    if not isinstance(raw.get("name"), str) or not raw["name"].strip():
        raise _InvalidData("Missing place name")
    if not isinstance(raw.get("timezone"), str):
        raise _InvalidData("Missing place timezone")
    try:
        ZoneInfo(raw["timezone"])
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise _InvalidData("Unknown place timezone; install tzdata on Windows") from exc
    keys = ("id", "name", "latitude", "longitude", "timezone", "country", "country_code",
            "admin1", "admin2", "admin3", "admin4", "feature_code", "population", "elevation")
    result = {key: raw[key] for key in keys if key in raw}
    result["location_id"] = location_id
    result["label"] = " · ".join(str(raw[key]) for key in ("name", "admin1", "admin2", "country")
                                  if raw.get(key))
    return result


async def geocode(query: str, country_code: str | None = None, *,
                  client: httpx.AsyncClient | None = None) -> dict[str, Any]:
    """Find live worldwide place candidates; never choose the first ambiguous hit.

    Use ``country_code`` for an ISO country filter or a qualifier such as
    ``London, England`` / ``西安, 陕西``. The caller may resolve candidates using
    explicit conversation context; otherwise it should ask the user to choose.
    ``client`` is optional dependency injection and remains owned by the caller.
    """
    if not isinstance(query, str) or not 2 <= len(query.strip()) <= 200:
        return {"status": "needs_input", "message": "请输入 2–200 个字符的城市或地点名称，可补充省份或国家。"}
    if country_code is not None and (not isinstance(country_code, str)
                                     or not re.fullmatch(r"[A-Za-z]{2}", country_code)):
        return {"status": "needs_input", "message": "国家筛选需使用两位 ISO 国家代码，例如 CN、JP、GB。"}
    params = {"name": query.strip(), "count": 10, "language": "zh", "format": "json"}
    if country_code:
        params["countryCode"] = country_code.upper()
    try:
        async with _client_scope(client) as session:
            payload, source = await _request(session, GEOCODING_URL, params)
        rows = payload.get("results", [])
        if not isinstance(rows, list):
            raise _InvalidData("Invalid location candidates")
        candidates = []
        seen = set()
        for row in rows:
            location = _location(row)
            if location["id"] not in seen:
                candidates.append(location)
                seen.add(location["id"])
        retrieved_at = _retrieved_at()
        result = {"query": query.strip(), "candidates": candidates, "source": source,
                  "retrieved_at": retrieved_at, "provider": "Open-Meteo / GeoNames"}
        if not candidates:
            return {**result, "status": "no_data", "message": "未找到匹配地点，请补充地区或尝试其他名称。"}
        result["evidence"] = _evidence(source, f"地点查询：{query.strip()}",
                                        {"candidates": candidates}, retrieved_at, "location_candidates")
        if len(candidates) > 1:
            return {**result, "status": "needs_input", "message": "找到多个同名或相近地点，请根据国家和行政区确认目的地。"}
        return {**result, "status": "success", "location": candidates[0]}
    except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
        return _failure(exc, GEOCODING_URL)


def _parse_date(value: str) -> date:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("Dates must use YYYY-MM-DD")
    return date.fromisoformat(value)


def _daily_rows(payload: dict, start: date, end: date) -> tuple[list[dict], dict, list[str]]:
    daily, units = payload.get("daily"), payload.get("daily_units")
    if not isinstance(daily, dict) or not isinstance(units, dict):
        raise _InvalidData("Missing daily data or units")
    expected = [(start + timedelta(days=i)).isoformat() for i in range((end - start).days + 1)]
    if daily.get("time") != expected:
        raise _InvalidData("Upstream dates do not match the requested interval")
    for variable in DAILY_VARIABLES:
        values = daily.get(variable)
        if not isinstance(values, list) or len(values) != len(expected) or not isinstance(units.get(variable), str):
            raise _InvalidData("Incomplete daily series or units")
        for value in values:
            if value is not None and (isinstance(value, bool) or not isinstance(value, (float, int))
                                      or not math.isfinite(value)):
                raise _InvalidData("Invalid weather value")
    rows = [{"date": day, **{variable: daily[variable][index] for variable in DAILY_VARIABLES}}
            for index, day in enumerate(expected)]
    for row in rows:
        code = row["weather_code"]
        row["weather_description"] = ("暂无天气现象数据" if code is None else
                                      WMO_DESCRIPTIONS.get(code, f"未知天气代码（{code}）"))
    warnings = []
    if any(value is None for variable in DAILY_VARIABLES for value in daily[variable]):
        warnings.append("部分日期或气象变量暂无预报值，null 表示缺测，不表示零；请勿据此断言不会下雨。")
    return rows, {key: units[key] for key in DAILY_VARIABLES}, warnings


async def forecast(location_id: int | str, start_date: str, end_date: str | None = None, *,
                   client: httpx.AsyncClient | None = None, now: datetime | None = None) -> dict[str, Any]:
    """Forecast for a provider-issued location ID and destination-local dates.

    Call ``geocode`` first. Coordinates/timezone are re-resolved from the ID by
    Open-Meteo; neither user text nor the model can supply invented coordinates.
    Today through today + 15 days is the supported inclusive forecast window.
    """
    if isinstance(location_id, bool) or not re.fullmatch(r"[1-9]\d{0,11}", str(location_id)):
        return {"status": "needs_input", "message": "请先查询并确认地点，提供查询结果中的 location_id。"}
    location_id = int(location_id)
    try:
        start = _parse_date(start_date)
        end = _parse_date(end_date) if end_date is not None else start
    except ValueError:
        return {"status": "needs_input", "message": "日期需要是真实日期，格式为 YYYY-MM-DD。"}
    if end < start:
        return {"status": "needs_input", "message": "结束日期不能早于开始日期。"}
    if now is not None and (not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None):
        raise ValueError("Injected clock must be a timezone-aware datetime")
    source = LOCATION_URL
    try:
        async with _client_scope(client) as session:
            try:
                raw, location_source = await _request(session, LOCATION_URL, {"id": location_id, "language": "zh"})
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    return {"status": "no_data", "message": "地点 ID 不存在，请重新查询并确认地点。", "source": LOCATION_URL}
                raise
            location = _location(raw)
            if location["id"] != location_id:
                raise _InvalidData("Provider returned a different location ID")
            destination_zone = ZoneInfo(location["timezone"])
            today = (now or datetime.now(timezone.utc)).astimezone(destination_zone).date()
            last_day = today + timedelta(days=FORECAST_DAYS - 1)
            window = {"start_date": today.isoformat(), "end_date": last_day.isoformat(),
                      "timezone": location["timezone"]}
            if start < today or end > last_day:
                return {"status": "out_of_range", "location": location, "available_range": window,
                        "requested_range": {"start_date": start.isoformat(), "end_date": end.isoformat()},
                        "source": "https://open-meteo.com/en/docs", "retrieved_at": _retrieved_at(),
                        "message": "实时预报仅支持目的地当地今天起 16 天（含今天）。该日期范围无法提供，不能当作已知天气。"}
            source = FORECAST_URL
            payload, source = await _request(session, FORECAST_URL, {
                "latitude": location["latitude"], "longitude": location["longitude"],
                "daily": ",".join(DAILY_VARIABLES), "timezone": location["timezone"],
                "start_date": start.isoformat(), "end_date": end.isoformat(),
                "temperature_unit": "celsius", "wind_speed_unit": "kmh", "precipitation_unit": "mm",
            })
        rows, units, warnings = _daily_rows(payload, start, end)
        if not any(row[variable] is not None for row in rows for variable in DAILY_VARIABLES):
            return {"status": "no_data", "message": "该地点与日期的气象变量目前均缺测，无法提供预报。",
                    "location": location, "source": source, "retrieved_at": _retrieved_at()}
        retrieved_at = _retrieved_at()
        data = {"location": location, "timezone": location["timezone"], "daily": rows, "units": units,
                "requested_range": {"start_date": start.isoformat(), "end_date": end.isoformat()},
                "available_range": window,
                "variable_notes": {"weather_code": "WMO 天气现象代码，代表当日最严重天气现象",
                                   "weather_description": "按 Open-Meteo 官方 WMO 码表翻译的中文解释，非额外预测；强度名称不代表中国降水量等级或官方预警",
                                   "precipitation_sum": "总降水量（雨、阵雨及雪的水当量）",
                                   "rain_sum": "大尺度天气系统降雨量；不包括单独统计的阵雨与雪",
                                   "precipitation_probability_max": "当日最高降水概率，不是降雨时长占比"},
                "forecast_not_observation": True}
        return {"status": "success", **data, "provider": "Open-Meteo", "source": source,
                "location_source": location_source, "retrieved_at": retrieved_at, "warnings": warnings,
                "evidence": _evidence(source, f"{location['label']} 天气预报", data, retrieved_at, "weather_forecast")}
    except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
        return _failure(exc, source)
