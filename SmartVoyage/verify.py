"""Explicit, repeatable live acceptance checks; model responses alone never pass.

Run ``python -m SmartVoyage.verify --live --network`` after starting the A2A
stack. Without --live this module only describes its checks and makes no calls.
These checks verify execution, context and evidence; they are not a benchmark of
every claim's factual accuracy or a substitute for human review of the answers.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from .config import ROOT, load_config
from .engine import TravelSession
from .runtime import chat


PLAN = [
    ("xian_tomorrow", "中国陕西西安明天的天气怎么样？请查真实天气预报。"),
    ("xian_day_after", "那后天呢？"),
    ("tokyo_same_date", "目的地改为日本东京，日期还是刚才同一天，查一下天气。"),
    ("caac_power_bank", "乘坐中国境内航班，可以携带没有3C标识的充电宝吗？请检索资料并引用来源，说明适用范围与核验日期。"),
    ("tickets_unconfigured", "请查询{date}从北京到上海的真实火车票票价和余票。不要模拟数据，不要替我下单。"),
]


def _events(result, tool, status=None):
    return [event for event in result.get("trace", []) if event.get("event") == "tool"
            and event.get("tool") == tool and (status is None or event.get("status") == status)]


def _delegated(result, agent):
    return any(event.get("actor") == "coordinator" for event in _events(result, "delegate__" + agent, "success"))


def _named(value, aliases):
    value = str(value).casefold().replace("’", "'")
    return any(alias.casefold() in value for alias in aliases)


def _forecast_records(result):
    return [item for item in result.get("retrieved_evidence", [])
            if item.get("metadata", {}).get("kind") == "weather_forecast"]


def evaluate_weather(result, expected_date, city):
    aliases = ("西安", "xi'an", "xian") if city == "xian" else ("东京", "東京", "tokyo")
    country = "CN" if city == "xian" else "JP"
    expected_zone = "Asia/Shanghai" if city == "xian" else "Asia/Tokyo"
    forecast_calls = _events(result, "travel__forecast", "success")
    records = _forecast_records(result)
    cited_ids = {item.get("id") for item in result.get("citations", [])}

    def matches(record):
        data, metadata = record.get("data", {}), record.get("metadata", {})
        place, days = data.get("location", {}), data.get("daily", [])
        same_region = city != "xian" or _named(place.get("admin1", ""), ("陕西", "shaanxi"))
        real_values = any(type(day.get(key)) in (int, float) and math.isfinite(day[key])
                          for day in days for key in ("temperature_2m_max", "temperature_2m_min", "precipitation_probability_max"))
        invoked = any(call.get("arguments", {}).get("location_id") == place.get("id")
                      and call.get("arguments", {}).get("start_date") == expected_date
                      and call.get("arguments", {}).get("end_date", expected_date) in (None, expected_date)
                      for call in forecast_calls)
        return (metadata.get("provenance") == "live_api"
                and urlsplit(record.get("source", "")).hostname == "api.open-meteo.com"
                and bool(record.get("retrieved_at"))
                and _named(place.get("name", ""), aliases) and place.get("country_code") == country
                and same_region and data.get("timezone") == expected_zone
                and [day.get("date") for day in days] == [expected_date]
                and real_values and invoked)

    valid_records = [record for record in records if matches(record)]
    cited_records = [record for record in records if record.get("id") in cited_ids]
    trip = result.get("context", {}).get("trip", {})
    criteria = {
        "weather_agent_selected": _delegated(result, "weather"),
        "forecast_tool_succeeded": bool(forecast_calls),
        "context_destination_correct": _named(trip.get("destination", ""), aliases),
        "context_date_correct": trip.get("start_date") == expected_date,
        "live_forecast_city_date_and_values_match": bool(valid_records),
        "answer_cites_matching_live_forecast": bool(cited_records) and all(matches(record) for record in cited_records),
        "no_model_error": not any(event.get("event") == "model_error" for event in result.get("trace", [])),
    }
    observed = [{"id": record.get("id"), "location": record.get("data", {}).get("location"),
                 "dates": [day.get("date") for day in record.get("data", {}).get("daily", [])],
                 "retrieved_at": record.get("retrieved_at"), "cited": record.get("id") in cited_ids}
                for record in records]
    return criteria, {"expected_date": expected_date, "expected_city": city, "forecasts": observed}


def evaluate_knowledge(result):
    citations = [item for item in result.get("citations", []) if str(item.get("id", "")).startswith("K")]
    relevant = [item for item in citations if "充电宝" in str(item.get("text", ""))
                and item.get("metadata", {}).get("source_type") == "official_summary"
                and item.get("temporal_status") not in ("expired", "not_yet_effective", "not_yet_published")]
    criteria = {
        "knowledge_agent_selected": _delegated(result, "knowledge"),
        "rag_returned_candidates": bool(_events(result, "travel__search_knowledge", "candidates")),
        "answer_has_valid_relevant_K_citation": bool(relevant),
        "citation_id_present_in_answer": any("[" + item["id"] + "]" in result.get("answer", "") for item in relevant),
        "answer_identifies_domestic_scope": "境内" in result.get("answer", "") or "国内" in result.get("answer", ""),
        "no_model_error": not any(event.get("event") == "model_error" for event in result.get("trace", [])),
    }
    return criteria, {"relevant_citation_ids": [item["id"] for item in relevant]}


def evaluate_unconfigured_tickets(result, expected_date):
    unavailable_calls = _events(result, "travel__query_tickets", "unavailable")
    def intended_query(event):
        args = event.get("arguments", {})
        return (args.get("kind") == "train" and "北京" in args.get("departure", "")
                and "上海" in args.get("arrival", "") and args.get("date") == expected_date)
    answer = result.get("answer", "")
    criteria = {
        "tickets_agent_selected": _delegated(result, "tickets"),
        "real_ticket_query_is_unavailable": any(intended_query(event) for event in unavailable_calls),
        "answer_explicitly_reports_missing_access": any(term in answer for term in
            ("未接入", "尚未接入", "未配置", "没有接入", "暂未接入", "尚未配置", "无法获取", "无法查询", "不能查询", "尚未完全接通", "无法为您提供")),
        "no_simulated_ticket_or_booking_tool_called": not any(
            "demo" in str(event.get("tool", "")) or "confirm" in str(event.get("tool", ""))
            for event in result.get("trace", []) if event.get("event") == "tool"),
        "no_ticket_candidates_or_quotes": not result.get("context", {}).get("recent_ticket_candidates") and not result.get("quotes"),
        "no_model_error": not any(event.get("event") == "model_error" for event in result.get("trace", [])),
    }
    return criteria, {"expected_date": expected_date, "unavailable_query_count": len(unavailable_calls)}


def _redact(value):
    """Strip configured credentials/private provider addresses from stored reports."""
    encoded = json.dumps(value, ensure_ascii=False)
    private = []
    for key, secret in os.environ.items():
        if key.startswith(("SMARTVOYAGE_", "AMAP_", "TAVILY_")) and key.endswith(("API_KEY", "TOKEN", "PASSWORD", "BASE_URL")) and secret:
            private.append(secret)
            if key.endswith("BASE_URL"):
                hostname = urlsplit(secret).netloc
                if hostname:
                    private.append(hostname)
    for secret in sorted(set(private), key=len, reverse=True):
        # JSON escaping keeps unusual credential characters safe to replace.
        encoded = encoded.replace(json.dumps(secret, ensure_ascii=False)[1:-1], "[REDACTED]")
    return json.loads(encoded)


def _write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_redact(report), ensure_ascii=False, indent=2), encoding="utf-8")
    # Windows readers/antivirus can briefly hold the existing report open.
    # Keep the temporary file intact and retry only this reversible replacement.
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (attempt + 1))


async def run_live(network=False, output=None):
    config = load_config()
    now = datetime.now(ZoneInfo(config["user_timezone"]))
    tomorrow = (now.date() + timedelta(days=1)).isoformat()
    day_after = (now.date() + timedelta(days=2)).isoformat()
    path = Path(output) if output else ROOT / "var" / "live_verification.json"
    report = {"schema_version": 1, "started_at": now.isoformat(), "timezone": config["user_timezone"],
              "mode": "real_model_a2a_mcp" if network else "real_model_local_mcp", "status": "running",
              "scope": "验证实际执行、上下文和证据匹配；不代表已逐句审查答案准确性，也不代表长期成功率。", "steps": []}
    session, history = TravelSession(), []
    _write_report(path, report)
    for index, (identifier, template) in enumerate(PLAN):
        query = template.format(date=day_after)
        if index >= 3:
            session, history = TravelSession(), []
        if identifier == "tickets_unconfigured" and os.getenv("SMARTVOYAGE_TICKET_BASE_URL", "").strip():
            report["steps"].append({"id": identifier, "query": query, "status": "skipped", "criteria": {},
                                    "reason": "真实票务接口已配置，未配置场景不适用；没有修改配置或发起真实购买。"})
            _write_report(path, report)
            print(f"{identifier}: SKIPPED", flush=True)
            continue
        started = time.monotonic()
        print(f"{identifier}: RUNNING", flush=True)
        try:
            result = await chat(query, history=history, session=session, network=network)
            if index < 3:
                criteria, observed = evaluate_weather(result, tomorrow if index == 0 else day_after, "tokyo" if index == 2 else "xian")
            elif identifier == "caac_power_bank":
                criteria, observed = evaluate_knowledge(result)
            else:
                criteria, observed = evaluate_unconfigured_tickets(result, day_after)
            if network:
                criteria["real_a2a_task_completed"] = any(event.get("event") == "A2A tasks/send" and event.get("status") == "success"
                                                            for event in result.get("trace", []))
            criteria["answer_nonempty"] = bool(result.get("answer", "").strip())
            status = "passed" if criteria and all(criteria.values()) else "failed"
            record = {"id": identifier, "query": query, "status": status, "criteria": criteria,
                      "observed": observed, "elapsed_seconds": round(time.monotonic() - started, 2),
                      **{key: copy.deepcopy(result.get(key)) for key in
                         ("answer", "trace", "citations", "retrieved_evidence", "warnings", "context", "tool_calls", "delegations")}}
            history.extend([{"role": "user", "content": query}, {"role": "assistant", "content": result.get("answer", "")}])
        except Exception as exc:
            record = {"id": identifier, "query": query, "status": "failed", "criteria": {"completed_without_exception": False},
                      "error_type": type(exc).__name__, "elapsed_seconds": round(time.monotonic() - started, 2),
                      "answer": "", "trace": [], "citations": [],
                      "reason": "执行异常；未把异常文本写入报告，避免暴露密钥或服务地址。请检查对应服务状态。"}
        except asyncio.CancelledError:
            report["status"] = "interrupted"
            report["steps"].append({"id": identifier, "query": query, "status": "interrupted", "criteria": {}})
            _write_report(path, report)
            raise
        report["steps"].append(record)
        _write_report(path, report)
        failed = [name for name, passed in record["criteria"].items() if not passed]
        print(f"{identifier}: {record['status'].upper()}" + ("; " + ", ".join(failed) if failed else ""), flush=True)
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    report["counts"] = {status: sum(step["status"] == status for step in report["steps"]) for status in ("passed", "failed", "skipped")}
    report["status"] = "failed" if report["counts"]["failed"] else "partial" if report["counts"]["skipped"] else "passed"
    _write_report(path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description="SmartVoyage真实联调验证；显式--live才调用付费模型/网络工具")
    parser.add_argument("--live", action="store_true", help="显式启用真实模型和实时工具，可能产生API费用")
    parser.add_argument("--network", action="store_true", help="经已启动的A2A Agent服务执行；省略则本进程MCP编排")
    parser.add_argument("--output", type=Path, default=None, help="报告路径；默认var/live_verification.json")
    args = parser.parse_args()
    # Third-party informational log messages can contain private API hosts.
    for name in ("httpx", "httpcore", "python_a2a", "urllib3"):
        logging.getLogger(name).setLevel(logging.ERROR)
    if not args.live:
        print("未发起模型或网络请求。执行真实联调需添加 --live；完整A2A联调再添加 --network。")
        for identifier, _ in PLAN:
            print(identifier)
        return 0
    try:
        report = asyncio.run(run_live(network=args.network, output=args.output))
    except KeyboardInterrupt:
        print("验证已中断，已完成步骤保存在报告中。")
        return 130
    except Exception as exc:
        print("验证未启动或未完成：" + type(exc).__name__ + "。未输出服务地址或认证信息。")
        return 1
    print("结果：" + report["status"] + "; " + json.dumps(report["counts"], ensure_ascii=False))
    return 0 if report["status"] == "passed" else 2 if report["status"] == "partial" else 1


if __name__ == "__main__":
    raise SystemExit(main())
