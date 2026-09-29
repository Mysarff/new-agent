"""Auditable development evaluations, with explicit opt-in for paid live calls."""
import argparse
import asyncio
import csv
import hashlib
import json
import logging
import math
import os
import statistics
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .config import ROOT, load_config, env_value
from .engine import TravelSession
from .knowledge import TravelKnowledge, fingerprint, ingest
from .model import Embeddings
from .presentation import CITATION
from .runtime import chat
from .verify import _write_report

DEFAULT_CASES = ROOT / 'evaluations/travel_cases.json'
SMOKE = ('weather_xian', 'rain_plan', 'knowledge_powerbank', 'tickets_unavailable', 'places_only')
INCOMPLETE = {'model_error', 'model_cancelled', 'run_timeout', 'context_error', 'budget_exhausted', 'round_limit', 'ungrounded_after_failure'}


def resolve_dates(value, today):
    if isinstance(value, str):
        return value.replace('{tomorrow}', (today + timedelta(days=1)).isoformat()).replace(
            '{day_after}', (today + timedelta(days=2)).isoformat())
    if isinstance(value, list):
        return [resolve_dates(item, today) for item in value]
    if isinstance(value, dict):
        return {key: resolve_dates(item, today) for key, item in value.items()}
    return value


def score_execution(case, result):
    """Checks of recorded execution, not a semantic answer-correctness judge."""
    trace, answer = result.get('trace', []), result.get('answer', '')
    selected = {row['tool'].removeprefix('delegate__') for row in trace
                if row.get('event') == 'tool' and row.get('tool', '').startswith('delegate__')}
    expected = case['agents']
    alternatives = expected if expected and isinstance(expected[0], list) else [expected]
    checks = {'routing_exact': any(selected == set(option) for option in alternatives),
              'no_incomplete_event': not any(row.get('event') in INCOMPLETE or
                  (row.get('event') == 'evidence_review' and row.get('status') != 'completed') for row in trace),
              'answer_nonempty': bool(answer.strip())}
    tool_rows = [row for row in trace if row.get('event') == 'tool']
    for tool, statuses in case.get('tools', {}).items():
        checks['tool:' + tool] = any(row.get('tool') == tool and row.get('status') in statuses for row in tool_rows)
    for tool in case.get('forbidden_tools', []):
        checks['forbidden:' + tool] = not any(row.get('tool') == tool for row in tool_rows)
    trip = result.get('context', {}).get('trip', {})
    for key, expected in case.get('context', {}).items():
        checks['context:' + key] = trip.get(key) in (expected if isinstance(expected, list) else [expected])
    evidence = result.get('retrieved_evidence', [])
    ids = set(CITATION.findall(answer))
    checks['citation_ids_valid'] = not (ids - {item['id'] for item in evidence}) and '[引用未验证]' not in answer
    if evidence:
        checks['evidence_cited'] = bool(ids)
    if case.get('documents'):
        cited_docs = {item.get('document_id') for item in evidence if item['id'] in ids}
        checks['expected_document_cited'] = bool(cited_docs & set(case['documents']))
    if case.get('place_review'):
        checks['place_review_completed'] = result.get('answer_review') == 'completed'
    if case.get('weather'):
        target = case['weather']
        def matches(item):
            data = item.get('data') or {}
            place = data.get('location') or {}
            names = {name.casefold().replace('’', "'") for name in target['names']}
            return (item.get('metadata', {}).get('kind') == 'weather_forecast'
                    and str(place.get('name', '')).casefold().replace('’', "'") in names
                    and place.get('country_code') == target['country']
                    and [row.get('date') for row in data.get('daily', [])] == [target['date']]
                    and item['id'] in ids)
        checks['cited_weather_matches_target'] = any(matches(item) for item in evidence)
    return checks


def percentile(values, probability):
    """Nearest-rank quantile; small sample P95 can equal the maximum."""
    return sorted(values)[max(0, math.ceil(len(values) * probability) - 1)] if values else None


def summarize(records):
    groups = {}
    for profile in dict.fromkeys(row['profile'] for row in records):
        attempted = [row for row in records if row['profile'] == profile and row['status'] != 'skipped']
        latencies = [row['elapsed_seconds'] for row in attempted]
        call_counts = [row['metrics']['model_calls'] for row in attempted if 'model_calls' in row.get('metrics', {})]
        complete_usage = [row['metrics']['total_tokens'] for row in attempted
                          if row.get('metrics', {}).get('total_tokens') is not None]
        reviewed = [row['manual_review'] for row in attempted if
                    row.get('manual_review', {}).get('status') == 'completed'
                    and row['manual_review'].get('reviewer')
                    and type(row['manual_review'].get('answer_correct')) is bool
                    and type(row['manual_review'].get('covers_user_request')) is bool]
        claims = [review for review in reviewed if type(review.get('claim_count')) is int
                  and type(review.get('supported_claim_count')) is int
                  and 0 <= review['supported_claim_count'] <= review['claim_count']]
        fully_reviewed = bool(attempted) and len(reviewed) == len(attempted)
        groups[profile] = {'attempted': len(attempted),
            'skipped': sum(row['profile'] == profile and row['status'] == 'skipped' for row in records),
            'execution_passed': sum(row['status'] == 'passed' for row in attempted),
            'execution_pass_rate': sum(row['status'] == 'passed' for row in attempted)/len(attempted) if attempted else None,
            'routing_exact_rate': sum(row.get('checks', {}).get('routing_exact', False) for row in attempted)/len(attempted) if attempted else None,
            'timeout_rate': sum(any(event.get('event') in ('run_timeout', 'model_cancelled') or
                event.get('error') in ('TimeoutError', 'ReadTimeout', 'ConnectTimeout', 'CancelledError')
                for event in row.get('trace', [])) or row.get('error_type') == 'TimeoutError' for row in attempted)/len(attempted) if attempted else None,
            'latency_p50_seconds': percentile(latencies, .50), 'latency_p95_seconds': percentile(latencies, .95),
            'latency_mean_seconds': statistics.mean(latencies) if latencies else None,
            'mean_model_calls': statistics.mean(call_counts) if call_counts and len(call_counts) == len(attempted) else None,
            'token_complete_records': len(complete_usage),
            'mean_total_tokens': statistics.mean(complete_usage) if len(complete_usage) == len(attempted) and attempted else None,
            'manual_reviewed_records': len(reviewed),
            'semantic_accuracy': sum(review['answer_correct'] for review in reviewed)/len(reviewed) if fully_reviewed else None,
            'request_coverage_rate': sum(review['covers_user_request'] for review in reviewed)/len(reviewed) if fully_reviewed else None,
            'supported_claim_rate': (sum(review['supported_claim_count'] for review in claims)
                /sum(review['claim_count'] for review in claims)) if fully_reviewed and len(claims) == len(reviewed)
                and sum(review['claim_count'] for review in claims) else None}
    pairs = {}
    for row in records:
        pairs.setdefault((row['case_id'], row['repeat']), {})[row['profile']] = row
    comparison = 'responsive' if 'responsive' in groups else 'compact' if 'compact' in groups else None
    comparable = [pair for pair in pairs.values() if all(pair.get(p, {}).get('status') not in (None, 'skipped')
                  for p in ('baseline', comparison))]
    return {'profiles': groups, 'paired_attempts': len(comparable),
            'comparison': comparison,
            'paired_delta_mean_seconds': statistics.mean(pair[comparison]['elapsed_seconds']-pair['baseline']['elapsed_seconds']
                for pair in comparable) if comparable else None,
            'paired_regressions': [pair['baseline']['case_id'] for pair in comparable
                if pair['baseline']['status'] == 'passed' and pair[comparison]['status'] != 'passed'],
            'quantile_method': 'nearest_rank; all attempted runs, including failures'}


def write_outputs(path, report):
    report['summary'] = summarize(report['records'])
    _write_report(path, report)
    # Review packet contains the full answers/evidence already in the redacted JSON.
    with path.with_suffix('.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['case_id', 'repeat', 'profile', 'status', 'seconds', 'model_calls', 'total_tokens', 'failed_checks'])
        for row in report['records']:
            writer.writerow([row['case_id'], row['repeat'], row['profile'], row['status'], row.get('elapsed_seconds'),
                row.get('metrics', {}).get('model_calls'), row.get('metrics', {}).get('total_tokens'),
                ';'.join(key for key, passed in row.get('checks', {}).items() if not passed)])


async def run_live(args, dataset, config):
    selected = set(args.cases.split(',')) if args.cases else set(SMOKE)
    if args.cases == 'all':
        selected = {case['id'] for case in dataset['cases']}
    unknown = selected - {case['id'] for case in dataset['cases']}
    if unknown:
        raise ValueError('Unknown case IDs: ' + ','.join(sorted(unknown)))
    now = datetime.now(ZoneInfo(config['user_timezone']))
    cases = [resolve_dates(case, now.date()) for case in dataset['cases'] if case['id'] in selected]
    output = Path(args.output or ROOT / 'var/benchmark.json')
    report = {'version': 1, 'status': 'running', 'started_at': now.isoformat(), 'scope': dataset['scope'],
              'dataset_sha256': hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
              'implementation_sha256': {name: hashlib.sha256((ROOT/'wayloom'/name).read_bytes()).hexdigest()
                  for name in ('benchmark.py','engine.py','model.py','metrics.py','grounding.py','a2a.py','runtime.py')},
              'configuration_sha256': hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
              'transport': 'a2a_mcp' if args.network else 'local_mcp',
              'model': env_value('WAYLOOM_MODEL'), 'repeats': args.repeats,
              'profile_order': 'AB/BA alternating by case and repeat, sequential execution',
              'notes': ['No paid model judge; semantic accuracy requires manual review.',
                        'Wall-clock latency includes MCP startup/cleanup; no TTFT or currency cost inferred.',
                        'Live API data and provider load can change between paired runs.'], 'records': []}
    write_outputs(output, report)
    for repeat in range(args.repeats):
        for index, case in enumerate(cases):
            profiles = ['baseline', args.comparison] if args.profile == 'both' else [args.profile]
            if (index + repeat) % 2:
                profiles.reverse()
            for profile in profiles:
                record = {'case_id': case['id'], 'repeat': repeat + 1, 'profile': profile, 'query': case['query'],
                          'model_options': {'enable_thinking': False if profile == 'responsive' else None,
                                            'compact_handoffs': profile == 'compact'}}
                if case.get('requires_unconfigured_tickets') and env_value('WAYLOOM_TICKET_BASE_URL'):
                    record.update(status='skipped', reason='Supplier is configured; missing-supplier case is not applicable.')
                else:
                    print(f"{case['id']} {profile} repeat={repeat+1}: RUNNING", flush=True)
                    started = time.monotonic()
                    try:
                        session = TravelSession(trip=case.get('initial_trip', {}))
                        result = await chat(case['query'], history=case.get('history', []), session=session,
                            network=args.network, config={**config, **record['model_options']})
                        checks = score_execution(case, result)
                        record.update(result, checks=checks, status='passed' if all(checks.values()) else 'failed')
                        record['manual_review'] = {'status':'pending', 'claim_count':None, 'supported_claim_count':None,
                            'answer_correct':None, 'covers_user_request':None, 'reviewer':None}
                    except Exception as exc:
                        record.update(status='failed', checks={'no_exception':False}, error_type=type(exc).__name__)
                    except asyncio.CancelledError:
                        record.update(status='interrupted', checks={'completed':False}, elapsed_seconds=round(time.monotonic()-started, 3))
                        report['records'].append(record)
                        report['status'] = 'interrupted'
                        write_outputs(output, report)
                        raise
                    record['elapsed_seconds'] = round(time.monotonic()-started, 3)
                report['records'].append(record)
                write_outputs(output, report)
                print(f"{case['id']} {profile}: {record['status']} {record.get('elapsed_seconds', '')}s", flush=True)
    report.update(status='completed', completed_at=datetime.now().astimezone().isoformat())
    write_outputs(output, report)
    return report


def run_rag(args, dataset, config):
    reports = []
    # Never rebuild or replace the application's existing index.
    with tempfile.TemporaryDirectory(prefix='wayloom-eval-') as temporary:
        for mode in (['bm25', 'hybrid'] if args.dense else ['bm25']):
            run_config = {**config, 'index_path': str(Path(temporary) / (mode + '.json'))}
            embedder = Embeddings() if mode == 'hybrid' else None
            built = ingest(run_config, embedder)
            retriever = TravelKnowledge(run_config, embedder)
            rows = []
            for case in dataset['rag_cases']:
                started = time.monotonic()
                result = retriever.search(case['query'], travel_date=case['date'], region=case['region'], top_k=5)
                chunks = result['chunks']
                relevant, forbidden = set(case['relevant']), set(case.get('forbidden', []))
                # Document labels, chunk-ranked top-k: duplicate chunks do not inflate recall.
                retrieved = [chunk['document_id'] for chunk in chunks]
                rows.append({'id': case['id'], 'retrieved': retrieved,
                    'recall_at_5': len(set(retrieved) & relevant)/len(relevant) if relevant else None,
                    'reciprocal_rank': next((1/rank for rank, doc in enumerate(retrieved, 1) if doc in relevant), 0) if relevant else None,
                    'filter_passed': not bool(set(retrieved) & forbidden), 'has_filter_label': bool(forbidden),
                    'elapsed_seconds': round(time.monotonic()-started, 4)})
            positive = [row for row in rows if row['recall_at_5'] is not None]
            filters = [row for row in rows if row['has_filter_label']]
            reports.append({'mode':mode, 'documents':built['documents'], 'chunks':built['chunks'], 'rows':rows,
                'positive_queries':len(positive), 'recall_at_5':statistics.mean(row['recall_at_5'] for row in positive),
                'mrr_at_5':statistics.mean(row['reciprocal_rank'] for row in positive),
                'filter_queries':len(filters), 'filter_pass_rate':sum(row['filter_passed'] for row in filters)/len(filters) if filters else None})
    report = {'scope':'Small curated development corpus; document relevance labels evaluated on top 5 chunks. Not answer accuracy.',
              'dataset_sha256':hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
              'knowledge_fingerprints':fingerprint(config['knowledge_dir']), 'results':reports}
    _write_report(Path(args.output or ROOT / 'var/rag_benchmark.json'), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Allow paid model and external API calls')
    parser.add_argument('--network', action='store_true', help='Use running A2A services')
    parser.add_argument('--rag', action='store_true', help='Measure local knowledge retrieval')
    parser.add_argument('--dense', action='store_true', help='Compare BM25 with configured embeddings; requires --live')
    parser.add_argument('--dataset', type=Path, default=DEFAULT_CASES)
    parser.add_argument('--cases', help='Comma-separated IDs, or all; defaults to 5 smoke cases')
    parser.add_argument('--profile', choices=('baseline','compact','responsive','both'), default='baseline')
    parser.add_argument('--comparison', choices=('compact','responsive'), default='responsive',
                        help='responsive explicitly disables thinking; only use with a provider that supports enable_thinking')
    parser.add_argument('--repeats', type=int, choices=range(1, 11), default=1)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--summarize', type=Path, help='Recompute a saved report after manual review; no API calls')
    args = parser.parse_args()
    if args.summarize:
        report = json.loads(args.summarize.read_text(encoding='utf-8'))
        write_outputs(args.output or args.summarize, report)
        print(json.dumps(report['summary'], ensure_ascii=False))
        return 0
    if args.dense and (not args.live or not args.rag):
        parser.error('--dense requires --rag --live')
    for name in ('httpx','httpcore','python_a2a','urllib3'):
        logging.getLogger(name).setLevel(logging.ERROR)
    dataset = json.loads(args.dataset.read_text(encoding='utf-8'))
    if not args.rag and not args.live:
        print('No API calls. Use --rag for offline retrieval, or --live for the paid paired evaluation.')
        print(', '.join(case['id'] for case in dataset['cases']))
        return 0
    config = load_config()
    if args.rag:
        report = run_rag(args, dataset, config)
        print(json.dumps([{k:v for k,v in result.items() if k != 'rows'} for result in report['results']], ensure_ascii=False))
        return 0 if all(result['recall_at_5'] == 1 and result['filter_pass_rate'] in (None, 1)
                        for result in report['results']) else 1
    report = asyncio.run(run_live(args, dataset, config))
    print(json.dumps(report['summary'], ensure_ascii=False))
    return 1 if any(row['status']=='failed' for row in report['records']) else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Interrupted; completed records were saved.')
        raise SystemExit(130)
    except Exception as exc:
        print('Evaluation could not finish: ' + type(exc).__name__ + '. Check configuration and saved report; private request details omitted.')
        raise SystemExit(1)
