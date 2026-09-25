"""Repeatable implementation tests plus a deliberately small synthetic retrieval smoke set."""
import argparse
import asyncio
import io
import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT, load_config
from .mcp_client import ToolHub
from .rag import Retriever, build_index


async def live_tools(config):
    async with ToolHub(config) as hub:
        results = []
        for name, arguments in [
            ('public__get_python_package', {'package': 'httpx'}),
            ('public__get_python_package', {'package': 'pydantic'}),
            ('public__get_github_repository', {'owner': 'Mysarff', 'repository': 'new-agent'}),
        ]:
            try:
                result = await hub.call(name, arguments)
                results.append({'tool': name, 'arguments': arguments, 'result': result})
            except Exception as exc:
                results.append({'tool': name, 'arguments': arguments, 'error': type(exc).__name__})
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live-tools', action='store_true', help='Verify actual public API calls through MCP')
    args = parser.parse_args()
    config = load_config()
    if args.live_tools:
        report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'checks': asyncio.run(live_tools(config))}
        target = ROOT / 'reports/live_tools.json'
        passed = all(item.get('result', {}).get('status') == 'success' for item in report['checks'])
        report['passed'] = passed
    else:
        # Dedicated evaluation index: never replace a user's optional dense index.
        evaluation_config = dict(config, index_path=str(ROOT / 'var/evaluation.json'))
        ingest = build_index(evaluation_config)
        retriever = Retriever(evaluation_config)
        cases = json.loads((ROOT / 'evaluation/retrieval_cases.json').read_text(encoding='utf-8'))
        outcomes = []
        for case in cases:
            hits = retriever.search(case['query'], 3)['chunks']
            outcomes.append(dict(case, hit_at_3=any(hit['source'] == case['source'] for hit in hits),
                                 returned_sources=[hit['source'] for hit in hits]))
        stream = io.StringIO()
        suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), top_level_dir=str(ROOT))
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'python': sys.version.split()[0],
                  'tests': {'run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors)},
                  'ingest': ingest, 'retrieval_smoke': {'cases': outcomes,
                  'hit_at_3': sum(item['hit_at_3'] for item in outcomes) / len(outcomes),
                  'scope': 'Synthetic author-written smoke set, not held-out benchmark or production quality evidence'},
                  'external_llm': 'not_tested_no_credentials', 'external_embeddings': 'not_tested_no_credentials',
                  'test_log': stream.getvalue()}
        target = ROOT / 'reports/verification.json'
        passed = result.wasSuccessful() and all(item['hit_at_3'] for item in outcomes)
        report['passed'] = passed
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'test_log'}, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
