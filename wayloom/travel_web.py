"""Optional live travel search and Chinese POI provider; no fabricated fallback."""
import hashlib
import os
from datetime import datetime, timezone
from urllib.parse import urlsplit

import httpx


def _evidence(source, title, data):
    now = datetime.now(timezone.utc).isoformat()
    return {'id': 'W' + hashlib.sha256((source + now).encode()).hexdigest()[:16], 'source': source,
            'title': title, 'data': data, 'retrieved_at': now, 'metadata': {'provenance': 'live_api'}}


async def search_places(query: str, city: str, client=None):
    key = os.getenv('AMAP_API_KEY', '')
    if not key:
        return {'status': 'unavailable', 'message': '未配置高德 Web 服务 Key，无法核实实时地点数据。'}
    if not query.strip() or not city.strip() or max(len(query), len(city)) > 120:
        return {'status': 'needs_input', 'message': '请提供明确城市和景点/场所关键词。'}
    if client is None:
        async with httpx.AsyncClient(timeout=20, follow_redirects=False) as session:
            return await search_places(query, city, session)
    try:
        response = await client.get('https://restapi.amap.com/v3/place/text',
                                    params={'key': key, 'keywords': query, 'city': city, 'citylimit': 'true',
                                            'offset': 8, 'page': 1, 'extensions': 'all'})
        response.raise_for_status()
        raw = response.json()
        if not isinstance(raw, dict) or raw.get('status') != '1':
            return {'status': 'error', 'message': '高德接口拒绝请求，请检查 Key 权限、配额与接口授权。'}
        rows = raw.get('pois', [])
        if not isinstance(rows, list) or any(not isinstance(item, dict) for item in rows):
            raise ValueError('Malformed places')
        places = [{k: item.get(k) for k in ('id', 'name', 'type', 'address', 'location', 'pname', 'cityname', 'adname', 'business')}
                  for item in rows]
    except (httpx.HTTPError, ValueError, TypeError):
        return {'status': 'error', 'message': '地点服务暂未返回有效数据，请检查网络与服务配置。'}
    return {'status': 'success' if places else 'no_data', 'places': places,
            'evidence': [_evidence('https://lbs.amap.com/api/webservice/guide/api/search',
                                   f'高德地点查询：{city} {query}', {'query': query, 'city': city, 'places': places})],
            'notice': '地点信息不等于门票余量或实时营业保证；开放与预约请核对景区官方公告。'}


async def search_travel_web(query: str, official_domains: list[str] | None = None, client=None):
    key = os.getenv('TAVILY_API_KEY', '')
    if not key:
        return {'status': 'unavailable', 'message': '未配置 Tavily Key，无法联网核实最新公告；本地RAG只是资料快照。'}
    if not 1 <= len(query.strip()) <= 1000:
        raise ValueError('Invalid search query')
    domains = official_domains or []
    if len(domains) > 8 or any(not d or urlsplit('https://' + d).hostname != d or '/' in d for d in domains):
        raise ValueError('Use plain domain names')
    if client is None:
        async with httpx.AsyncClient(timeout=25, follow_redirects=False) as session:
            return await search_travel_web(query, domains, session)
    try:
        response = await client.post('https://api.tavily.com/search', headers={'Authorization': 'Bearer ' + key},
                                     json={'query': query, 'search_depth': 'basic', 'max_results': 5,
                                           'include_domains': domains, 'include_answer': False})
        response.raise_for_status()
        payload = response.json()
        rows = payload.get('results', []) if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ValueError('Malformed search results')
        evidence = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get('url'), str):
                continue
            if any(row.get(field) is not None and not isinstance(row[field], str) for field in ('content', 'title')):
                raise ValueError('Malformed search excerpt')
            url = urlsplit(row['url'])
            if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password:
                continue
            evidence.append(_evidence(row['url'], str(row.get('title', '')),
                {'snippet': str(row.get('content') or '')[:4000], 'published_at': row.get('published_date')}))
    except (httpx.HTTPError, ValueError, TypeError):
        return {'status': 'error', 'message': '搜索服务暂未返回有效数据，请检查网络与服务配置。'}
    for row in evidence:
        row['metadata']['provenance'] = 'web_search_excerpt'
    return {'status': 'success' if evidence else 'no_data', 'evidence': evidence,
            'notice': '搜索摘录可能不完整，需检查原站、发布日期与适用范围；无结果不代表没有公告。'}
