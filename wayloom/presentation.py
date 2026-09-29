"""User-facing formatting; keep evidence IDs and numeric values intact internally."""
import re
from urllib.parse import quote, urlsplit

CITATION = re.compile(r'\[([KW][A-Za-z0-9_-]+)\]')
CELSIUS = re.compile(r'\^\s*(?:\{\s*\\circ\s*\}|\\circ)\s*(?:\\(?:text|mathrm)\s*\{\s*C\s*\}|\{\s*C\s*\}|C)')


def readable_units(text):
    """Normalize known Celsius notation without evaluating math or changing numbers."""
    text = CELSIUS.sub('℃', text)
    def simple_temperature(match):
        body = match.group(1).strip().replace(r'\sim', '～').replace(r'\approx', '约')
        body = body.replace(r'\,', ' ').replace(r'\!', '').replace('°C', '℃')
        if '℃' in body and re.fullmatch(r'[\d\s℃.＋+−\-～~至到约,，—–]+', body):
            return body
        return match.group(0)
    text = re.sub(r'\\\((.*?)\\\)', simple_temperature, text, flags=re.S)
    text = re.sub(r'(?<!\$)\$(?!\$)([^$\n]+)\$(?!\$)', simple_temperature, text)
    text = re.sub(r'(?<=℃)\s*\\sim\s*', '～', text)
    return text


def safe_source_url(value):
    if not isinstance(value, str) or any(ord(c) < 32 for c in value):
        return None
    try:
        parts = urlsplit(value)
        if parts.scheme not in ('http', 'https') or not parts.hostname or parts.username or parts.password:
            return None
        return quote(value, safe=':/?#%&=+;,!@-._~')
    except ValueError:
        return None


def source_catalog(items, answer=''):
    by_id = {item['id']: item for item in items if isinstance(item, dict) and isinstance(item.get('id'), str)}
    order = list(dict.fromkeys(identifier for identifier in CITATION.findall(answer) if identifier in by_id))
    order.extend(identifier for identifier in by_id if identifier not in order)
    return [{**by_id[identifier], 'number': index} for index, identifier in enumerate(order, 1)]


def format_answer(answer, catalog, scope):
    if not re.fullmatch(r'[a-z0-9-]+', scope):
        raise ValueError('Invalid citation anchor scope')
    numbers = {item['id']: item['number'] for item in catalog}
    text = readable_units(answer)
    text = CITATION.sub(lambda match: f"[来源{numbers[match[1]]}](#source-{scope}-{numbers[match[1]]})"
        if match[1] in numbers else '〔来源未核验〕', text)
    return re.sub(r'^#{1,3}\s+', '#### ', text, flags=re.M)


def trip_overview(trip):
    result = [('目的地', trip.get('destination') or '尚未确定')]
    if trip.get('departure'):
        result.insert(0, ('出发地', trip['departure']))
    start, end = trip.get('start_date'), trip.get('end_date')
    if start and start == end:
        day = start + '（当天）'
    elif start and end:
        day = start + ' 至 ' + end
    else:
        day = start or end or '尚未确定'
    result.extend([('出行日期', day), ('同行人数', f"{trip['travelers']}人" if trip.get('travelers') else '未提供'),
                   ('旅行偏好', '、'.join(trip.get('preferences') or []) or '未提供')])
    return result
