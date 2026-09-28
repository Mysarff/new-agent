"""Per-turn telemetry. Nested delegation durations must not be added twice."""
import time


def model_event(actor, event, started, response=None, **fields):
    usage = (response or {}).pop('_usage', {})
    if not isinstance(usage, dict):
        usage = {}
    return {'actor': actor, 'event': event, 'elapsed_ms': round((time.monotonic() - started) * 1000),
            'usage': {key: value for key, value in usage.items()
                      if key in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'reasoning_tokens')
                      and type(value) is int and value >= 0}, **fields}


def summarize_trace(trace, elapsed_ms):
    calls = [row for row in trace if row.get('event') in
             ('model', 'model_error', 'model_cancelled', 'evidence_review')]
    leaf_tools = [row for row in trace if row.get('event') == 'tool'
                  and row.get('actor') != 'coordinator']
    tokens = {}
    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'reasoning_tokens'):
        values = [row.get('usage', {}).get(key) for row in calls]
        tokens[key] = sum(values) if values and all(type(v) is int for v in values) else None
    return {'engine_elapsed_ms': elapsed_ms, 'model_calls': len(calls),
            'model_elapsed_ms': sum(row.get('elapsed_ms', 0) for row in calls),
            'leaf_tool_calls': len(leaf_tools),
            'leaf_tool_elapsed_ms': sum(row.get('elapsed_ms', 0) for row in leaf_tools),
            'usage_reported_calls': sum(bool(row.get('usage')) for row in calls), **tokens}
