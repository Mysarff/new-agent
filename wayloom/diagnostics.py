"""Safe, actionable diagnostics: never return provider bodies, URLs or credentials."""
import asyncio
import json
from datetime import datetime, timezone
from uuid import uuid4

import httpx


class ModelResponseError(ValueError):
    """The provider returned HTTP success without a usable assistant message."""


def model_failure(exc):
    code, message = 'unexpected', '模型请求未完成，请通过连接自检检查模型接口。'
    status = None
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 401:
            code, message = 'authentication', '模型接口拒绝了身份验证（HTTP 401），请检查 API Key 是否有效且与服务地址匹配。'
        elif status == 403:
            code, message = 'permission', '模型接口拒绝访问（HTTP 403），请检查账号、工作空间和模型访问权限。'
        elif status == 429:
            code, message = 'rate_limit', '模型接口返回限流或配额限制（HTTP 429），请检查供应方控制台后重试。'
        elif status in (400, 404, 422):
            code, message = 'configuration', f'模型接口拒绝了请求（HTTP {status}），请核对 Base URL、模型名及工具调用/思考参数支持情况。'
        elif status >= 500:
            code, message = 'provider_unavailable', f'模型服务暂时不可用（HTTP {status}），请稍后重试。'
        else:
            code, message = 'http_error', f'模型接口返回 HTTP {status}，请检查供应方控制台。'
    elif isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        code, message = 'timeout', '等待模型响应超时，请稍后重试；若持续发生，请检查网络或模型响应速度。'
    elif isinstance(exc, (httpx.ConnectError, httpx.ProxyError)):
        code, message = 'connection', '无法连接模型服务，请检查服务地址、网络和代理；这不能说明 API Key 已失效。'
    elif isinstance(exc, httpx.RequestError):
        code, message = 'transport', '模型响应传输中断，请检查网络后重试。'
    elif isinstance(exc, ModelResponseError):
        code, message = 'response_format', '接口返回成功，但没有符合协议的模型消息，请检查 Chat Completions 兼容性。'
    # Exception strings and HTTP response bodies can contain request headers or
    # user text. Only controlled categories and numeric statuses cross the UI.
    result = {'code': code, 'message': message}
    if status is not None:
        result['http_status'] = status
    return result


async def check_connections(config, *, network=False, model=None):
    """Explicit on-demand live probe. Does not replace an end-to-end acceptance run."""
    from .model import ChatModel
    from .a2a import endpoint

    async def check_model():
        try:
            probe_model = model or (ChatModel(enable_thinking=config['enable_thinking'])
                                    if 'enable_thinking' in config else ChatModel())
            marker = uuid4().hex
            tool = {'type': 'function', 'function': {'name': 'wayloom_connection_check',
                'description': '连接自检：原样返回用户提供的标记，不执行任何业务操作。',
                'parameters': {'type': 'object', 'properties': {'marker': {'type': 'string'}},
                               'required': ['marker'], 'additionalProperties': False}}}
            response = await asyncio.wait_for(probe_model.complete([
                {'role': 'system', 'content': '这是工具协议自检。必须调用 wayloom_connection_check，原样传入 marker，不输出普通回答。'},
                {'role': 'user', 'content': 'marker=' + marker}], [tool],
                tool_choice={'type': 'function', 'function': {'name': 'wayloom_connection_check'}}), 30)
            calls = response.get('tool_calls') or []
            try:
                valid = (isinstance(calls, list) and len(calls) == 1
                         and calls[0]['function']['name'] == 'wayloom_connection_check'
                         and json.loads(calls[0]['function']['arguments']) == {'marker': marker})
            except (ValueError, TypeError, KeyError):
                valid = False
            return {'name': '对话模型与工具协议', 'status': 'ok' if valid else 'error',
                    'message': '模型已响应，工具调用格式正确。' if valid else '模型已响应，但未按要求返回工具调用；请检查模型的工具协议支持。'}
        except Exception as exc:
            return {'name': '对话模型与工具协议', 'status': 'error', **model_failure(exc)}

    async def check_agent(agent):
        try:
            async with httpx.AsyncClient(timeout=4, follow_redirects=False, trust_env=False) as client:
                response = await client.get(endpoint(agent) + '/health')
                response.raise_for_status()
                data = response.json()
            if data.get('status') != 'ok' or data.get('agent') != agent['id']:
                raise ValueError('Unexpected health response')
            return {'name': agent['name'], 'status': 'ok', 'message': '专家进程可连接。'}
        except Exception:
            return {'name': agent['name'], 'status': 'error',
                    'message': '专家服务未就绪，请在项目目录运行 python -m wayloom.stack，并保持启动器运行。'}

    checks = await asyncio.gather(check_model(), *(check_agent(a) for a in config['agents'] if network))
    return {'checked_at': datetime.now(timezone.utc).isoformat(), 'checks': checks,
            'notice': '本次检查会调用模型 API；专家健康检查只验证进程可连接，不代表天气、搜索、RAG 等业务已全部通过。'}
