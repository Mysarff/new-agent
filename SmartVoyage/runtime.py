import time

from .a2a import RemoteAgents
from .config import load_config
from .engine import Engine, TravelSession
from .mcp_client import ToolHub
from .model import ChatModel


async def chat(query, history=None, session=None, network=False, config=None):
    started = time.monotonic()
    config = config if config is not None else load_config()
    model = ChatModel(enable_thinking=config['enable_thinking']) if 'enable_thinking' in config else ChatModel()
    config = {**config, 'enable_thinking': model.enable_thinking}
    if network:
        result = await Engine(model, None, config, session or TravelSession(), RemoteAgents(config)).run(query, history)
    else:
        async with ToolHub(config) as hub:
            result = await Engine(model, hub, config, session or TravelSession()).run(query, history)
            result['discovery'] = hub.discovery
    result['metrics']['total_elapsed_ms'] = round((time.monotonic()-started)*1000)
    return result


async def direct_tool(name, arguments):
    config = load_config()
    async with ToolHub(config) as hub:
        return await hub.call(name, arguments)
