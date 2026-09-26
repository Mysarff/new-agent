from .a2a import RemoteAgents
from .config import load_config
from .engine import Engine, TravelSession
from .mcp_client import ToolHub
from .model import ChatModel


async def chat(query, history=None, session=None, network=False):
    config = load_config()
    model = ChatModel()
    if network:
        return await Engine(model, None, config, session or TravelSession(), RemoteAgents(config)).run(query, history)
    async with ToolHub(config) as hub:
        result = await Engine(model, hub, config, session or TravelSession()).run(query, history)
        result['discovery'] = hub.discovery
        return result


async def direct_tool(name, arguments):
    config = load_config()
    async with ToolHub(config) as hub:
        return await hub.call(name, arguments)
