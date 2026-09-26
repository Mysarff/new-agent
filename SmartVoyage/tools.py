"""Real MCP tools. Confirmation is deliberately absent from the model's tool catalog."""
import logging
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from .config import load_config
from .knowledge import TravelKnowledge
from .model import Embeddings
from .tickets import DemoBookingStore, query_tickets as provider_tickets
from .travel_web import search_places as places, search_travel_web as web_search
from .weather import geocode as geo, forecast as weather

READ = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
PREPARE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)


def build_server():
    config = load_config()
    server = FastMCP('SmartVoyageTravelTools', log_level='WARNING')

    @server.tool(annotations=READ)
    async def geocode(query: str, country_code: str | None = None) -> dict:
        """实时搜索任意地点，返回带国家/行政区/时区的候选ID。country_code是可选ISO国家代码。不要擅自选择同名地点。"""
        return await geo(query, country_code)

    @server.tool(annotations=READ)
    async def forecast(location_id: int, start_date: str, end_date: str | None = None) -> dict:
        """以 geocode 返回的地点ID查询指定ISO日期区间的真实天气预报。最多目的地今天起16天，无样例兜底。"""
        return await weather(location_id, start_date, end_date)

    @server.tool(annotations=READ)
    def search_knowledge(query: str, travel_date: str | None = None, region: str | None = None,
                         top_k: Annotated[int, Field(ge=1, le=10)] = 5, include_historical: bool = False) -> dict:
        """检索旅行专业知识/官方公告快照。传旅行日期及层级地区如CN/北京；过滤不适用或过期资料。查不到不代表没有公告。"""
        embedding = Embeddings()
        return TravelKnowledge(config, embedding if embedding.model else None).search(
            query, travel_date=travel_date, region=region, top_k=top_k, include_historical=include_historical)

    @server.tool(annotations=READ)
    async def search_places(query: str, city: str) -> dict:
        """通过配置的高德API实时查询指定城市的景点、餐饮和室内场所，无固定城市表。未配置返回unavailable。"""
        return await places(query, city)

    @server.tool(annotations=READ)
    async def search_travel_web(query: str, official_domains: list[str] | None = None) -> dict:
        """通过配置的搜索API查询最新出行公告和公开资料。优先传景区/主管部门官方域名；结果仅是摘录，需要核对有效期。"""
        return await web_search(query, official_domains)

    @server.tool(annotations=READ)
    async def query_tickets(kind: str, departure: str, arrival: str, date: str) -> dict:
        """查询已接入供应方的真实票务，不生成模拟价格或余票。date为ISO日期，未接入返回unavailable。"""
        return await provider_tickets(kind, departure, arrival, date)

    @server.tool(annotations=READ)
    def query_demo_tickets(kind: str, departure: str, arrival: str, date: str) -> dict:
        """仅查询用户主动创建的本地模拟票务；不代表真实余票，不能购票或支付。"""
        return DemoBookingStore(config['booking_db']).query(kind, departure, arrival, date)

    @server.tool(annotations=PREPARE)
    def prepare_demo_booking(ticket_id: str, quantity: Annotated[int, Field(ge=1, le=5)]) -> dict:
        """为已展示的模拟票号和用户明确数量准备5分钟报价。必须由页面显示并由用户点击确认，当前调用不预订。"""
        return DemoBookingStore(config['booking_db']).prepare(ticket_id, quantity)

    return server


if __name__ == '__main__':
    # Some providers use query-string keys: never log request URLs.
    logging.getLogger('httpx').setLevel(logging.WARNING)
    build_server().run(transport='stdio')
