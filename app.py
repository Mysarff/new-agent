"""SmartVoyage's local travel workspace."""
import asyncio
import json
import os
import uuid
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from SmartVoyage.config import integrations, load_config
from SmartVoyage.engine import TravelSession
from SmartVoyage.knowledge import ingest
from SmartVoyage.model import Embeddings
from SmartVoyage.presentation import format_answer, safe_source_url, source_catalog, trip_overview
from SmartVoyage.runtime import chat, direct_tool
from SmartVoyage.tickets import DemoBookingStore

st.set_page_config(page_title='SmartVoyage · 行知', page_icon='🧭', layout='wide')
config = load_config()
available = integrations()
for key, value in [('messages', []), ('journey', TravelSession()), ('weather_candidates', []),
                   ('weather_result', None), ('knowledge_result', None), ('demo_tickets', []), ('page_quote', None)]:
    if key not in st.session_state:
        st.session_state[key] = value
if not Path(config['index_path']).exists():
    ingest(config)
journey = st.session_state.journey
store = DemoBookingStore(config['booking_db'])


def call_tool(name, arguments):
    try:
        return asyncio.run(direct_tool('travel__' + name, arguments))
    except Exception as exc:
        return {'status': 'error', 'message': f'服务暂不可用（{type(exc).__name__}），请检查配置后重试。'}


def evidence_panel(items, scope='knowledge', answer=''):
    catalog = source_catalog(items, answer)
    for item in catalog:
        meta = dict(item.get('metadata', {}))
        meta['source_type'] = {'official_summary': '官方资料摘要', 'user_supplied_unverified': '用户资料（未经核验）',
                               'synthetic': '模拟资料'}.get(meta.get('source_type'), meta.get('source_type'))
        # Only controlled scope/number values enter this HTML; source text never does.
        st.markdown(f'<span id="source-{scope}-{item["number"]}"></span>', unsafe_allow_html=True)
        with st.expander(f"来源{item['number']} · {item.get('title', '资料来源')}"):
            source = safe_source_url(item.get('source', ''))
            data = item.get('data') or {}
            places = data.get('places') if isinstance(data, dict) else None
            if source:
                st.link_button('查看地图接口说明' if places is not None else '打开原始来源', source)
            else:
                st.caption('本地导入资料，原始文件由资料提供者维护。')
            st.caption(' · '.join(f'{label}：{meta[key]}' for key, label in
                [('source_type', '资料类型'), ('published_at', '发布'), ('checked_at', '核验'),
                 ('valid_from', '生效'), ('valid_to', '截止')] if meta.get(key)))
            if item.get('temporal_status') == 'expired':
                st.warning('这份公告已超过记录中的有效期，仅供历史查询。')
            if item.get('retrieved_at'):
                st.caption('实时获取：' + item['retrieved_at'])
            if isinstance(places, list):
                st.caption('地图返回的名称、地址与分类。是否室内、开放时间、预约和门票尚需核实；接口说明不是场所官网。')
                rows = [{'名称': p.get('name') or '未提供', '地址': p.get('address') or '未提供',
                         '分类': p.get('type') or '未提供'} for p in places if isinstance(p, dict)]
                if rows:
                    st.table(pd.DataFrame(rows).set_index('名称'))
            elif item.get('metadata', {}).get('kind') == 'weather_forecast':
                st.caption('以下数值直接来自本次天气接口；属于预报，可能随时间更新。')
                rows = [{'日期': row['date'], '最低温 ℃': row.get('temperature_2m_min'),
                         '最高温 ℃': row.get('temperature_2m_max'), '降水概率 %': row.get('precipitation_probability_max'),
                         '天气现象': row.get('weather_description')} for row in data.get('daily', [])]
                if rows:
                    frame = pd.DataFrame(rows).set_index('日期')
                    for column in ('最低温 ℃', '最高温 ℃', '降水概率 %'):
                        frame[column] = frame[column].map(lambda value: '暂无' if pd.isna(value) else format(value, 'g'))
                    st.table(frame)
            elif item.get('text'):
                st.text(item['text'])
            elif isinstance(data, dict) and data.get('snippet'):
                st.text(data['snippet'])
            elif item.get('data'):
                st.json(item['data'], expanded=False)


def render_answer(result, scope):
    catalog = source_catalog(result.get('retrieved_evidence', []), result['answer'])
    if 'answer_review' not in result and any(isinstance(item.get('data'), dict) and 'places' in item['data'] for item in catalog):
        st.info('这条历史回答尚未经过新增的来源复核。请重新查询；地图资料本身不能证明馆藏、室内条件或开放情况。')
    st.markdown(format_answer(result['answer'], catalog, scope))
    for warning in result.get('warnings', []):
        st.caption(warning)
    if result.get('trace'):
        actors = list(dict.fromkeys(t['actor'] for t in result['trace'] if t.get('actor') != 'coordinator'))
        names = {a['id']: a['name'] for a in config['agents']}
        st.caption('本轮协作：' + (' → '.join(names.get(a, a) for a in actors) or '旅行协调器'))
        with st.expander('查看执行详情'):
            metrics = result.get('metrics', {})
            if metrics:
                st.caption('回答设置：' + result.get('response_mode', '环境配置'))
                elapsed = metrics.get('total_elapsed_ms', metrics['engine_elapsed_ms']) / 1000
                st.caption(f"本轮用时 {elapsed:.1f} 秒 · 已记录模型请求 {metrics['model_calls']} 次 · 业务工具调用 {metrics['leaf_tool_calls']} 次")
                if metrics.get('total_tokens') is not None:
                    st.caption(f"模型服务返回的 Token 用量：{metrics['total_tokens']}。包含各专家和来源复核。")
                else:
                    st.caption('部分请求未返回用量，无法计算完整 Token 总数。')
            st.dataframe(pd.DataFrame([{k: json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v
                                       for k, v in row.items()} for row in result['trace']]), hide_index=True)
    if catalog:
        st.caption('点击回答中的“来源”可定位到对应资料，展开后查看原始数据。')
    evidence_panel(result.get('retrieved_evidence', []), scope, result['answer'])


def confirm_quote(quote, prefix):
    st.warning('本地模拟报价 · 不是真实余票，不会购买或付款')
    st.json(quote, expanded=False)
    if st.button('确认这笔模拟订单', key=prefix + quote['quote_id']):
        result = store.confirm(quote['quote_id'], request_id='ui-' + quote['quote_id'])
        if result.get('status') == 'success':
            st.success('模拟订单已记录。重复确认不会重复扣减库存。')
        else:
            st.warning(result.get('message', '报价已过期或库存发生变化，请重新查询。'))
        st.json(result, expanded=False)


with st.sidebar:
    st.markdown('### 🧭 SmartVoyage')
    st.caption('行知 · 把旅途问题一件件办清楚')
    st.divider()
    st.markdown('**当前行程**')
    st.caption('从对话中整理，方便接着问“那里”或“那后天呢”。如有误，直接在对话中纠正。')
    for label, value in trip_overview(journey.trip):
        st.text(f'{label}：{value}')
    if journey.trip.get('notes'):
        with st.expander('补充需求'):
            st.text(journey.trip['notes'])
    if st.button('开始一段新行程', use_container_width=True):
        st.session_state.messages = []
        st.session_state.journey = TravelSession()
        st.session_state.demo_tickets = []
        st.session_state.page_quote = None
        st.rerun()
    with st.expander('四位助手各自做什么'):
        for agent in config['agents']:
            st.markdown('**' + agent['name'] + '**')
            st.caption(agent['description'])
    with st.expander('服务连接与设置'):
        labels = {'model': '对话模型', 'weather': '天气查询', 'embeddings': '语义向量检索',
                  'places': '高德地点查询', 'live_search': '最新公告搜索', 'tickets': '真实票务查询'}
        for key, label in labels.items():
            status = '公共接口，无需密钥' if key == 'weather' else '已填写配置' if available[key] else '待配置'
            st.caption(f'{label} · {status}')
        st.caption('配置存在不代表调用成功；具体结果以本轮响应为准。')
        st.caption('服务模式：' + ('独立 Agent 服务' if os.getenv('SMARTVOYAGE_A2A') == '1' else '本机协作'))
        fast_response = st.checkbox('快速回答（实验）', value=False)
        st.caption('减少模型思考等待，复杂问题可能更容易遗漏。需要模型支持关闭思考；如报错请关闭。默认沿用本地模型配置。')

st.title('把下一程，想得更周全。')
st.caption('SmartVoyage 行知旅行助手 · 实时天气、旅行安排、官方知识与公告、票务查询')
conversation, weather_tab, knowledge_tab, ticket_tab = st.tabs(['💬 旅行对话', '🌦️ 实时天气', '📚 知识与公告', '🎫 票务演练'])

with conversation:
    if not st.session_state.messages:
        st.info('试试：“明天去陕西西安，天气怎么样？如果下雨，怎么安排？”然后追问“那后天呢？”')
        st.caption('也可以问：“坐飞机能带什么充电宝？” · “故宫预约有哪些要求？”')
    for message_index, message in enumerate(st.session_state.messages):
        with st.chat_message(message['role']):
            if message['role'] == 'assistant':
                render_answer(message['result'], f'reply-{message_index}')
            else:
                st.write(message['content'])
    with st.form('travel_chat', clear_on_submit=True):
        query = st.text_area('这趟旅程，你想解决什么？', placeholder='告诉我你的目的地、时间，或继续刚才的问题。', height=90)
        submitted = st.form_submit_button('发送给旅行助手', disabled=not available['model'], type='primary')
    if not available['model']:
        st.caption('配置对话模型后可使用自动协作；天气和资料检索可独立使用。')
    if submitted and query.strip():
        history = [{'role': m['role'], 'content': m['content']} for m in st.session_state.messages]
        st.session_state.page_quote = None
        st.session_state.messages.append({'role': 'user', 'content': query})
        with st.chat_message('user'):
            st.write(query)
        try:
            with st.spinner('正在选择助手、查询证据并整理回答…', show_time=True):
                request_config = {**config, 'enable_thinking': False} if fast_response else config
                result = asyncio.run(chat(query, history, journey, network=os.getenv('SMARTVOYAGE_A2A') == '1', config=request_config))
                result['response_mode'] = '快速回答（实验）' if fast_response else '环境配置'
            st.session_state.messages.append({'role': 'assistant', 'content': result['answer'], 'result': result})
            st.rerun()
        except Exception as exc:
            st.error(f'本轮未完成（{type(exc).__name__}），请检查服务和配置后重试。')
    if journey.demo_enabled:
        for quote in journey.quotes:
            confirm_quote(quote, 'chat-')

with weather_tab:
    st.subheader('先确认地点，再看当地的天气')
    st.caption('地点来自实时地理查询，支持不同国家和城市。预报范围为目的地当地今天起 16 天。')
    with st.form('find_city'):
        c1, c2 = st.columns([3, 1])
        place = c1.text_input('城市或地点', value='西安')
        country = c2.text_input('国家代码（可选）', placeholder='例如 CN、JP、GB')
        lookup = st.form_submit_button('查询地点')
    if lookup:
        with st.spinner('查询真实地点…'):
            found = call_tool('geocode', {'query': place, 'country_code': country.strip() or None})
        st.session_state.weather_candidates = found.get('candidates', [])
        st.session_state.weather_result = None
        if not st.session_state.weather_candidates:
            st.warning(found.get('message', '未取得有效地点。'))
    candidates = st.session_state.weather_candidates
    if candidates:
        with st.form('weather_dates'):
            chosen = st.selectbox('确认你要去的地点', candidates, format_func=lambda x: x['label'])
            a, b = st.columns(2)
            start = a.date_input('预报开始日期', date.today(), key='forecast_start')
            end = b.date_input('预报结束日期', date.today() + timedelta(days=2), key='forecast_end')
            fetch = st.form_submit_button('获取实时预报', type='primary')
        if fetch:
            with st.spinner('获取当地天气…'):
                st.session_state.weather_result = call_tool('forecast', {'location_id': chosen['location_id'],
                    'start_date': start.isoformat(), 'end_date': end.isoformat()})
    result = st.session_state.weather_result
    if result:
        if result.get('status') == 'success':
            st.markdown('**' + result['location']['label'] + '**')
            st.caption(f"当地时区：{result['timezone']} · 获取时间：{result['retrieved_at']}")
            frame = pd.DataFrame(result['daily']).rename(columns={'date': '日期', 'temperature_2m_min': '最低温 ℃',
                'temperature_2m_max': '最高温 ℃', 'precipitation_probability_max': '降水概率 %',
                'precipitation_sum': '降水量 mm', 'wind_speed_10m_max': '最大风速 km/h'})
            st.dataframe(frame[['日期', '最低温 ℃', '最高温 ℃', '降水概率 %', '降水量 mm', '最大风速 km/h']], hide_index=True, use_container_width=True)
            st.line_chart(frame.set_index('日期')[['最低温 ℃', '最高温 ℃']])
            st.link_button('查看天气数据来源', result['source'])
            st.caption('预报会变化；降水概率不是降雨时长占比。')
        else:
            st.warning(result.get('message', '天气查询未完成。'))

with knowledge_tab:
    st.subheader('让旅行规则与公告有据可查')
    st.caption('内置民航充电宝、铁路退票、故宫预约与限期公告的官方摘要快照。支持上传自己的资料。')
    with st.form('knowledge_search'):
        question = st.text_input('要查什么？', value='坐飞机携带充电宝有哪些要求？')
        a, b = st.columns(2)
        travel_day = a.date_input('资料适用的旅行日期', date.today())
        region = b.text_input('适用地区（可选）', placeholder='例如 CN/北京；留空检索全部地区')
        historical = st.checkbox('同时查历史公告（会标记过期）')
        search = st.form_submit_button('检索知识与公告', type='primary')
    if search:
        try:
            with st.spinner('检索匹配的知识单元…'):
                st.session_state.knowledge_result = call_tool('search_knowledge', {'query': question,
                    'travel_date': travel_day.isoformat(), 'region': region or None, 'include_historical': historical})
        except Exception as exc:
            st.error(f'检索未完成（{type(exc).__name__}），修改资料后请重建索引。')
    result = st.session_state.knowledge_result
    if result:
        st.caption(result.get('notice', result.get('message', '')))
        mode = {'hybrid_bm25_dense_rrf': '关键词与语义混合检索', 'bm25': '关键词检索'}.get(result.get('mode'), '未完成')
        st.write(f"找到 {len(result.get('chunks', []))} 个候选知识单元 · {mode}")
        evidence_panel(result.get('chunks', []))
        with st.expander('查看日期和地区过滤结果'):
            st.json(result.get('filtered_chunks', {}))
    with st.expander('添加资料与重建索引'):
        st.caption('导入资料标为“用户提供、未经核验”。JSON 可保留发布日期与有效期，但不能通过上传自称官方核验。')
        uploaded = st.file_uploader('旅行资料（JSON、Markdown、TXT，最多 2 MB）', type=['json', 'md', 'txt'])
        if st.button('保存这份资料', disabled=uploaded is None):
            try:
                if uploaded.size > 2_000_000:
                    raise ValueError('文件超过2 MB')
                text = uploaded.getvalue().decode('utf-8-sig')
                if uploaded.name.lower().endswith('.json'):
                    documents = json.loads(text)
                    documents = documents if isinstance(documents, list) else [documents]
                else:
                    documents = [{'title': Path(uploaded.name).name, 'text': text}]
                for document in documents:
                    document['id'] = 'upload-' + uuid.uuid4().hex
                    document['source_type'] = 'user_supplied_unverified'
                path = Path(config['knowledge_dir']) / ('upload-' + uuid.uuid4().hex + '.json')
                path.write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding='utf-8')
                st.success('资料已保存，请重建索引后查询。')
            except (ValueError, TypeError, UnicodeError):
                st.error('资料格式不正确，请检查 UTF-8 编码、JSON 格式和文件大小。')
            except OSError:
                st.error('当前资料目录不可写。使用只读部署时，请从本机资料目录添加文件后重建索引。')
        dense = st.checkbox('使用已配置的向量模型（会调用 API）', value=available['embeddings'])
        if st.button('重新切分并建立知识索引'):
            try:
                with st.spinner('正在切分、编码并建立索引…'):
                    report = ingest(config, Embeddings() if dense else None)
                st.success(f"已索引 {report['documents']} 篇资料、{report['chunks']} 个知识单元。")
            except Exception as exc:
                st.error(f'索引未完成（{type(exc).__name__}），旧索引仍保留。')

with ticket_tab:
    st.subheader('票务状态与本地演练')
    if not available['tickets']:
        st.info('真实票务接口尚未接入。提供供应方 API 文档后可适配；目前不会提供虚构的真实票价或余票。')
    enabled = st.checkbox('开启本地模拟票务，理解这不会购买真实车票', value=journey.demo_enabled)
    if enabled != journey.demo_enabled:
        journey.demo_enabled = enabled
        journey.quotes.clear()
        journey.candidates.clear()
        st.session_state.page_quote = None
        st.session_state.demo_tickets = []
    if enabled:
        with st.form('demo_route'):
            a, b, c = st.columns(3)
            origin = a.text_input('模拟出发地', '北京')
            destination = b.text_input('模拟目的地', '西安')
            day = c.date_input('模拟出行日期', date.today() + timedelta(days=1))
            kind = st.selectbox('票务类型', ['train', 'flight', 'attraction', 'concert'])
            seed = st.form_submit_button('为这条路线创建并查询模拟数据')
        if seed:
            try:
                store.seed_demo(origin, destination, day.isoformat())
                tickets = store.query(kind, origin, destination, day.isoformat())['tickets']
                st.session_state.demo_tickets = tickets
                st.session_state.page_quote = None
                journey.update({'departure': origin, 'destination': destination, 'start_date': day.isoformat(), 'end_date': day.isoformat()})
                journey.candidates = tickets
                st.rerun()
            except (ValueError, TypeError):
                st.warning('请填写有效的出发地、目的地和日期。')
        if st.session_state.demo_tickets:
            st.dataframe(st.session_state.demo_tickets, hide_index=True)
            with st.form('prepare_quote'):
                ticket = st.selectbox('选择模拟票', st.session_state.demo_tickets, format_func=lambda x: x.get('service', x['id']))
                quantity = st.number_input('模拟数量', 1, 5, 1)
                prepare = st.form_submit_button('查看模拟报价')
            if prepare:
                result = store.prepare(ticket['id'], int(quantity))
                st.session_state.page_quote = result.get('quote')
                if not result.get('quote'):
                    st.warning(result.get('message', '未取得报价，请重新查询。'))
            if st.session_state.page_quote:
                confirm_quote(st.session_state.page_quote, 'manual-')

st.divider()
st.caption('SmartVoyage · 天气来自 Open-Meteo / GeoNames；资料提供原始来源和日期；工具未接入时会如实说明。')
