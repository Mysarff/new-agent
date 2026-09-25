import asyncio
import json
from pathlib import Path

import streamlit as st

from opsatlas.cli import chat
from opsatlas.config import load_config
from opsatlas.model import Embeddings
from opsatlas.rag import Retriever, build_index

st.set_page_config(page_title='知维 OpsAtlas', page_icon='🧭', layout='wide')
config = load_config()
st.title('知维 OpsAtlas')
st.caption('软件运维与故障排查 · 专家协作 · 可追溯知识')
st.info('内置知识为 AI 生成的虚构演练资料；实际故障需要结合你的环境、日志与授权流程判断。')

if 'messages' not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader('工作方式')
    mode = st.radio('选择模式', ['知识检索（无密钥）', 'Agent 对话（需要模型）'])
    st.caption('检索模式只展示原文片段，不模拟模型回答。')
    for agent in config['agents']:
        with st.expander(agent['id']):
            st.write(agent['description'])
    if st.button('清空对话'):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    dense = st.checkbox('启用向量模型重建索引（可能产生 API 费用）')
    if st.button('重建知识索引'):
        try:
            with st.spinner('切分文档并建立索引…'):
                report = build_index(config, Embeddings() if dense else None)
            st.success(f"已索引 {report['documents']} 篇文档、{report['chunks']} 个知识单元")
        except Exception as exc:
            st.error(f'索引失败：{type(exc).__name__}。请检查知识文件和模型配置。')
    st.caption('添加资料：将 Markdown 或 TXT 放入 knowledge/，再重建索引。')

if not Path(config['index_path']).exists():
    build_index(config)


def render_result(result):
    st.markdown(result['answer'])
    for warning in result.get('warnings', []):
        st.warning(warning)
    evidence = result.get('retrieved_evidence', [])
    if evidence:
        with st.expander(f'查看原文与来源 · {len(evidence)} 条'):
            for item in evidence:
                st.markdown(f"**{item['id']} · {item.get('title', item['source'])}**")
                st.caption(f"来源：{item['source']} | 类型：{item.get('metadata', {}).get('provenance', 'unknown')}")
                # Quoted documents remain data; never render as unsafe HTML.
                st.text(item.get('text') or json.dumps(item.get('data', {}), ensure_ascii=False, indent=2))
    if result.get('trace'):
        with st.expander('查看 Agent 与工具调用记录'):
            st.dataframe(result['trace'], use_container_width=True)


for message in st.session_state.messages:
    with st.chat_message(message['role']):
        if message['role'] == 'assistant':
            render_result(message['result'])
        else:
            st.write(message['content'])

query = st.chat_input('例如：发布后网关出现 502，应该先查什么？')
if query:
    history = [{'role': m['role'], 'content': m['content']} for m in st.session_state.messages]
    st.session_state.messages.append({'role': 'user', 'content': query})
    with st.chat_message('user'):
        st.write(query)
    with st.chat_message('assistant'):
        try:
            with st.spinner('正在处理…'):
                if mode.startswith('知识检索'):
                    embedding = Embeddings()
                    retrieved = Retriever(config, embedding if embedding.model else None).search(query)
                    result = {'answer': '以下是检索候选原文，请展开来源查看。' if retrieved['chunks'] else '知识库没有找到匹配证据。',
                              'retrieved_evidence': retrieved['chunks'], 'warnings': [retrieved['notice']]}
                else:
                    result = asyncio.run(chat(query, config, history))
            render_result(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result['answer'], 'result': result})
        except Exception as exc:
            st.error(f'处理未完成（{type(exc).__name__}）。请检查 .env、网络和索引；修改知识文件后需要重建索引。')
