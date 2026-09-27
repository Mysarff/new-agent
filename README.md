# SmartVoyage 行知旅行助手

围绕真实旅行问题工作的 Agent：查询目的地天气、根据同行人与天气调整行程、检索景区公告和旅行规则，并在已接入供应方时查询票务。支持连续对话，例如先问“北京明天会下雨吗”，再问“那后天呢”“换成西安，带孩子有什么室内活动”。

**天气来自真实公共接口，城市与日期不写死；模型根据上下文和专家能力选择任务；知识回答保留来源和有效期。** 没有配置的接口会明确报告不可用，不用固定样例冒充查询成功。

## 四个 Agent 各自解决什么问题

| Agent | 负责的实际问题 | 可以使用的工具 | 当前边界 |
| --- | --- | --- | --- |
| 天气 | 某地某天是否适合出行；比较多个城市或日期的温度、降水和风力 | 地名候选查询、真实天气预报 | 目的地当地今天起16天内；不是实况保证或官方灾害预警 |
| 行程 | 根据目的地、日期、同行人和偏好找场所，调整雨天与亲子安排 | 高德地点检索、联网资料搜索；使用上游天气证据 | 高德和搜索需要各自的 Key；没有交通工具时不声称算出了最优路线 |
| 公告与知识 | 充电宝乘机限制、景区预约、列车停运退票；核对规则适用日期 | 本地 RAG、最新公开资料搜索 | 本地资料是快照；“今天是否临时关闭”仍须联网核验 |
| 票务 | 按线路和日期查接入方的价格/库存；演示“选第二个，订两张”的上下文 | 真实票务查询适配器、明确启用的本地演练 | 没有预装真实出票平台；不执行真实购买或支付 |

协调器从 `config/travel.json` 读取能力描述，由支持工具调用的模型决定调用哪个专家，可以先天气、再行程、再公告，也可以只回答用户需要的一项。四个角色共用配置的模型，区别在任务提示、工具权限与可独立运行的服务；不是四个经过独立专业训练的模型。

每轮先用独立的简短上下文提示词，只向模型提供 `update_trip_context` 工具，要求提交完整七字段快照：出发地、目的地、开始日期、结束日期、人数、偏好和备注。没有变化的字段保留旧值，未知标量填 `null`，未知偏好填 `[]`；不允许用空对象跳过这一步。完成状态提取后，第二阶段协调器再根据能力描述动态选择专家。这一步固定的是先整理上下文的顺序，字段值及后续专家选择仍由模型结合语境生成。

会话保存用户明确给出的目的地、出发地、日期、人数、偏好和最近票务候选。改变路线或日期会清空旧票务候选与报价，减少“第二个”指向上一条行程的错误。缺少必要条件时应追问，不能擅自补出人数或目的地；具体理解质量仍取决于模型。

当本轮上下文已明确开始和结束日期时，天气工具调用还必须落在这个日期窗口内；模型擅自扩大查询范围会收到 `date_mismatch`，需修正后重试。协调处理默认受 `run_timeout=240` 秒限制，包含上下文提取与专家等待；超时会标明本轮未完成并保留已取得的来源。这个时限不包含进入协调器前的 MCP 初始化与退出清理。

## 先运行，再按需要接 API

要求 Python 3.12。以下为 Windows PowerShell，新建本地环境：

```powershell
git clone https://github.com/Mysarff/new-agent.git
cd new-agent
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m SmartVoyage.main ingest
python -m SmartVoyage.stack
```

已有 `.env` 时保留自己的配置，不要用示例覆盖。Linux/macOS 激活命令为 `source .venv/bin/activate`，首次复制用 `cp .env.example .env`。

完整启动器打开 [本机页面](http://127.0.0.1:8510)，并启动四个独立专家服务，默认端口为8611—8614。用 Ctrl+C 停止启动器及其创建的子进程。端口可在配置中调整；页面端口可用 `python -m SmartVoyage.stack --port 8511` 指定。

只启动单进程页面也可以：

```powershell
streamlit run app.py --server.address 127.0.0.1
```

此模式默认页面为 [8501 端口](http://127.0.0.1:8501)，专家在当前应用进程内执行，工具仍通过真实 MCP 子进程通信。未配置模型时，可使用页面中的直接天气与知识检索功能；这些操作不会假装经过 LLM 智能路由。自然语言 Agent 对话必须先配置模型。

### API 配置清单

只在本机 `.env` 保存密钥，不要提交到 GitHub。项目启动时读取这个文件，修改后需重启服务。

| 功能 | 配置项 | 是否必需 |
| --- | --- | --- |
| 自然语言 Agent、上下文理解、动态路由 | `SMARTVOYAGE_BASE_URL`、`SMARTVOYAGE_API_KEY`、`SMARTVOYAGE_MODEL` | Agent 对话必需；模型须支持 Chat Completions 的 `tools/tool_calls` |
| 地名与天气 | Open-Meteo 公共接口 | 当前实现无需 API Key，需要网络连接 |
| 景点、餐饮、室内场所 | `AMAP_API_KEY` | 可选，高德 Web 服务 Key，需开通地点搜索权限 |
| 最新公告与公开资料 | `TAVILY_API_KEY` | 可选；未配时不能声称已联网核实最新公告 |
| 本地词法检索 | 无 | 无密钥即可运行；不产生模型回答 |
| 向量混合检索 | `SMARTVOYAGE_EMBEDDING_MODEL`，可另配 `SMARTVOYAGE_EMBEDDING_BASE_URL/API_KEY` | 可选；地址与密钥为空时沿用对话模型配置 |
| 接入已有票务服务 | `SMARTVOYAGE_TICKET_BASE_URL`，以及供应方需要的 `SMARTVOYAGE_TICKET_API_KEY` | 可选；服务必须满足下文的查询契约 |

对话配置示例，以下值只是占位内容：

```dotenv
SMARTVOYAGE_BASE_URL=https://your-provider.example/v1
SMARTVOYAGE_API_KEY=your-own-key
SMARTVOYAGE_MODEL=your-tool-calling-model
```

这是可替换的 OpenAI 兼容接口适配器，不限定某一家模型公司，也不保证所有兼容接口都实现相同的工具能力。模型、搜索、地点和向量服务可能有调用费用或配额。天气服务能力依据 [Open-Meteo 预报文档](https://open-meteo.com/en/docs)与[地名接口文档](https://open-meteo.com/en/docs/geocoding-api)；商用或高频使用请核对供应方的现行授权与用量方案。

### 命令行操作

```powershell
# 无密钥建立词法索引
python -m SmartVoyage.main ingest

# 本地资料检索，地区用显式国家/城市层级
python -m SmartVoyage.main search "故宫预约规则" --date 2026-10-01 --region "CN/北京"

# 查看真正从 MCP 服务发现的工具
python -m SmartVoyage.main discover

# 配好模型后，在当前进程运行专家
python -m SmartVoyage.main chat "查北京明天的天气，说明数据来源"

# 配好模型后，只启动独立专家，再在另一个终端发起 A2A 对话
python -m SmartVoyage.stack --no-ui
python -m SmartVoyage.main chat "充电宝坐国内飞机有什么限制？" --network
```

命令行 `chat` 每次为新会话；连续指代演示使用网页会话。示例日期用于说明格式，实际旅行请填写真实日期。

## 可以怎样演示

1. 在直接天气页面输入任意能被地名服务识别的地点，确认候选后查询日期。尝试同名城市、多个日期和超过预报范围的日期，查看明确的候选与错误状态。
2. 配好模型后说“明天去北京，带一个孩子，查天气并给雨天备选”，然后说“换成西安，日期改后天”。展开执行记录，检查实际选择的专家、工具参数与更新后的行程。
3. 问“故宫周一能去吗，需要预约吗”。查看引用、来源链接和核验日期；遇到最新开放状态，在没有 Tavily 时应说明无法确认。
4. 在票务演练区主动创建某条路线的模拟数据，查询后说“第二个，两张”。Agent 只能准备模拟报价，用户核对后通过页面单独确认。演练价格、库存和订单始终标注虚构。

第2项的真实地点推荐需要高德或搜索配置，第3项的最新公告核验需要搜索配置。缺少这些配置时仍可演示天气、已有知识检索与明确的不可用提示，不能将受限演示称作全部业务已接通。

回答中的“来源1、来源2”可点击定位到本条回答的资料卡片；天气卡片展示接口实际返回的温度等数值，地图卡片展示名称、地址和分类。内部证据编号保留在执行记录及报告中，不在正文显示。温度采用普通 `℃` 文本，并兼容修正历史回答中的常见 LaTeX 摄氏度写法；只改变显示形式，不改温度数值。

左栏优先显示当前行程：单日日期合并，未知人数和偏好标为“未提供”，补充需求和服务配置折叠展示。“已填写配置”不等于接口健康检查成功。行程状态来自对话提取，用户可直接在对话中纠正。

本轮查询涉及地图地点时，最终草稿额外经过一次模型来源复核：对照工具字段删去缺少证据的具体场所描述，明确室内条件、开放和预约信息的缺口。复核最多等待 `evidence_review_timeout=90` 秒，且仍受整轮240秒时限约束；失败则不展示原草稿，保留工具资料供查看。这会增加一次模型调用及相应耗时/费用。复核是质量控制，仍可能漏检，不能视为独立事实核查或准确率保证；历史回答不会自动付费重查，会显示未复核提示。

## 旅行知识与 RAG

默认 `travel_knowledge/` 放置4篇自行归纳的官方资料摘要，本次资料核验日期为 **2026-09-26**：

| 资料 | 原始出处 | 时效处理 |
| --- | --- | --- |
| 境内航班充电宝的3C与召回限制 | [中国民航局](https://www.caac.gov.cn/XWZX/MHYW/202506/t20250626_227805.html) | 记录2025-06-28起适用；不是完整航空行李规则 |
| 故宫常规预约与参观入口 | [故宫订票须知](https://www.dpm.org.cn/subject_booking/) | 原页未明确本段有效期，保留未知状态 |
| 国铁列车晚点、停运退票 | [中国铁路12306](https://kyfw.12306.cn/otn/gonggao/saleTicketMeans.html) | 摘记第51、52条；不覆盖所有票种或自愿退票 |
| 故宫2026-07-27特别开放 | [单日活动公告](https://www.dpm.org.cn/announce_detail/379398.html) | 真实历史资料，默认当前检索排除 |

摘要不是实时公告库，也不代表已经覆盖全国所有景区政策。历史公告详情页直接访问曾超时，使用官方页面搜索提取内容核对，文件保留获取限制。核验时间不等于持续有效，查询不到不等于没有公告。

把自己的 JSON、Markdown 或 TXT 放入 `travel_knowledge/`，再运行 `python -m SmartVoyage.main ingest`。JSON 推荐包含 `id/title/text/source/source_type/published_at/checked_at/valid_from/valid_to/regions/tags`；字段与导入说明见 `travel_knowledge/FORMAT.rst`。Markdown/TXT 默认标为未核实资料，不会因正文自称“官方”而自动获得可信标签。

文档按标题与滑动窗口切成知识单元，默认650字符、100字符重叠，保留来源、原文位置和稳定 `K-` 引用ID。短摘要可能只有一个单元，不为增加数量强行拆碎。中文 BM25 使用双字词元，属于词法检索；需要接模型生成带依据的回答时，才是完整的 RAG 问答流程。

日期和地区过滤发生在检索排名之前。过期资料仅在显式历史检索时返回，并标明过期；未来才生效的资料不能当作较早行程日的证据。`CN/北京` 可同时匹配全国与北京规则；程序不会通过写死城市名单猜测国家。没有填写有效期代表未知，不能解释为永久有效。

可选语义向量检索：

```dotenv
SMARTVOYAGE_EMBEDDING_BASE_URL=https://your-provider.example/v1
SMARTVOYAGE_EMBEDDING_API_KEY=your-own-key
SMARTVOYAGE_EMBEDDING_MODEL=your-embedding-model
```

```powershell
python -m SmartVoyage.main ingest --dense
```

使用兼容 `/embeddings` 的真实服务生成向量，与 BM25 通过 RRF 融合。更换资料、切分参数、向量提供商、模型或维度后需要重建；索引不匹配会报错。构建失败保留先前索引。实现使用本地 JSON 与 NumPy，面向小型资料库。向量单元测试使用 mock 验证机制，不代表真实语义召回已达到某个效果。

## 票务接入边界

项目提供**查询适配契约**，没有把任意 Key 填进去就能查询12306、携程或航空公司的通用接口。接入你持有权限的供应方时，需要实现或提供一个符合契约的后端：

```text
GET <SMARTVOYAGE_TICKET_BASE_URL>/tickets
查询参数：kind、departure、arrival、date（YYYY-MM-DD）
可选认证：Authorization: Bearer <SMARTVOYAGE_TICKET_API_KEY>
响应：{"tickets": [...]}
```

每张票需包含 `id/kind/departure/arrival/date/service/seat/source/price/currency/remaining`，可附 `booking_url`。币种为三位大写代码；未知库存应为 `null`，不能填0；结果的票种、路线和日期须与查询一致。完整校验和边界见 `docs/architecture.md`。

未配置、供应方报错或数据不匹配时，系统明确报告未取得可用票务，不回退为假价格。真实查询不会创建订单。演练只在本地 SQLite 保存虚构库存与订单，报价最长5分钟有效；确认由页面处理，确认操作不在模型的 MCP 工具目录中。

## 架构、扩展与验证

完整模式是 **网页协调器 → HTTP A2A 专家 → MCP 工具 → 外部 API / 本地知识库**。项目锁定 `python-a2a==0.5.4`，使用其 `AgentCard/Task/Message` 数据模型实现 `tasks/send` JSON-RPC 交互。它是本项目的明确协议实现，不声称覆盖最新版 A2A 规范的全部方法或流式任务生命周期。MCP 使用 `mcp==1.18.0`，默认 stdio，也支持可信配置的 Streamable HTTP 服务。

| 入口 | 职责 |
| --- | --- |
| `config/travel.json` | 专家描述、权限、端口、MCP 服务与调用预算 |
| `SmartVoyage/engine.py` | 模型路由、上下文、工具循环、证据与引用 |
| `SmartVoyage/a2a.py`、`stack.py` | 独立专家 HTTP 服务、进程启动与停止 |
| `SmartVoyage/mcp_client.py`、`tools.py` | MCP 工具发现、参数校验、工具调用 |
| `SmartVoyage/weather.py`、`travel_web.py` | 天气、地点、公开资料接口 |
| `SmartVoyage/knowledge.py` | 切分、日期/地区过滤、BM25与可选向量 |
| `SmartVoyage/tickets.py` | 只读真实票务适配器、本地确认式演练 |
| `app.py` | 多轮对话、直接查询、来源展示和演练确认 |

添加专家可修改配置的能力描述和工具权限；添加 MCP 工具后由运行时发现其参数，不用往协调器添加关键词分支。真正的新能力仍需实现相应工具。参数约束、权限、预报窗口与确认流程属于必要业务边界，不是把城市或答案写死。

```powershell
python -m unittest discover -s tests -v
```

自动化测试检查天气位置与日期、资料有效期与切分、上下文和权限、工具协议以及票务演练的报价与重复确认。受控模型/向量夹具用于验证流程；通过这些测试不等于真实 LLM 回答质量或外部供应商已验收。真实模型、各外部 API 与向量服务应按各自配置另做联网验证，失败需保留原始状态。

### 真实模型与 A2A 联调验收

先启动 `python -m SmartVoyage.stack`（或 `--no-ui`），保持四个专家服务运行，再在另一个终端执行：

```powershell
# 只列出检查内容，不调用模型或网络
python -m SmartVoyage.verify

# 使用真实模型，经独立 A2A 服务与 MCP 工具执行；可能产生模型/API费用
python -m SmartVoyage.verify --live --network
```

默认报告写入 `var/live_verification.json`，逐步保存问题、回答、工具记录、引用、上下文及每项判据。省略 `--network` 时仍是真实模型和 MCP 调用，但专家在当前进程执行；`--output` 可指定其他报告路径。

这组检查覆盖“西安明天 → 那后天呢 → 同一天改成东京”的多轮日期与目的地继承、充电宝规则的 RAG 引用，以及未接入真实票务时如实说明不可用。天气判据核对工具执行、地点/国家/时区/日期、实际数值与引用；网络模式还检查 A2A 任务完成。已配置真实票务时，不适用的“未接入”用例会跳过并使汇总结果标为部分验收。

报告中的通过只代表这些执行、上下文和证据判据通过，不是对全部回答逐句事实核查，也不是所有场景的长期成功率评估。该命令不单独验收高德、Tavily、所有专家组合或向量召回质量；是否启用向量取决于当前索引。读取报告的最终 `status` 和各步骤判据判断完成情况，不把仍在执行、跳过或失败的报告描述为全部通过。

当前按本机可信环境设计，默认服务只监听127.0.0.1，没有多租户登录、文档访问控制或生产支付能力。网页会话各自保存对话，知识库及模拟库存为本地共享资源。检索资料可能发送给配置的模型提供商，导入私有资料前需确认其使用要求。更多协议和数据流说明见 `docs/architecture.md`。
