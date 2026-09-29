# Wayloom 票务接口与模拟演练

## 真实供应方：仅查询，不下单

环境变量：

- `WAYLOOM_TICKET_BASE_URL`：供应方或适配服务的 HTTPS 根地址，例如 `https://your-provider.example/api/v1`。
- `WAYLOOM_TICKET_API_KEY`：供应方需要认证时填写；以 `Authorization: Bearer ...` 请求头发送。此密钥不应提交到仓库。

项目没有假定任何商业供应方已经接入。用户提供接口文档后，可由适配服务转换成下面的统一契约；配置任意供应方 URL 不代表其已经兼容该契约。未配置时返回 `unavailable`，不会自动转为虚构票价。仅本机调试允许 HTTP。

请求为 `GET {BASE_URL}/tickets`，查询参数为 `kind`、`departure`、`arrival`、`date`。城市由用户输入，不使用城市白名单；日期是有效的 `YYYY-MM-DD`。当前是否售票、支持哪种票、城市别名和车站映射由供应方决定。种类可采用 `train`、`flight`、`attraction`、`concert` 等供应方支持的值，客户端不硬编码种类列表。

```json
{
  "tickets": [
    {
      "id": "supplier-ticket-id",
      "kind": "train",
      "departure": "用户请求的出发地",
      "arrival": "用户请求的目的地",
      "date": "2031-04-23",
      "service": "供应方返回的班次",
      "seat": "供应方返回的席别",
      "price": 123.45,
      "currency": "CNY",
      "remaining": 3,
      "source": "供应方名称及来源说明",
      "booking_url": "https://your-provider.example/booking/example"
    }
  ]
}
```

这是字段示例，不是真实报价。`booking_url` 可省略；其余字段必填。`remaining` 未知时返回 `null`，不能用假库存填充。`price` 为有限非负数，`currency` 是三位大写货币代码。返回的 `kind`、`departure`、`arrival`、`date` 必须与查询完全对应，适配层可把供应方规范名称映射回查询值。所有记录必须有唯一票号，每次最多 500 条。

调用签名：

```python
await query_tickets(kind, departure, arrival, date, client=None,
                    base_url=None, api_key=None, timeout=15.0)
```

成功返回 `status="success"`、`source_kind="live"`、`tickets`、`fetched_at` 和来源地址；没有记录返回 `no_data`。超时、HTTP 拒绝、网络错误、数据字段异常会明确返回 `provider_error`，不把失败变成有效证据。默认超时 15 秒，不跟随重定向。调用者参数错误抛出 `ValueError`。

此模块不存在真实下单/支付/出票接口。供应方购票链接只供用户打开并在供应方页面自行确认；不能把查询成功描述为订票成功。

## 本地模拟：用户主动开启，页面单独确认

`DemoBookingStore(path, max_quantity=5, quote_ttl_seconds=300)` 构造时只建表，不初始化票。页面只有在用户点击“开启模拟演练”后，才调用 `seed_demo(departure, arrival, date)`。演练线路和日期来自本次输入，价格、班次和库存采用固定虚构模板。重复开启相同线路不会补回已消耗的库存。

- `query(kind, departure, arrival, date)`：查询已开启的模拟数据；返回 `tickets`（以及兼容别名 `data`）。
- `prepare(ticket_id, quantity)`：创建报价，返回 `quote` 与 `requires_confirmation=true`。数量必须为整数并符合配置上限，报价最多 5 分钟有效，不锁库存、不扣款。
- `confirm(quote_id, request_id)`：仅由独立的页面确认按钮调用，**不得注册为 LLM/MCP 工具**。同一事务内重查报价有效期、价格、币种、行程、库存，再创建本地模拟订单。

同一请求或同一报价的重复确认返回原模拟订单，库存只扣一次。同一请求 ID 用于另一个报价会报错。两个不同报价争抢余票时，数据库写事务保证不超卖。所有模拟记录、报价、结果均带 `source_kind="synthetic"` 和中文模拟说明，模拟订单编号以 `SIM-` 开头。此订单没有实际履约能力。

请勿把演练模板当作实时机票、火车票或景区票数据。真实接口失败时也不会自动调用 `seed_demo`。
