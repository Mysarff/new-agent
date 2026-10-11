# MySQL 部署、迁移与验证

MySQL 保存模拟票、报价、订单和幂等请求。BM25/向量检索仍负责从公告正文中找到相关证据，当前知识索引仍为 JSON + NumPy；本次未把知识库、会话或用户账户迁入 MySQL，也没有新增 FAQ 分支。更换业务数据库不能代替检索，也不能直接减少模型推理耗时。

## 新部署

先安装 Python 3.12 项目依赖，并启动 Docker Desktop 或 Docker Engine：

```powershell
python -m pip install -r requirements.txt
python -m wayloom.database setup
docker compose up -d mysql --wait
python -m wayloom.database init
python -m wayloom.database status
python -m wayloom.stack
```

`setup` 首次复制 `.env.example`（仅当 `.env` 不存在），补齐缺失的 MySQL 字段。应用密码随机生成，root 密码单独保存在被 Git 忽略的 `var/mysql-root-password`，通过 Docker secret 仅挂载到 MySQL 服务，不传给页面或 MCP。已有 API Key、非空数据库配置和 root 密码文件均保留。不要把 `.env` 或 `var` 提交到仓库。

默认数据库/账户为 `wayloom`，宿主机连接为 `127.0.0.1:3309`，容器间连接为 `mysql:3306`。`compose.yaml` 的 `wayloom-mysql-data` 是独立持久卷，与旧 `smartvoyage-data` 应用卷分开。数据库只绑定回环地址；修改端口后重建容器即可，数据卷保留。

如需整套容器启动：

```powershell
python -m wayloom.database setup
docker compose up -d --build --wait
```

访问 http://127.0.0.1:8501 。Compose 等待数据库通过应用账户连接健康检查，再运行建表和页面。页面为本机专家模式，独立 A2A 专家用宿主机 `wayloom.stack`。Compose 环境变量优先于 `.env`，避免容器错误连接到自己的 localhost。

停止服务用 `docker compose stop`；移除容器但保留数据用 `docker compose down`。**不要使用 `down -v` 保存数据**，该选项会删除持久卷。MySQL 镜像的账户和密码初始化仅对空数据卷生效；更改 `.env` 不会自动轮换已存在库的密码，应通过数据库管理操作更新并同步配置，不能靠删除卷解决。

## 迁移既有 SQLite

先停止本项目的页面和专家服务，防止迁移期间仍有写入。只启动 MySQL、初始化空表，再导入：

```powershell
docker compose up -d mysql --wait
python -m wayloom.database init
python -m wayloom.database migrate-sqlite --source var/demo_booking.sqlite3
python -m wayloom.database status
python -m wayloom.stack
```

导入以只读方式打开 SQLite，并读取一致快照；旧文件保留。目标四张业务表必须为空，已有记录会拒绝导入，不覆盖、合并或重置库存。票号、报价、订单、请求 ID 和原到期时间保留。导入记录在一个事务中提交，失败全部回滚。请保留迁移前的 SQLite 备份及其存在的 WAL 文件；迁移后业务统一写 MySQL，两个库不会同步。

显式离线兼容模式可设置 `WAYLOOM_BOOKING_BACKEND=sqlite`；仅在有意使用旧库/离线测试时设置。MySQL 连接失败不会自动切换。设置错误或 MySQL 不可用时，票务返回 `unavailable`，页面与天气/知识模块仍能打开。

## 并发与失败处理

- 查询以 `(kind, departure, arrival, date, price_cents, id)` 联合索引检索路线。
- 准备报价只保存快照，不锁住库存等待用户。
- 人工确认在短事务内依次锁定报价和库存行，重查价格、行程、库存、有效期后扣库存。
- 请求 ID、报价 ID 的唯一约束防止重复订单；重放请求绑定到原报价，不能挪到别的报价。
- 库存更新、订单、请求绑定和报价完成状态一并提交或回滚；唯一键冲突、死锁、锁等待超时最多重试三次整个确认事务。
- 不缓存库存、报价确认或订单结果；连接超时3秒、读写超时10秒、锁等待5秒，错误不输出密钥或供应方响应正文。
- 不在每次页面刷新或工具发现时建表。`init` 是显式可重复的空表初始化命令，不重置现有业务数据；将来修改表结构需要单独的版本化迁移。

行锁语义参考 [MySQL 8.4 locking reads](https://dev.mysql.com/doc/refman/8.4/en/innodb-locking-reads.html)，启动依赖参考 [Docker Compose depends_on](https://docs.docker.com/reference/compose-file/services/#depends_on)。

## 验证

```powershell
# 普通离线回归；MySQL 集成测试显示 skipped
python -m unittest discover -s tests -p 'test_travel_*.py' -v
# 真实 Docker MySQL，创建随机临时库，结束后只删除临时库
python scripts/test_mysql.py
```

真实库测试继承报价/库存基础用例，另测12个并发请求、不同报价共用请求 ID、故障回滚、迁移、真实 MCP 子进程及 Streamlit 重复确认。脚本使用项目容器内的 root secret 创建 `wayloom_test_<随机标识>`，应用账户执行测试，业务库不参与。CI 使用独立 `wayloom_test` 服务库；直接启用测试时会检查测试库名称，拒绝清空业务库。

这些是模拟业务一致性检查，不是模型回答准确率、生产吞吐量或真实出票成功率。真实模型延迟仍应按 `docs/evaluation.md` 的固定问题集分别统计 P50/P95、模型调用次数和质量判据，再决定是否减少协调器/专家/复核调用。当前资料规模很小，应优先增加有来源、有日期的有效资料与人工标注评测，不必为使用 MySQL 增加向量数据库。
