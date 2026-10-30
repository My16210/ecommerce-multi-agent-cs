# 部署与运维手册（L1 公网只读演示）

面向**执行部署的人**。目标是一台 Linux 服务器 + 一个域名，跑
[app.py](../app.py) 的只读演示，Caddy 自动签 HTTPS。

**这不是生产客服系统的手册。** 身份、审计、备份策略、监控告警、多副本都还没有
做——边界写在最后一节，别把它当成"上线完成"。

---

## 0. 前置条件

| 项 | 要求 |
| --- | --- |
| 服务器 | Linux x86_64，2 核 4G 起（`docker-compose.prod.yml` 给 app 限了 2G/2 核） |
| Docker | Docker Engine 24+，Compose v2（`docker compose version` 能跑） |
| 域名 | A/AAAA 记录已指向服务器公网 IP |
| 端口 | 80/443 对公网开放；**不要**再对外开放 8501/8502 |
| 模型 Key | 通义千问（路由/汇总/知识库生成）+ 硅基流动（embedding/rerank/图片） |
| 磁盘 | 至少 5G 空闲；知识库索引、会话库、日志都在 `data/` `logs/` |

选区域时的现实约束（部署前定，不是代码问题）：

- 国内大陆服务器：域名需要 ICP 备案，且模型 API 的可达性要在那台机器上实测。
- 香港/海外服务器：通常不需要备案，但访问国内模型 API 的延迟要实测。

---

## 1. 准备代码和数据

```bash
git clone <repo> ecom_cs
cd ecom_cs
```

### 1.1 配置

```bash
cp .env.prod.example .env.prod
# 编辑 .env.prod：DOMAIN、DEMO_INVITE_CODE、QWEN_API_KEY、SILICONFLOW_API_KEY
chmod 600 .env.prod
```

邀请码用随机串，不要手编：

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(24))"
```

检查模板里这几项：

- `DEMO_MODE=1`（compose 也会强制）
- `DEMO_INVITE_CODE` 非空。**留空会让所有人都进不来**，这是有意的 fail closed。
- `DEMO_DAILY_TOKEN_BUDGET` 按预算填；20 万 token 大致是几块钱量级。
- `ENABLE_QUESTION_REVIEW=false`，公网演示不攒用户问题。
- `APP_ENV=prod`（compose 已经写死）。它和 `DEMO_MODE` / `AUTH_ENABLED` 一起
  构成启动闸门：生产环境如果两者都没开，进程会直接拒绝启动，而不是裸奔。
- 如果以后在这套演示前面再加一层自己的反向代理，并且需要按真实客户端 IP 限流，
  才设 `TRUST_PROXY_HEADERS=1`。当前 Caddy 直接终止 TLS，默认的 TCP 对端就够用；
  **不要在没有可信代理时打开它**，否则 `X-Forwarded-For` 可以被客户端随意伪造。

### 1.2 建订单库和向量库

两个一次性任务，复用同一个镜像：

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml run --rm app \
  python main.py --init-db

docker compose --env-file .env.prod -f docker-compose.prod.yml run --rm app \
  python -m ecom.knowledge.cli.ingest
```

说明：

- `--init-db` 读仓库里的种子 CSV，约 6.5 万笔订单，几十秒。
- `ingest` 会调 embedding API，产生一次费用；语料在
  `data/knowledge/documents/`，索引写到 `data/knowledge/vector_store/`。
- 如果这两个命令 OOM，先临时把 `docker-compose.prod.yml` 里 app 的
  `mem_limit` 调到 `4g`，跑完再改回 `2g`。这两个任务不经过 Caddy，不对外。
- 数据目录是 bind mount（`./data`）。在宿主上已经跑过 ingest 的话，
  容器里直接可用，不必重跑。

---

## 2. 启动

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d --build
docker compose --env-file .env.prod -f docker-compose.prod.yml ps
```

第一次启动 Caddy 会申请证书，通常几十秒内完成。看日志：

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml logs -f caddy
docker compose --env-file .env.prod -f docker-compose.prod.yml logs -f app
```

### 验收

```bash
# 1) 健康检查（不经过邀请码）
curl -fsS "https://$DOMAIN/_stcore/health"

# 2) 首页能打开，且要求邀请码
curl -sI "https://$DOMAIN" | head

# 3) 浏览器打开，输邀请码，问一句「订单 52644 到哪了」
```

再验一次只读边界（在服务器上，不经公网）：

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml exec app \
  python main.py "订单 52644 我要退货"
```

预期答复明确说只读演示、不能提交退货申请。

---

## 3. 日常操作

### 3.1 看今天用了多少 token

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml exec app \
  python -c "from ecom import demo; print(demo.tokens_used_today())"
```

账本文件：`data/orchestrator/demo_usage.sqlite`（表 `usage(day, tokens)`）。
按**容器本地日期**分天；容器时区默认 UTC，想按北京时间重置就在 compose 里加
`TZ=Asia/Shanghai`（app 服务）。

### 3.2 换邀请码 / 改预算

```bash
# 编辑 .env.prod 里的 DEMO_INVITE_CODE 或 DEMO_DAILY_TOKEN_BUDGET
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d app
```

`up -d` 会重建 app 容器；Caddy 不用动。换码后旧码立刻失效（所有人都会掉线）。

### 3.3 清过期会话

启动时会清一次，长驻期间不会自己清。外挂 cron：

```cron
# 每天 04:10（服务器时区）
10 4 * * * cd /opt/ecom_cs && docker compose --env-file .env.prod -f docker-compose.prod.yml exec -T app python -m ecom.orchestrator.cli.session purge >> logs/purge.log 2>&1
# 每天 04:20：审计保留期清理（默认 90 天；AUDIT_RETENTION_DAYS<=0 时是空操作）
20 4 * * * cd /opt/ecom_cs && docker compose --env-file .env.prod -f docker-compose.prod.yml exec -T app python -m ecom.auth.cli purge-audit >> logs/purge.log 2>&1
```

保留期是 `.env.prod` 的 `SESSION_TTL_DAYS`（模板是 7 天）。工单存在同一个
`data/orchestrator/sessions.db` 的 `tickets` 表里，**跟着会话一起过期、一起删**；
`purge` 的输出里有 `tickets` 计数。正在排队的工单如果所属会话被清掉，工单也
一并消失——这正是想要的：不能给一个已经不存在的会话留客服入口。

### 3.4 日志

- 应用日志：`logs/`（bind mount），请求 trace 在 `logs/requests.jsonl`——
  **里面是用户说过的话，别整份贴到公开地方**。
- 容器日志：`docker compose ... logs app|caddy`。

---

## 4. 排障

| 现象 | 先查什么 |
| --- | --- |
| 打开页面显示"没有配置邀请码" | `.env.prod` 的 `DEMO_INVITE_CODE` 是否为空；改完有没有 `up -d app` |
| 邀请码对却进不去 | 浏览器里是不是还留着旧码；换一个会话/清 cookie 再试；看 app 日志 |
| 一直"今天的演示额度已经用完" | 查上面 3.1 的用量；要临时放开就改 `DEMO_DAILY_TOKEN_BUDGET` |
| 429 / "问得有点快" | 限流，等一分钟；或调 `DEMO_RATE_LIMIT_PER_MINUTE` |
| 登录账号问答 429 "请求过于频繁" | 账号级限流（默认 30 次/分钟）；等一分钟，或调 `CHAT_RATE_LIMIT_PER_MINUTE` |
| 登录账号 429 "个人 token 额度已经用完" | 这个账号当天额度用完（默认 20 万）；查 `CHAT_USAGE_DB_PATH` 里的 `principal_usage`，或调 `CHAT_DAILY_TOKEN_BUDGET` |
| 登录账号 429 "全站 token 额度已经用完" | 所有账号当天合计达到部署上限（默认 100 万）；查同一张表按天求和，或调 `CHAT_GLOBAL_DAILY_TOKEN_BUDGET`（<= 0 关闭） |
| 登录账号问答 503 "用量账本暂时不可用" | 故意的 fail closed：查 `data/orchestrator/chat_usage.sqlite` 的权限/磁盘。修好之前不会继续花 token |
| 退出登录后仍能用同一枚 token | 如果那是长期 API token，`/auth/logout` 不会吊销它；界面已经会提示。让管理员跑 `python -m ecom.auth.cli rotate <账号>` |
| 证书申请失败 | 域名 A/AAAA 是否指向本机；80/443 是否被占用；`logs caddy` 里的 ACME 报错 |
| 502 | `docker compose ps` 看 app 是否 healthy；`curl` 容器内 `/_stcore/health` |
| 答案全是"知识库查不了" | `SILICONFLOW_API_KEY` 是否有效、服务器能不能访问它的 API |
| 用户说转了人工但没人接 | 先看 `/health` 的 `auth` 和 `demo`：演示模式 / 未开身份层时**本来就没有人工**，答复会明说。开了身份层却没人接，是队列里真的没人（`GET /tickets?status=open` 用 agent token 看；`cli.session list` 的概览里有未关闭工单数） |
| 客服台按钮不出现 | 登录账号角色是不是 `agent` / `admin`；手工粘贴的普通 token 不显示按钮，这是有意的 |
| 图片识别失败 | `VISION_*` 配置；控制台模型是否仍然上架；`logs/app.log` 里有原因 |
| 数据目录权限拒绝 | 容器以 uid 1000 运行：`sudo chown -R 1000:1000 data logs` |
| 磁盘满 | 检查 `data/analytics/output`、`logs/`、旧会话；演示模式不跑数据分析，通常不会涨很快 |

看容器内真实健康状态（深检查，不走公网）：

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml exec app \
  python main.py "订单 52644 到哪了" -v
```

---

## 5. 更新、备份、回滚

### 更新代码

```bash
git pull
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d --build
```

数据库 schema 目前没有迁移机制。如果新版本改了 `orders` / `sessions` 表结构，
要先备份再重建（订单库可以 `--init-db` 重灌；会话库不能）。

### 备份

最少备份这些（都在 `data/`）：

- `data/orchestrator/sessions.db` —— 用户会话 + 转人工工单（`tickets` 表），**不可重建**
- `data/orchestrator/demo_usage.sqlite` —— 当日用量，丢了预算会重新计
- `data/orchestrator/auth.db` —— principal / 登录会话 / 审计；丢了所有人都要重新建号。审计默认保留 90 天，启动时自动清一次，长驻进程靠 `auth.cli purge-audit`
- `data/orders.sqlite` —— 可重建（`--init-db`），但备份更省事
- `data/knowledge/vector_store/` —— 可重建（`ingest`），体积较大，按需

```bash
tar czf backup-$(date +%F).tgz data/orchestrator data/orders.sqlite
```

备份文件里同样有用户会话，按敏感数据保存。

### 回滚

```bash
git checkout <上一个可用的 tag/commit>
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d --build
```

配置回滚：恢复上一份 `.env.prod` 后 `up -d app`。

### 停止 / 彻底移除

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml down
# 连 Caddy 的证书卷一起删（下次启动会重新申请）：
docker compose --env-file .env.prod -f docker-compose.prod.yml down -v
```

`down` **不会**删 `./data` 和 `./logs`（bind mount）。要清数据得手动删。

---

## 6. 这一层**没有**做什么

以下每一条都是"真上线"必须补的，不能因为 Demo 能跑就当成已完成：

1. **身份与权限**：`AUTH_ENABLED` 提供 token / 角色 / 审计 / 会话归属 / 历史回看，以及每个账号的问答限流和每日 token 预算；公网演示仍用共享邀请码（没开它），也没有产品级用户体系（注册、找回密码、MFA、OIDC、多租户）。问答限流是进程内的，多副本各算各的；每日预算落 SQLite，跨副本仍有效。
   转人工工单是**演示级闭环**：5 秒轮询、没有实时推送、SLA、排队超时、在线状态和客服技能分组；公网演示里这个能力整个是 404。
2. **写操作**：退货申请在演示里被硬关掉，没有"演示账号可以写"的旁路。
3. **数据分析**：整条路被停用；沙箱不是强隔离，不要在公网打开它。
4. **备份与迁移**：没有自动备份、没有 schema 迁移。
5. **多副本 / 高可用**：会话和预算都在本地 sqlite，单副本。
6. **监控告警**：只有容器健康检查；没有 token 成本告警、没有错误率面板。
7. **合规**：没有隐私政策、没有数据删除请求流程；会话保留期是唯一的兜底。
8. **速率限制的分布式一致性**：限流是进程内的，多副本各算各的。

这些缺口和优先级见 [上线清单.md](上线清单.md)。
