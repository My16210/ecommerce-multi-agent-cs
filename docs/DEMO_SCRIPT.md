# 面试演示脚本

> 目标：**8–10 分钟**内让面试官看到"路由 → 并行 worker → 汇总 → 来源"这条链，
> 以及"这不是只能跑一次的 demo"（有评测、有测试、有边界）。
>
> 所有命令都是 PowerShell，先 `cd D:\worksoftspace\ecom_cs`。
> 不要用 `uv run`（会触发网络同步），直接用 `.venv`。

---

## 0. 面试前 5 分钟自检

```powershell
cd D:\worksoftspace\ecom_cs
.\.venv\Scripts\python.exe main.py "订单 52644 到哪了" -v
```

- 能出订单结果 → 环境 OK。
- 浏览器先开好 `http://localhost:8501`。
- **不要**开 Docker（这台机器的 8502 被 Docker 占着）。
- **不要**现场 `uv sync` / `npm build` / 切 `AUTH_ENABLED=1`。

---

## 1. 三分钟极简版（时间紧就只跑这三条）

```powershell
.\.venv\Scripts\python.exe main.py "订单 52644 到哪了，顺便问下退货政策" -v
.\.venv\Scripts\python.exe -m ecom.mcp.server --list-tools
.\.venv\Scripts\python.exe -m ecom.orchestrator.cli.eval_router
```

一条讲架构，一条讲第二消费面，一条讲评测。

---

## 2. 八分钟完整版

### ① 多意图（90 秒，核心）

```powershell
.\.venv\Scripts\python.exe main.py "订单 52644 到哪了，顺便问下退货政策" -v
```

预期：`路由：order、knowledge　来源 llm　置信度 0.95`；order 几十毫秒、knowledge
十秒左右；`决策：answer（2 个单元给出结论，汇总方式 llm）`；约 20 秒。

说：一句话两个需求 → 路由拆两个 worker 并行；超时分开配；汇总只整理、不引入新事实；
两个来源冲突时如实说冲突。

### ② 数据分析（60–90 秒，可选）

```powershell
.\.venv\Scripts\python.exe main.py "哪个地区卖得最好" -v
```

预期：模型写 pandas → 子进程执行 → 出结论，约 30–60 秒。

说：答案不是模型心算，是写代码跑出来的；超时是 `subprocess` 真杀；沙箱是黑名单，
不是强隔离（文档里写清了）。**超过 60 秒就 Ctrl+C 跳过**。

### ③ MCP（30 秒）

```powershell
.\.venv\Scripts\python.exe -m ecom.mcp.server --list-tools
```

预期：`get_order` / `get_order_returns` / `search_knowledge` 三个只读工具。

说：第二个消费面；不经过 Supervisor；写操作和 PII 默认不注册；和对话链路共用
同一批业务函数。

### ④ Streamlit（2–3 分钟）

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

→ `http://localhost:8501`

演示：同一句多意图问题 → 展开「决策链」→ 侧边栏切「订单」→ 切「知识库」问
“退换货政策是什么”。图片上传跳过。

说：界面是第二个视图，不是第二套实现；排查仍用 `main.py -v`。

### ⑤ 评测 + 测试（60 秒）

```powershell
.\.venv\Scripts\python.exe -m ecom.orchestrator.cli.eval_router --show-errors
.\.venv\Scripts\python.exe -m pytest -q
```

预期：规则 `exact 41/50（82.0%）` / `covered 42/50（84.0%）`；测试
`872 passed, 2 skipped`，约 21 秒。

再补一句：真模型（qwen-plus，50 次调用）是 `46/50 = 92%`，multi 66.7%、trap 75%。

说：路由有分数、有错题分类；测试不联网，2 条 skip 只差真 Milvus。

### ⑥ 可选：API / React（被问到再开）

```powershell
.\.venv\Scripts\uvicorn.exe api:app --host 127.0.0.1 --port 8000
```

→ `http://127.0.0.1:8000/docs`、`http://127.0.0.1:8000/ui/`

说：三个入口共用 `answer()` / `answer_stream()`；`/ui` 是构建产物，没有就跳过。
在 `/ui` 问一句知识库问题，指一下正文上方的「思考过程」：路由、worker 完成、
决策是真实阶段，正文是 token 真流；答完自动折叠。

如果被问「关掉标签页记录怎么没了」：匿名模式刻意把会话列表放 `sessionStorage`
（`session_id` 是凭证，不写 `localStorage`）；`AUTH_ENABLED=1` 登录后，
React 会调 `GET /sessions` 恢复该账号的历史，明细按需加载。**别现场切开关**，
指一下 README「登录后回看历史」即可。

如果被问「转人工真的能用吗」：说清四件事——① 说「转人工」走的是窄正则短路，
不调模型，避免“先答完再转”；② 工单落库，一个会话只有一张未关闭工单（部分唯一
索引），客服在工作台认领、看上下文、回复，回复写进同一个会话，admin 还能
强制关闭或把卡住的单转派/放回队列；React 客服台和 `admin.py` 客服工作台是同一个
`tickets` 模块，队列每 5 秒自动刷新；③ **认领之后机器人闭嘴**：用户消息照常落盘
给客服看，但不再调模型，关单后恢复；④ 演示模式 / 未开身份层时接口是 404，
答复会**明说没有人工**，不做“请耐心等待”的假承诺。
**同样别现场切 `AUTH_ENABLED`**；要演示就提前起一个开了身份层的实例。

### ⑦ 可选：admin（30 秒，讲边界）

```powershell
.\.venv\Scripts\streamlit.exe run admin.py --server.port 8503
```

→ `http://127.0.0.1:8503`，口令在 `.env` 的 `ADMIN_PASSWORD`。

说：模型只能登记退货，通过/拒绝是人点的；只改申请状态并留痕，不退款、不改订单原值。
开了身份层时同一个后台还会多一个「客服工作台」（agent 登录只看它和订单），
和 React 客服台共用 `tickets` 状态机，认领 / 回复 / 关单都写审计。

---

## 3. 兜底

| 情况 | 处理 |
|---|---|
| 模型 API 慢/断 | 订单那条仍能走规则路由 + 订单库，最多等 8 秒；知识库/分析跳过，改跑评测和测试 |
| 8501 被占 | `streamlit run app.py --server.port 8504` |
| analytics 超 60 秒 | `Ctrl+C`，说“数据分析是长任务，我提前跑过一份” |
| 中文乱码 | 用 Windows Terminal；或 `chcp 65001` |
| `/ui` 404 | 别现场 build；直接用 Streamlit |
| 被问鉴权 | 口头讲 `AUTH_ENABLED=1` 时的 token/角色/会话归属/客户绑定，**别现场切** |

---

## 4. 收尾一句话

> 重点不是接了几个模型，是每个结论都能追到来源：路由有 confidence、worker 有耗时和
> 错误、汇总有来源、请求有 trace；同时文档写清了没做的部分——强隔离、OIDC、多租户。
