# MCP 工具服务

> 一句话：把「查订单 / 查退货 / 检索知识库」这三个能力，按 MCP 协议暴露给别的
> Agent 调用。**它不跑 Supervisor**——调用方已经知道要干什么，不需要再路由一次。

- 代码：`ecom/mcp/`（`tools.py` 是纯函数，`server.py` 才认识 MCP SDK）
- 传输：**只支持 stdio**（客户端拉起本地进程），不监听端口
- 依赖：可选，`uv sync --extra mcp`
- 测试：`uv run pytest tests/mcp`

---

## 1. 为什么要有这一层

这个项目原来只有一个入口：自然语言 → 路由 → worker → 汇总。它解决的是
「用户不知道自己该找谁」。MCP 解决的是相反的问题：

> 另一个 Agent 已经知道「我要查订单 52644」，它不该为了这件事去调用一次
> 大模型做意图识别，再等一段自然语言回答。

所以两条路**共用业务函数**，但不互相调用：

```
对话链路   app.py / api.py → supervisor → workers → tools/order_tools.py
                                                    ↖
MCP 客户端  ──stdio──→ ecom/mcp/server.py → ecom/mcp/tools.py ─┘
```

- MCP 不经过 Supervisor：少一次模型调用、少一层可能挂的地方。
- 对话链路不经过 MCP：中间多一个进程和一轮协议，换不来任何隔离收益。
- 两边都调 `order_tools.py` 和 `ecom.knowledge` 的检索函数，**规则只有一份**。
  复制一份到 MCP 里，两份迟早会对同一笔订单给出不同结论。

---

## 2. 工具清单

| 工具 | 默认 | 作用 | 会写库吗 |
|---|---|---|---|
| `get_order(order_id)` | ✅ 开 | 订单状态、金额、区域、时间 + 物流 | 否 |
| `get_order_returns(order_id)` | ✅ 开 | 某订单已有的退货申请 | 否 |
| `search_knowledge(query, top_k=3)` | ✅ 开 | 知识库混合检索 + rerank，只返回原文和出处 | 否 |
| `list_customer_orders(customer_id, limit=10)` | ❌ 关 | 按客户号查名下订单 | 否 |
| `create_return(order_id, reason, confirm=False)` | ❌ 关 | 提交一笔「待审核」退货申请 | **是** |

默认关闭的两个工具由服务端开关决定**注不注册**：

```bash
MCP_ENABLE_PII=1      # 注册 list_customer_orders
MCP_ENABLE_WRITES=1   # 注册 create_return
```

为什么是「不注册」而不是「注册了再拒绝」：工具列表是调用方唯一的发现机制。
一个永远返回 `not_allowed` 的工具，模型会先调一次、被拒、再放弃——白跑一轮；
列表里没有它，模型第一步就知道这个能力不存在。**能力发现要说真话。**

`create_return` 即使开关打开，也还要求调用方传 `confirm=true`；演示模式
（`DEMO_MODE=1`）下无论开关如何都拒绝落库。三件事防的不是同一种误操作：

| 闸 | 谁决定 | 挡住什么 |
|---|---|---|
| `MCP_ENABLE_WRITES` | 服务端 | 这个能力根本不对机器开放 |
| `confirm=true` | 调用方 | 模型把「能退吗」直接执行成申请 |
| `DEMO_MODE` | 部署方 | 公网演示环境被写脏 |

### 没有暴露的能力

- **数据分析（`analyze_sales`）**：它会执行模型生成的 Python。对话链路里这条
  已经是最需要隔离的一环，再开一个没有对话上下文的入口只会扩大攻击面。
- **按订单号改状态 / 审核退货**：审核是人的事，入口在
  `ecom/orchestrator/admin/`，不在 `tools/`。
- **会话 / 记忆**：MCP 调用方自己维护上下文，服务端不做会话。

---

## 3. 返回契约

所有工具返回同一个信封，**失败不抛异常**：

```json
// 成功
{"ok": true, "data": {...}, "meta": {"elapsed_ms": 3.2}}

// 失败
{"ok": false, "error": {"code": "not_found", "message": "查不到订单号 99999。"},
 "meta": {"elapsed_ms": 1.1}}
```

错误码只有五个：

| code | 含义 | 调用方该怎么做 |
|---|---|---|
| `invalid_input` | 参数形状不对 | 改参数重试 |
| `not_found` | 确实查不到 | 转告用户 / 换单号 |
| `not_allowed` | 业务规则不允许 | **不要重试**（状态不可退、重复申请、演示只读） |
| `unavailable` | 依赖暂时坏了 | 稍后重试 |
| `internal` | 未预期错误 | 稍后重试；细节只进服务端日志 |

为什么不直接抛异常：MCP 里异常会变成一次协议错误，模型通常只看到
"tool failed"；而 `invalid_input` 的 message 是它**能据此自我纠正**的话。
内部异常的原始信息（可能带 DSN、路径）只写日志，不出现在 message 里。

### 字段裁剪

只读查单**不返回** `customer_name` / `customer_id`。对话链路需要它们
（判断是不是本人），按单号查单的 MCP 调用方不需要——返回客户姓名只是顺手把
一份 PII 交给了一个我们不知道会不会转存的客户端。字段清单是白名单
（`tools.py` 的 `_ORDER_FIELDS` 等），以后表里加一列不会自动泄露。

---

## 4. 用法

### 4.1 安装与自检

```bash
uv sync --extra mcp

# 当前开放了哪些工具（受环境变量影响）
uv run python -m ecom.mcp.server --list-tools

# 订单库能不能打开、当前开关状态
uv run python -m ecom.mcp.server --check
```

### 4.2 客户端配置

以 Claude Code / Claude Desktop 的 MCP 配置为例（路径换成自己的仓库绝对路径）：

```json
{
  "mcpServers": {
    "ecom-customer-service": {
      "command": "uv",
      "args": ["run", "--directory", "D:/worksoftspace/ecom_cs", "python", "-m", "ecom.mcp.server"],
      "env": {
        "MCP_ENABLE_PII": "0",
        "MCP_ENABLE_WRITES": "0"
      }
    }
  }
}
```

`.env` 会被 `ecom/settings.py` 照常加载，所以客户端配置里**不需要、也不应该**
写模型 key 或数据库口令。要覆盖就用 `env`，但别把密钥写进客户端的配置文件——
那份文件通常不在 `.gitignore` 里。

### 4.3 stdout 是协议

stdio 传输在 stdout 上跑 JSON-RPC。任何 `print()` 都会打断它，症状是
「服务起来了，但一个响应都收不到」，而不是报错。本项目两个 logger 的 handler
都挂在 **stderr**，所以正常日志是安全的；新加代码时别自己往 stdout 打东西。
`--list-tools` / `--check` 是例外：它们不跑协议。

---

## 5. 安全边界（别把这里说大）

- **没有鉴权**：stdio 的安全性来自「谁能拉起这个进程」，不是来自协议。客户端
  配置里有这条命令的人就能调。真要多人/远程调用，需要的是 HTTP 传输 + 身份 +
  审计，v1 刻意没做。
- **没有限流 / 预算**：`ecom/demo.py` 的限流和 token 预算是给公网对话入口的。
  MCP 是本地进程，调用方自己负责频率；`search_knowledge` 的 rerank 可能产生
  外部 API 费用。
- **公网演示不部署它**：`docker-compose.prod.yml` 只跑 app + Caddy，MCP 不在
  镜像的启动命令里，也没有端口。L1 演示的只读承诺不依赖这个模块。
- **写操作不是事务性的跨系统操作**：`create_return` 只插一行「待审核」，
  不触发退款、不发通知。和对话链路是同一条边界。
- **知识库检索会加载 embedding / rerank**：首次调用可能几秒。它只读，不写
  向量库，也不播放 `record_question`（不会把问题塞进人审队列）。

---

## 6. 已知限制

| 限制 | 为什么现在不做 |
|---|---|
| 只有 stdio | HTTP 传输要先回答「谁能调、算谁的、怎么限流」，没想清楚不开 |
| 没有 resources / prompts | 这三个能力都是「调用一次拿结果」，没有需要订阅的资源 |
| 没有流式 | 单次查询都是毫秒到几秒，流式只增加协议复杂度 |
| `search_knowledge` 不生成答案 | 调用方自己就是模型，它要原文和出处，不要二手结论 |
| 1~4 位订单号认不出 | 和对话链路同一条边界，见 `ecom/orchestrator/utils/ids.py` |

---

## 7. 测试

```bash
uv run pytest tests/mcp -q
```

- `tests/mcp/test_tools.py`：**不装 SDK、不连网络、不碰真库**。校验、字段裁剪、
  开关、confirm 闸都在这里。
- `tests/mcp/test_server.py`：`importorskip("mcp")`，验工具列表随开关变化、
  调用返回信封、`--list-tools` 输出可解析。

`tests/mcp/conftest.py` 的 autouse 夹具把两个开关和 `DEMO_MODE` 钉成确定值——
否则测试结果会取决于跑测试那台机器的 `.env`。
