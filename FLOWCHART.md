# 流程图

配着 [README.md](README.md) 看。这里只画**代码里真实存在的东西**：图的形状、状态怎么流、每个节点在哪一行。

---

## 一、图的形状

```mermaid
graph TD
    START((START)) --> router
    router -->|"superstep：三个节点同时进入<br>不在 plan 里的那个直接 return {}"| worker_knowledge
    router --> worker_order
    router --> worker_analytics
    worker_knowledge --> aggregate
    worker_order --> aggregate
    worker_analytics --> aggregate
    aggregate -->|'decision == "re_route"'| router
    aggregate -->|其它| END((END))
```

同一个东西，去掉渲染依赖：

```
                     ┌────────────────────────────────────┐
                     │            re_route                │
                     ▼                                    │
 START ──────► router ──┬──► worker_knowledge ──┐         │
                       ├──► worker_order     ──┼──► aggregate ──┬──► END
                       └──► worker_analytics ──┘               │
                                                               └──► END
```

**三条 worker 边永远都在，哪怕这次只用了一个。** 这不是疏漏，是取舍：条件边要动态决定连到谁，得用 LangGraph 的 `Send`（map-reduce），计划在 `router` 里已经算好了，把它摊成三条固定边 + 节点自己判空，换来的是一张**面试时能直接画在白板上**的图。代价是 trace 里会看到两个"没参与"的节点。见 [supervisor.py:14-25](ecom/orchestrator/core/supervisor.py#L14-L25)。

---

## 二、状态怎么流

`CSState`（[state.py](ecom/orchestrator/core/state.py)）里两个字段的走法要单独说，它们是多 agent 最容易被追问的地方。

### `results`：并行写 + 跨轮累加

```mermaid
graph LR
    subgraph R1["第 1 轮"]
        W1a[worker_order] -->|"append {worker, round:1}"| RA["results"]
        W1b[worker_analytics] -->|"append {worker, round:1}"| RA
    end
    subgraph R2["第 2 轮（重路由后）"]
        W2[worker_knowledge] -->|"append {worker, round:2}"| RA
    end
    RA -->|"round_results(results, 2)"| AGG[aggregate 只看本轮]
```

`results: Annotated[list, operator.add]`——并行节点各自 append，合起来，谁也不覆盖谁。代价是**它跨轮只增不减**：往 add 型通道写 `[]` 等于没写。

所以每条结果由 `base.stamp_round` 盖上 `round` 标记，`aggregate` 用 `base.round_results` 挑出本轮的。**没有 reducer 的字段在并行写时 LangGraph 会直接报错**——那是保护，不是麻烦，它逼你想清楚"两个节点同时写同一个键"该合并还是该覆盖。

### `tried`：由 router 一次写入，worker 不碰

`tried` 也是"多个节点想写同一个键"的场合，但这里**没上 reducer**，改成由 `router` 一次性写入（[supervisor.py:63](ecom/orchestrator/core/supervisor.py#L63)）。理由：计划本来就是 router 定的，它顺手记下来最自然；而且这样绕开了并发写。**能用设计绕开的并发问题，就不该用锁或 reducer 去救。**

---

## 三、router 节点的三层优先级

代码在 [router.py:142-215](ecom/orchestrator/core/router.py#L142-L215)。顺序就是优先级，**从"聪明"到"兜底"**：

```
route(question, tried)
  │
  ├─ ① LLM 分类          ── 成功 → source="llm"
  │     └─ 失败（超时/没key/解析不出）→ 落到 ②
  │       · max_retries=0（重试权收回来）
  │       · ROUTER_TIMEOUT=8s 单次，ROUTER_BUDGET=20s 总时长
  │
  ├─ ② 关键词规则        ── 命中 → source="rule"
  │     订单号(6位数字)/物流/退货… → order
  │     销量/排名/趋势/占比…       → analytics
  │     政策/保修/发票/流程…       → knowledge
  │     └─ 一个都没命中 → 落到 ③
  │
  └─ ③ 最小默认计划      ── source="fallback"，plan=["knowledge"]
        knowledge 是唯一"什么都能聊两句"的，答不出会走 insufficient
        （比甩一句"我帮您转人工"体验好）

  最后统一：plan = [n for n in plan if n not in tried]   ← 回环不原地打转
```

**`router_source` 必须记进 trace**：它是判断路由健康度的关键指标——"rule / fallback 的比例一涨，说明 LLM 那边在坏"。

低置信度（< 0.6）时会往计划里**补一个 knowledge**。这是**故意的不对称**：漏调一个 worker = 少答一半（用户看得出来），多调一个 = 多花一次调用（用户看不出来）。

---

## 四、aggregate 节点的决策树

代码在 [aggregate.py:176-258](ecom/orchestrator/core/aggregate.py#L176-L258)。**短路返回，顺序不能换**：

```
aggregate(state)
  │
  ├─ ① usable 非空 ──────────────────► decision = "answer"
  │      合并 usable；failed 的补一句"这块暂时用不了"；
  │      thin 的澄清话术、上一轮的 pending_note 都附在后面
  │
  ├─ ② 想重路由吗？ ─────────────────► decision = "re_route"
  │      can_retry = plan 非空          ← 空计划重来还是空
  │                & untried 非空       ← 没有新单元 = 原地打转
  │                & round < MAX_ROUNDS ← 回环有界
  │                & alive              ← 全挂了别换人再挂一次
  │
  ├─ ③ 手上有话可说吗？ ─────────────► decision = "answer"
  │      thin/failed 有话说 → 如实转达；
  │      全挂了的话再追加一句 HANDOFF
  │
  └─ ④ 什么都说不出来 ──────────────► decision = "human"
         HANDOFF_ONLY（唯一真正该转人工的情况）
```

两条最容易被问的：

- **`ok=False`（坏了）不触发重路由，`insufficient=True`（看过了没有）才触发。** 前者多半是同一个网络/同一台机器的问题，再试一次大概率还是坏——那叫"服务一抖动就换个 worker 再抖一次"。
- **`len(usable) == 1` 时不调模型**，原样返回。没有东西可合并时调模型，只会把一段本来忠实的话改写成可能不忠实的另一段话——多一次幻觉机会，零信息增益。

---

## 五、一次请求的完整时序

以 `订单 100002 到哪了，顺便问下你们的退货政策是什么` 为例，这是 **2026-10-10 真跑出来的一次**（模型和知识库都可用）：

```mermaid
sequenceDiagram
    participant U as 用户
    participant R as router
    participant K as worker_knowledge
    participant O as worker_order
    participant A as aggregate

    U->>R: 问题
    Note over R: ① LLM 分类成功，置信度 0.95<br>plan=[order, knowledge], source=llm<br>round=1, tried=[order,knowledge]
    par 同一超步，并行
        R->>K: question
        K-->>A: ok=True, answer="七天无理由…"（8.3s）
    and
        R->>O: question
        O-->>A: ok=True, answer="订单 100002 的物流…"（11ms）
    end
    Note over A: ① usable 有 2 个 → decision=answer<br>汇总方式 llm；sources=[售后政策.md, 退换货流程.md, 100002]
    A-->>U: 物流 + 退货政策，并**如实报出两者对同一笔订单的说法冲突**
```

整个过程：**第一次 invoke 就结束，没有回环**。两个 worker 的耗时差三个数量级（8.3 秒 vs 11 毫秒）——这就是为什么它们的超时是分开配的（90 / 20 / 180）。

同一天更早的一次运行里 dashscope 不通，那次走的是另一条完整路径：**LLM 路由超时 8 秒 → 规则命中 → 知识库 worker 超时（ok=False）→ 订单那半照常答出来 → 明说"知识库这块暂时查不了"**，总耗时 17.6 秒而不是 180 秒。两条路径都不是摆拍，一次请求把三四条兜底链路全演示了。

---

## 六、三个子系统的调用边界

三个 worker 看起来都是"→ aggregate"的箭头，实际跑起来不一样：

```
worker_knowledge ──► 进程内 import      ecom.knowledge.retriever / generator / vector_store
                     │  (只 import 4 个入口，不 import core/graph.py：绕开联网搜索
                     │   + 待审核队列的副作用。这个理由和撞名无关，合并前后都成立)
                     │
worker_order     ──► 本进程 + SQLite    ecom/orchestrator/tools/order_tools.py（纯函数，无模型）
                     │
worker_analytics ──► subprocess        python -m ecom.analytics.cli.analyze "问题"
                     │  子进程 stdout 只吐一行 JSON；超时设成比 worker 超时短 5s，
                     │  让子进程自己的超时先触发（否则是父进程单方面放弃）
                     └─ PYTHONIOENCODING=utf-8（GBK 控制台）
```

**合并前这三行里有三处是为"跨仓库"写的**：知识库那条要 `sys.path.insert` 挂 sibling 仓库、还要一个 `_assert_imports_came_from()` 运行时闸门；数据分析那条的命令里写的是 `../sales_agent/analyze_json.py`，还要一个 `SALES_AGENT_PYTHON` 配置项来决定用哪个解释器。**现在都删了**：sibling 就是自己，子进程用 `sys.executable`（同一个 venv）。

**但数据分析那条边界留着**，理由换了一个：不再是被命名逼的，而是它执行的是模型生成的代码，需要一个能被真杀死的进程边界。完整版理由见 [workers/analytics.py](ecom/orchestrator/workers/analytics.py) 的模块 docstring。

**界面（`app.py`）也走这条边界**——它调的是同一个 `run_analytics()`，而不是"为了少起一个进程"直接 import 那个图。界面不该是那条边界的例外。

---

## 七、文件 → 节点

| 节点 | 文件 | 关键函数 |
|---|---|---|
| 图的装配 | [ecom/orchestrator/core/supervisor.py](ecom/orchestrator/core/supervisor.py) | `build_graph` / `_router_node` / `_worker_node` |
| 状态契约 | [ecom/orchestrator/core/state.py](ecom/orchestrator/core/state.py) | `CSState` / `WorkerResult` / `make_initial_state` |
| router | [ecom/orchestrator/core/router.py](ecom/orchestrator/core/router.py) | `route` / `_rule_route` / `parse_route` |
| aggregate | [ecom/orchestrator/core/aggregate.py](ecom/orchestrator/core/aggregate.py) | `aggregate` / `synthesize` / `_split` |
| worker 公共契约 | [ecom/orchestrator/workers/base.py](ecom/orchestrator/workers/base.py) | `run` / `ok_result` / `error_result` / `stamp_round` |
| 三个 worker | [ecom/orchestrator/workers/](ecom/orchestrator/workers/) | `knowledge.py` / `order.py` / `analytics.py` |
| 订单工具层 | [ecom/orchestrator/tools/order_tools.py](ecom/orchestrator/tools/order_tools.py) | 纯函数，`OrderNotFound` / `ReturnNotAllowed` |
| 订单库 | [ecom/orchestrator/db/](ecom/orchestrator/db/) | `engine.py`（双后端）/ `schema.sql` / `seed.py` |
| 知识库子系统 | [ecom/knowledge/](ecom/knowledge/) | `retriever` / `vector_store` / `embedding` / `generator` / `cli` |
| 数据分析子系统 | [ecom/analytics/](ecom/analytics/) | `core/graph.py` / `core/sandbox.py` / `agents/` / `cli/analyze.py` |

四个入口，各管一件事：

| 入口 | 给谁用 | 为什么是它 |
|---|---|---|
| [main.py](main.py) | **排查问题** | 决策链摊在眼前；界面会把它藏起来 |
| [app.py](app.py) | 演示 | 四个 Tab = 四个面，共用一份进程 |
| [api.py](api.py) | 被别的程序调 | `def` 不是 `async def`——图里全是阻塞调用 |
| `python -m ecom.analytics.cli.analyze` | **给编排层当子进程** | stdout 只吐一行 JSON；契约写在名字里 |
