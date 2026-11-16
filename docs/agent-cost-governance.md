# Agent 成本归因与优化验证 · 项目说明

> 这份文档写给第一次接触这个工具的人（面试官 / 技术评审），不假设你读过代码。
> 它讲的是**工具本身**：为什么需要、怎么设计、验证出过什么、边界在哪。
> 外部样本（open_deep_research）的完整数据在
> [agent-cost-open-deep-research.md](agent-cost-open-deep-research.md)——那份是证据，这份是说明。
> 要改代码看 `ecom/agentcost/` 下的模块 docstring，每个函数都写了"为什么这么写"。

---

## 一、它是什么

一个**回答"这笔钱花得对不对"的工具**，不是成本面板。它回答三个问题：

| 问题 | 靠什么回答 |
|---|---|
| 哪一次调用花了多少钱 | call 级记录（节点 / 模型 / 入出 token / 延迟） |
| 这笔钱该不该花 | 按节点、按模型的归因，以及同一任务内部的结构对比 |
| 改完是否更划算 | 改动前后的同批任务对比，质量用命中率兜底 |

和"只看总量的成本统计"的区别只有一条：**它能告诉你钱花在哪一个节点、哪一类 token 上**。总量是结论，结构才是可以行动的信息。

它现在有两个被治理的样本：第一个是我自己写的电商多 Agent 客服系统（ecom_cs），第二个是 LangChain 官方的 open_deep_research。**工具是主体，两个系统都是样本。**

### 它不做什么

- **不评测答案质量。** ecom_cs 那套 11 题评测集属于宿主项目，是"改动后质量不能掉"的护栏，不是这个工具的能力。
- **不替人判断"这笔钱值不值"。** 降级模式便宜是因为它根本没检索，完整模式贵是因为它真的读了资料——工具只把代价和构成摆出来。
- **不做自动优化。** 它验证人提出的改动，不自己改参数、不自己调结构。
- **不给外部项目提优化。** 接入 open_deep_research 时没有改对方一行代码，也没有为省钱调整它的配置。

---

## 二、为什么需要它

一次用户请求不是一个模型调用，是一串：

- ecom_cs 的知识库问答：路由 1 次 + 知识库 1 次 + 汇总 1 次；
- ecom_cs 的数据分析：最多 6 次，跨 DeepSeek（生成代码 / 修正）和 Qwen（审查）两个模型；
- open_deep_research：同样三道题，无搜索 5–6 次，有搜索 18–23 次。

只要记录停在"这次请求总共用了多少 token"，就同时丢掉三件事：

1. **钱算不出来。** token 是跨模型相加的，单价不同；一个模型名填错，误差就是 2.4–4 倍（输入 0.27 / 0.1111 = 2.43，输出 1.10 / 0.2778 = 3.96）。
2. **改动不可验证。** "省了 19.5%" 只有在同一批任务、同一套判定下重复测量才成立。
3. **结构不可见。** open_deep_research 加搜索后成本涨 8.4 倍，只看总量会得出"搜索很贵"这种无法行动的结论；逐次记录才能看到涨的是 `researcher_tools` 的**输入**。

---

## 三、三层结构

| 层 | 做什么 | 落在哪 |
|---|---|---|
| 采集 | 每次模型调用记一条：节点、模型、入出 token、延迟、来源 | ecom_cs 走 trace 的 `calls`；外部项目走 LangChain callback |
| 归因 | 单价表 + 成本三分法 + 按节点 / 模型聚合 | `ecom/agentcost/pricing.py` |
| 验证 | 改动前后同批任务对比，质量用命中判定兜底 | `ecom/orchestrator/cli/eval_answers.py`、外部样本 adapter |

**只有采集层需要为每个新系统重写。** 接入 open_deep_research 时，归因和验证两层一行没改——这是它能同时接两个系统、而且第二个不是我的代码的原因。

---

## 四、成本三分法（核心设计）

成本有三种"零"，必须分开：

| 情况 | 应该返回 | 为什么 |
|---|---|---|
| 走了规则、没调模型（`source="rule"`，token 全 0） | `0.0` | 真的没花钱 |
| 调了模型，但 provider 没返回 usage | `None` | 成本**未知**，不是 0 |
| 有 usage，但单价表里没有这个模型 | `None` | 算不出来，不是 0 |

把后两种算成 0，成本表会静静地少算，而且看不出来——报表每一行都"正常"。所以这里的原则是：**算不出来就返回 None，绝不返回 0 顶替。**

这条设计有实测支撑：analytics 早期一条记录只有 `model=None`（把 DeepSeek 和 Qwen 两路 token 相加了），`unknown_calls` 是 1。按 provider 拆成两条 call 之后降到 **0**——不是"少了一个未知"，是那个未知本来就不该存在。

---

## 五、数据模型

一条 call 记录八个字段：

| 字段 | 含义 |
|---|---|
| `node` | LangGraph 节点名（如 `ods/researcher_tools`） |
| `kind` | 调用类型（路由 / 生成 / 审查 / 规则） |
| `model` | provider 报的真实模型名（如 `qwen-plus`） |
| `tokens_in` / `tokens_out` | 入 / 出 token |
| `latency_ms` | 单次延迟 |
| `source` | `llm` / `rule` / `unknown` |
| `usage_given` | provider 有没有真的返回 usage |

模型名必须经 `resolve_model()` 归一到单价档位（`qwen-plus` → `qwen`，`deepseek-chat` → `deepseek`）——不归一的话，价格填了也永远查不到，而且是静默的。

`slot["calls"]` 保持原来的"调用次数"语义，逐次明细放在新增的 `call_records`，**只在非空时出现**，所以 order / knowledge 的旧口径不被污染。

---

## 六、三条结论

### 结论一：rerank 3→2 省 19.5%，准确率不掉——但钱降得比 token 少

同批 5 道 policy 题，只改 `RERANK_TOP_K`（3→2）：

| 指标 | 改前 | 改后 | 变化 |
|---|---|---|---|
| 成本 | 0.0010700613 | 0.0008610175 | **−19.5%** |
| 输入 token | 6,301 | 4,452 | −29.3% |
| 输出 token | 1,144 | 961 | −16.0% |
| 命中 | 5/5 | 5/5 | 不掉 |

**能推出**：减少塞进提示词的检索块，确实同时降 token 和成本，且不影响关键事实。
**不能推出**：这是"最优值"。2→1 没测；样本只有 5 题、只覆盖 policy。

**顺带暴露的一件事**：token 降 29.3%，钱只降 19.5%。因为输出单价是输入的 2.5 倍（0.2778 vs 0.1111），而这次改动主要砍的是输入。**结论是优化要盯输出，不是盯总 token。**

### 结论二：analytics 单题成本差 3 倍，根因是审查轮次，不是代码复杂度

5 道 analytics 题，加诊断字段（只加输出，不改行为）：

| 题 | 成本 | `retry_count` | `review_round` |
|---|---|---|---|
| A01 | $0.000920 | 0 | 1 |
| A02 | $0.002028 | 0 | 2 |
| A03 | $0.000661 | 0 | 0 |
| A04 | $0.001998 | 0 | 2 |
| A05 | $0.000939 | 0 | 1 |

**成本随 `review_round` 单调上升**：0 轮 $0.00066、1 轮 $0.00092 / $0.00094、2 轮 $0.00203 / $0.00200。2 轮意味着第一次审查没过、改写一版、又复审一次。

**这条结论直接否掉了一个提案**：把 `MAX_EXEC_RETRY` 从 3 改成 1。`retry_count` 五题全是 0，这个开关从未生效——**改造前后没有任何区别**。如果没先测就改，会得到一个"看起来是优化、实际是空操作"的提交。

同类还排除过三个：把检索候选数 `TOP_K` 从 5 改成 3（真正的旋钮是 `RERANK_TOP_K`，当时已经是 3，所以那次是空操作）、"把 schema 前置到 analyst 提示词"（schema 早就写在提示词里）、"self_corrector 最多 1 轮"（对审查循环早已是现状）。

**不能推出**：审查轮次多就"浪费"。A02 / A04 正是靠那一轮修正才过的；关掉是拿质量换钱。

### 结论三：外部样本的成本结构会因运行模式整体反转

同一个系统、同样三道题，只改"有没有检索"：

| | 降级模式（无搜索） | 完整模式（Tavily） |
|---|---|---|
| 调用 | 16 | 63（3.9 倍） |
| 成本 | $0.006458 | $0.054156（8.4 倍） |
| 最贵节点 | `final_report_generation` 48.2% | `researcher_tools` **69.1%** |

完整模式下 `researcher_tools` 39 次调用、输入 297,504 token、输出 15,744，**输入输出比 18.9 : 1**。钱不是花在"写"，是花在"反复读"。

**这条结论证明的是工具，不是 ODS**：它说明只看总量会得到"搜索让成本涨了 8 倍"，逐次记录才能定位到涨的是哪一类 token。详细数据见外部样本报告。

---

## 七、边界与局限

诚实列出，不弱化：

1. **样本小。** 外部样本 3 题、单一模型（qwen-plus）；rerank 实验 5 题、只覆盖 policy。不构成基准。
2. **成本是下界。** open_deep_research 的 `final_report` 三题都顶到 3000 token 上限（默认 10000），`compress_research` 两次顶 2000。输出被截断，总成本只会更高。
3. **汇率是假设值。** `PRICES` 里的人民币 → 美元按 **7.2** 折算，实际结算汇率不同会整体偏移。
4. **思考模式未确认。** 若账号默认开思考模式，qwen-plus 输出价应为 1.1111（8 元/M），当前填的 0.2778 会低估 **4 倍**。只有账单能确认。
5. **没有多模型分工对照。** open_deep_research 默认是 gpt-4.1-mini 做摘要、gpt-4.1 做研究与报告，这里统一成一个模型，绝对值不可外推。
6. **没有物理拆仓库。** 工具在 `ecom/agentcost/`，与宿主同仓库；独立性的证据是"接住了外部项目"，不是"搬进了新文件夹"。

---

## 八、目录与复现

```
ecom/agentcost/          # 工具本体，不 import 任何业务子系统
  __init__.py            # 公共 API 出口
  pricing.py             # 单价表 + 成本三分法 + 汇总
ecom/orchestrator/adapters/open_deep_research.py   # 外部样本采集
ecom/orchestrator/cli/eval_answers.py              # 同批任务对比
```

依赖方向是**单向**的：`orchestrator → agentcost`。反过来由 `tests/test_package_layout.py::test_agentcost_stays_under_ecom_and_never_reaches_back` 守着——`agentcost` 一旦 import `ecom.orchestrator` / `analytics` / `knowledge`，测试就红。它的输入只能是调用记录这个纯 dict。

复现：

```bash
# 全量测试（904 passed）
python -m pytest tests -q

# 同批任务对比（policy / analytics）
python -m ecom.orchestrator.cli.eval_answers --category policy
python -m ecom.orchestrator.cli.eval_answers --category analytics

# 外部样本（必须用对方的 venv，超时不要低于 600 秒）
cd D:/worksoftspace/open_deep_research
.venv/Scripts/python.exe D:/worksoftspace/ecom_cs/ecom/orchestrator/adapters/open_deep_research.py --index 1 --timeout 600
```

数据落点：`logs/requests.jsonl`（每个请求的 trace，含 `calls` 与 `cost`）、`data/orchestrator/answer_eval_runs/`（逐题全文，未入库）、`data/orchestrator/ods_runs/`（外部样本逐次调用，未入库）。

---

## 九、还没做的

不是待办清单，是"现在经不起追问的地方"：

- **第二个外部样本。** 现在只有一个非自建项目。两个才能说"通用"，一个只能说"接住过一次"。
- **评测集 11 题**。扩展到 30–50 题没做；现在任何"准确率不掉"的结论都建立在 5–11 题的样本上。
- **按模型聚合的报表。** 现在能按节点聚合，按模型聚合还没有正式的 CLI 输出。
- **`MAX_REVIEW_ROUNDS` 2→1 的实验。** 它是唯一还剩的成本杠杆，但属于拿质量换钱，需要人工复核答案才敢做，因此没做。
