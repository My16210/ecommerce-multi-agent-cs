"""Agent 成本归因工具本体（agentcost）。

**边界**：本包只接受「调用记录」（纯 dict）做单价换算与汇总，**不 import 任何
业务子系统**（orchestrator / analytics / knowledge / ui / mcp / auth 一律不碰），
也不认识任何具体的 agent 系统。采集是调用方的事——对 ecom_cs 是 trace 里的
`calls`，对外部项目是 `ecom/orchestrator/adapters/` 里的 callback。

依赖方向是单向的：`orchestrator → agentcost`。反过来不成立，所以它才能同时接
自己的系统和别人的系统。这条约束由 `tests/test_package_layout.py` 里的一条
测试守着，不是靠这段注释。

公共 API 都在这里转出，调用方只需要 `import ecom.agentcost`。
"""

from .pricing import (
    PRICE_VERSION,
    PRICES,
    cost_of,
    cost_of_call,
    fmt_cost,
    price_of,
    resolve_model,
    summarize_cost,
    unpriced_models,
)

__all__ = [
    "PRICES",
    "PRICE_VERSION",
    "cost_of",
    "cost_of_call",
    "fmt_cost",
    "price_of",
    "resolve_model",
    "summarize_cost",
    "unpriced_models",
]
