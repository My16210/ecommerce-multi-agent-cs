"""
编排层：**一个 Supervisor 加三个 Worker**，用户一句自然语言进来，一条答复出去。

    core/router.py        决定这一问该由谁办（模型路由 → 关键词规则 → 兜底）
    core/supervisor.py    建图：router → 三个 worker 并行 → aggregate →（回环或结束）
    core/aggregate.py     汇总各 worker 的结论，并决定"答了 / 再查一轮 / 转人工"
    core/state.py         图的状态（含那个只能追加的 results 通道）
    workers/              三个 worker：knowledge / order / analytics
    tools/order_tools.py  纯函数，给 order worker 用
    db/                   订单库：SQLite 默认、MySQL 可选，同一份 schema
    utils/                日志、trace、JSON 兜底解析、token 计量

【这个包原来叫 `cs`，而那个名字是被一次事故逼出来的】

它原先叫 `cs`，因为本项目的配置曾放在 `config/settings.py`，和
rag_knowledge_base 撞了顶层包名——进程内 import 时，rag 的 16 个
`from config.settings import ...` 全被劫持到我们这份上，报出来的是
`ImportError: cannot import name 'LOGS_PATH' from 'config.settings'`，
而 `LOGS_PATH` 和故障原因一点关系都没有。于是当时选了个"足够独特、永远撞不上"
的名字让开。

合并之后 sibling 就是自己，没有"别人的包"了，"独特命名"这个诉求**本身消失了**。
这个包改名成 `ecom.orchestrator`、和另外两个子系统平级——不是换了个更独特的
名字，是那个约束不存在了。

留着的教训只有一条：顶层包名就是全局命名空间，所以全项目只有一个 `ecom`。
`tests/test_package_layout.py` 守着这件事。
"""
