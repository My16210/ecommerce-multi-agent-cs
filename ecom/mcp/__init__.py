"""
MCP 工具层：把已有的业务函数包成一份**对外的、稳定的 JSON 契约**。

【这个包和另外三块的分工】

    ecom/orchestrator   一条对话链路：路由 → worker → 汇总
    ecom/mcp            一个工具服务：别的 Agent / 客户端按需调用单个能力

两者**共用同一批业务函数**（`tools/order_tools.py`、`ecom.knowledge.retriever` 等），
而不是互相调用。MCP 这条路上不跑 Supervisor：调用方已经知道要查订单，不需要
再花一次模型调用去路由。反过来，对话链路也不经过 MCP——中间多一层进程/协议
换不来任何东西，只会多一个会挂的地方。

【为什么 `tools.py` 不 import MCP SDK】

`mcp` 是可选依赖（`uv sync --extra mcp`）。工具层的契约、校验和错误码都能在
**没有装 SDK** 的环境里被测试，只有 `server.py` 认识 `MCPServer`。这样
"业务规则"和"协议接线"分开，前者不因为后者换版本而改。

【v1 只做 stdio，不做 HTTP / OAuth】

stdio 由客户端拉起一个本地进程，没有监听端口、没有鉴权问题、没有 CORS。
HTTP 传输要回答"谁能调、调用算谁的、怎么限流"——那些问题在 `ecom/demo.py`
里有一半答案，但**不是给机器调用方准备的**。没想清楚之前不开这个口子。
"""
