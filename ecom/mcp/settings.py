"""
MCP 工具服务的开关与上限。

【为什么写操作和 PII 默认关闭】

MCP 的调用方是**另一个 Agent**，不是已经在界面上登录的人。默认开一个能改
数据、能按客户号拉订单的接口，等于把"模型可以替用户办事"这件事交给一个
我们看不见的客户端去决定。所以 v1 的姿态是：

    get_order / get_order_returns / search_knowledge   默认开（只读、按单号）
    list_customer_orders（按客户号）                    默认关
    create_return（写）                                 默认关

要开就显式设 `MCP_ENABLE_PII=1` / `MCP_ENABLE_WRITES=1`。**写操作即使开了，
也仍然要求调用方显式传 `confirm=true`**——两把钥匙，不是一把。

【上限为什么写在这里而不是照搬对话链路】

对话链路的 `MAX_QUESTION_CHARS=2000` 是给"用户整段话"的，里面可能带着订单号、
退货理由、上下文。MCP 的 `search_knowledge` 只收一个检索词，500 字已经远超
正常用法；给 2000 只会让一次调用把检索 prompt 撑大。**同一类参数在不同入口
取不同的值不是不一致，是两个入口的输入形状本来就不一样。**
"""
import os


def _flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


# 写操作（目前只有 create_return）。打开后仍然要求 confirm=true。
MCP_ENABLE_WRITES = _flag("MCP_ENABLE_WRITES")

# 按客户号查订单。客户号本身是 PII 的入口，整条工具默认不注册。
MCP_ENABLE_PII = _flag("MCP_ENABLE_PII")

# 检索词上限（字符）。超了直接拒绝，不截断——截断会让调用方以为查的是整句。
MCP_MAX_QUERY_CHARS = int(os.getenv("MCP_MAX_QUERY_CHARS", "500"))

# 检索一次最多返回几条。rerank 的默认是 3，这里允许调用方在上限内自己挑。
MCP_MAX_TOP_K = int(os.getenv("MCP_MAX_TOP_K", "5"))

# list_customer_orders 一次最多返回几条。**必须有个上限**：DataCo 里一个大客户
# 名下可能有上百笔订单，原样吐出去会把调用方的上下文撑爆。
MCP_MAX_CUSTOMER_ORDERS = int(os.getenv("MCP_MAX_CUSTOMER_ORDERS", "20"))
