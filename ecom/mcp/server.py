"""
MCP 服务端：把 `tools.py` 里的函数注册成工具，只走 stdio。

【stdio 的硬约束：stdout 是协议，日志只能走 stderr】

MCP 的 stdio 传输在 stdout 上跑的是一行一个 JSON-RPC。任何一句
`print("调试一下")` 都会把这条流打断——客户端的表现是"服务起来了，但一个
响应都收不到"，而不是报错。本项目的两个 logger
（`ecom/orchestrator/utils/logger.py`、`ecom/knowledge/utils/logger.py`）
本来就把 handler 挂在 stderr 上，所以这里不用额外改；但**新加日志时别再自己
`print` 到 stdout**。`--list-tools` / `--check` 是例外：它们不跑协议，本来就该
往 stdout 打给人看。

【为什么按开关决定"注册不注册"，而不是"注册了再拒绝"】

工具列表是调用方（另一个 Agent）唯一的发现机制。注册一个永远返回
`not_allowed` 的工具，模型会先试着调一次、被拒、再放弃——多一轮没有信息量的
往返；而列表里没有它，模型从第一步就知道这个能力不存在。**能力发现要说真话。**
`tools.py` 里每个受控工具仍然会在函数入口再查一次开关，那是给"有人绕过
`build_server()` 直接调用"留的兜底。
"""
import argparse
import asyncio
import json

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from ecom.mcp import settings, tools

INSTRUCTIONS = """\
电商客服系统的工具服务。所有工具都返回同一个信封：

  成功  {"ok": true,  "data": {...}, "meta": {"elapsed_ms": 12.3}}
  失败  {"ok": false, "error": {"code": "...", "message": "..."}, "meta": {...}}

错误码：invalid_input（参数形状不对，改参数重试） / not_found（确实查不到） /
not_allowed（业务规则不允许，不要重试） / unavailable（依赖暂时坏了，可稍后重试） /
internal（未预期错误，可稍后重试）。

只读工具可以放心调用；create_return 会把一笔「待审核」申请写进库，必须先得到
用户的明确同意再传 confirm=true。通过/拒绝退货是人做的，工具层没有这个入口。
"""


def _read_only() -> ToolAnnotations:
    return ToolAnnotations(readOnlyHint=True, openWorldHint=False)


def build_server() -> MCPServer:
    """
    组装工具服务。**每次调用都重新读开关**——测试和运维都能通过环境变量
    决定这次拉起哪些工具，不用改代码。
    """
    server = MCPServer(
        name="ecom-customer-service",
        title="电商客服工具",
        version="0.1.0",
        instructions=INSTRUCTIONS,
    )

    @server.tool(annotations=_read_only())
    def get_order(order_id: str) -> dict:
        """按订单号查订单状态、金额与物流。不返回客户姓名和客户号。

        order_id 是 5 位以上纯数字，例如 "52644"。
        """
        return tools.get_order(order_id)

    @server.tool(annotations=_read_only())
    def get_order_returns(order_id: str) -> dict:
        """查某订单已有的退货申请（只读）。先确认订单存在，再返回申请列表。"""
        return tools.get_order_returns(order_id)

    @server.tool(annotations=_read_only())
    def search_knowledge(query: str, top_k: int = 3) -> dict:
        """在售后政策 / 制度 / 商品 FAQ 里检索相关段落，只返回原文和出处，不生成答案。

        query 最多 500 字；top_k 默认 3，最多 5。
        """
        return tools.search_knowledge(query, top_k=top_k)

    # 下面两个默认不注册。列表里没有它们，调用方从发现阶段就知道能力不存在。
    if settings.MCP_ENABLE_PII:

        @server.tool(annotations=_read_only())
        def list_customer_orders(customer_id: str, limit: int = 10) -> dict:
            """按客户号查名下订单（最近的在前）。需要服务端开启 MCP_ENABLE_PII。

            customer_id 形如 "C20755"；limit 默认 10，最多 20。
            """
            return tools.list_customer_orders(customer_id, limit=limit)

    if settings.MCP_ENABLE_WRITES:

        @server.tool(
            annotations=ToolAnnotations(
                readOnlyHint=False,
                destructiveHint=True,
                idempotentHint=False,
                openWorldHint=False,
            )
        )
        def create_return(order_id: str, reason: str, confirm: bool = False) -> dict:
            """为订单提交一笔「待审核」的退货申请。需要服务端开启 MCP_ENABLE_WRITES。

            **只有用户明确同意之后才传 confirm=true。** 不传 confirm 时只做参数
            校验、不会落库。申请提交后仍需人工审核，不代表已通过。
            """
            return tools.create_return(order_id, reason, confirm=confirm)

    return server


async def _describe(server: MCPServer) -> list:
    return [
        {"name": tool.name, "description": tool.description}
        for tool in await server.list_tools()
    ]


def _check() -> tuple:
    """深检查：订单库能不能打开、当前开放了哪些工具。返回 (报告, 是否通过)。"""
    report = {
        "writes_enabled": settings.MCP_ENABLE_WRITES,
        "pii_enabled": settings.MCP_ENABLE_PII,
        "database": None,
    }
    try:
        from ecom.orchestrator.db.engine import get_db

        db = get_db()
        report["database"] = {"backend": db.backend, "orders": db.count("orders")}
        healthy = True
    except Exception as exc:  # noqa: BLE001 —— 检查命令要把原因说出来
        report["database"] = {"error": f"{type(exc).__name__}: {exc}"}
        healthy = False
    return report, healthy


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ecom.mcp.server",
        description="电商客服 MCP 工具服务（stdio）。不带参数时直接跑协议。",
    )
    parser.add_argument("--list-tools", action="store_true", help="打印当前开放的工具后退出")
    parser.add_argument("--check", action="store_true", help="检查订单库与开关后退出")
    args = parser.parse_args(argv)

    server = build_server()

    if args.list_tools:
        print(json.dumps(asyncio.run(_describe(server)), ensure_ascii=False, indent=2))
        return 0

    if args.check:
        report, healthy = _check()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if healthy else 1

    asyncio.run(server.run_stdio_async())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
