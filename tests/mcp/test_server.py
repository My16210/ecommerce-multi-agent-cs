"""
MCP 服务端的接线测试：**工具列表随开关变，工具调用走同一个信封**。

这里只验"协议层有没有把函数接对"，业务行为在 test_tools.py。所以调用
`get_order` 时故意用非法订单号——它在校验阶段就返回，不碰真库，也就不需要
灌数据。
"""
import asyncio
import json

import pytest

pytest.importorskip("mcp")

from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

from ecom.mcp import server as mcp_server  # noqa: E402


def _names(server) -> set:
    return {tool.name for tool in asyncio.run(server.list_tools())}


def _call(server, name: str, arguments: dict) -> dict:
    result = asyncio.run(server.call_tool(name, arguments))
    assert result.is_error is False
    return json.loads(result.content[0].text)


def test_default_surface_has_only_read_tools():
    assert _names(mcp_server.build_server()) == {
        "get_order",
        "get_order_returns",
        "search_knowledge",
    }


def test_customer_tool_is_registered_only_when_enabled(monkeypatch):
    monkeypatch.setattr(mcp_server.settings, "MCP_ENABLE_PII", True)
    assert "list_customer_orders" in _names(mcp_server.build_server())


def test_write_tool_is_registered_only_when_enabled(monkeypatch):
    monkeypatch.setattr(mcp_server.settings, "MCP_ENABLE_WRITES", True)
    assert "create_return" in _names(mcp_server.build_server())


def test_unknown_tool_is_not_callable():
    with pytest.raises(ToolError):
        asyncio.run(mcp_server.build_server().call_tool("drop_table", {}))


def test_tool_call_returns_json_envelope():
    out = _call(mcp_server.build_server(), "get_order", {"order_id": "abc"})
    assert out["ok"] is False
    assert out["error"]["code"] == "invalid_input"
    assert out["meta"]["elapsed_ms"] >= 0


def test_list_tools_command_prints_json(capsys):
    assert mcp_server.main(["--list-tools"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {item["name"] for item in payload} == {
        "get_order",
        "get_order_returns",
        "search_knowledge",
    }
