"""
MCP 工具层的测试夹具。

**不灌全量种子。** 这里要验的是"输入怎么校验、输出裁了哪些字段、开关怎么拦"，
三笔手写订单就够了；`tests/orchestrator` 里那份 65752 笔的种子是给对话链路
和业务规则用的，搬过来只会让这组测试从毫秒级变成秒级。

`_mcp_defaults` 是 autouse 的：`.env` 里可能开着 PII / 写操作，而"默认关闭"
本身就是要验的行为。每条用例都在**确定的默认值**上开始，要开的自己再
monkeypatch——不这么写，测试结果就取决于跑测试那台机器的 .env。
"""
import pytest

from ecom.mcp import settings, tools
from ecom.orchestrator.db.engine import Database


@pytest.fixture(autouse=True)
def _mcp_defaults(monkeypatch):
    monkeypatch.setattr(settings, "MCP_ENABLE_WRITES", False)
    monkeypatch.setattr(settings, "MCP_ENABLE_PII", False)
    # 演示模式的默认值也钉住：.env 里有没有 DEMO_MODE 不该改变这组测试的结论。
    monkeypatch.setattr("ecom.demo.DEMO_MODE", False)


@pytest.fixture
def db(tmp_path) -> Database:
    database = Database(path=tmp_path / "orders.sqlite", dsn="")
    database.init_schema()
    database.executemany(
        "INSERT INTO orders (order_id, customer_id, customer_name, status, amount, "
        "region, created_at, paid_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("52644", "C20755", "张三", "COMPLETE", 1234.5, "华东", "2024-01-02", None),
            ("52645", "C20755", "张三", "PENDING", 99.0, "华东", "2024-02-03", None),
            # 写操作成功路径单独用一笔干净订单，别和前两条的断言互相影响。
            ("52646", "C20756", "李四", "COMPLETE", 20.0, "华北", "2024-03-04", None),
        ],
    )
    database.execute(
        "INSERT INTO logistics (order_id, carrier, tracking_no, from_city, to_city, "
        "status, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("52644", None, None, None, "上海", "Advance shipping", "2024-01-05"),
    )
    database.execute(
        "INSERT INTO return_requests (return_id, order_id, reason, status, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("R0000000001", "52644", "尺码不合适", "待审核", "2024-01-06 10:00"),
    )
    return database


@pytest.fixture
def tools_module():
    """显式拿模块本身——测试里 monkeypatch 的是它的属性，不绕 `from ... import`。"""
    return tools
