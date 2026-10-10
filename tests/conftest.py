"""
测试公用件。

**这里没有一条测试会联网。** 所有模型调用都用下面的 `FakeLLM` 顶掉。
理由不只是"跑得快"：真调模型意味着测试结果随模型版本、网络、当天负载浮动，
那样的测试没法当回归用——它昨天红今天绿，你只会学会忽略它。
"""
from pathlib import Path

import pytest

from cs.db.engine import Database
from cs.db.seed import seed


class FakeMessage:
    """冒充 LangChain 的 AIMessage：只要 content 和 usage_metadata 两个字段。"""

    def __init__(self, content, usage=None):
        self.content = content
        self.usage_metadata = usage if usage is not None else {"input_tokens": 10, "output_tokens": 5}


class FakeLLM:
    """
    按调用顺序吐预设内容；用完就重复最后一条。

    重复最后一条是刻意的：路由失败会重试 ROUTER_ATTEMPTS 次，测试里想表达的是
    "一直坏"，而不是"第三次恰好好了"。要表达后者，就往 contents 里多塞几条。
    """

    def __init__(self, *contents, usage=None):
        self.contents = list(contents) or [""]
        self.usage = usage
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        index = min(len(self.calls) - 1, len(self.contents) - 1)
        return FakeMessage(self.contents[index], self.usage)

    @property
    def call_count(self) -> int:
        return len(self.calls)


class ExplodingLLM:
    """调用即抛。用来验证"模型挂了系统还在"。"""

    def __init__(self, exc=None):
        self.exc = exc or RuntimeError("模型不可用")
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        raise self.exc


@pytest.fixture
def db(tmp_path) -> Database:
    """一个灌好种子数据的临时 SQLite，与真实 data/orders.sqlite 完全隔离。"""
    database = Database(path=tmp_path / "orders.sqlite", dsn="")
    database.init_schema()
    seed(database, force=True)
    return database


@pytest.fixture(autouse=True)
def _no_trace_file(tmp_path, monkeypatch):
    """
    把 trace 的输出目录换到 tmp：测试绝不该往仓库的 logs/ 里写东西
    （那会让 git status 永远脏，也会把测试的假问题混进真实的请求日志里）。
    """
    monkeypatch.setattr("cs.utils.trace.LOGS_DIR", Path(tmp_path) / "logs")
