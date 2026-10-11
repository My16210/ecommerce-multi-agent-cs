"""
测试公用件。

**这里没有一条测试会联网。** 所有模型调用都用下面的 `FakeLLM` 顶掉。
理由不只是"跑得快"：真调模型意味着测试结果随模型版本、网络、当天负载浮动，
那样的测试没法当回归用——它昨天红今天绿，你只会学会忽略它。
"""
import shutil
from pathlib import Path

import pytest

from ecom.orchestrator.db.engine import Database
from ecom.orchestrator.db.seed import seed


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


@pytest.fixture(scope="session")
def _seed_template(tmp_path_factory) -> Path:
    """
    整个测试会话只灌一次种子库，落在会话自己的 tmp 目录里，供各测试复制。

    种子换成真实的 65752 笔订单 + 62897 条物流之后，这一句 `seed()` 要读 7.8MB
    CSV 再写 12.8 万行，约 0.5 秒。原来"每个测试各灌一遍"的写法在这里就变成了
    纯粹的重做——本文件 29 条测试要跑 15 秒，而它们要的其实是**同一份初始状态**：
    没有任何一条测试去改这个库的初始内容，改的都是自己往里写的那几笔。
    """
    path = tmp_path_factory.mktemp("order_seed") / "template.sqlite"
    database = Database(path=path, dsn="")
    database.init_schema()
    seed(database, force=True)
    return path


@pytest.fixture
def db(tmp_path, _seed_template) -> Database:
    """
    一个灌好种子数据的临时 SQLite，与真实 data/orders.sqlite 完全隔离。

    实现是"复制模板"，不是"各灌一遍"（见 `_seed_template`）。复制一个几 MB 的
    库文件是毫秒级，同时**保住了每个测试拿到独立库**这件事——这一点不能丢：
    好几条测试的断言就是"库里没多出东西"（`get_returns(...) == []`），
    要是几条测试共用同一个库，它们会互相把对方的结果看进去，红了还找不到原因。
    """
    path = tmp_path / "orders.sqlite"
    shutil.copyfile(_seed_template, path)
    return Database(path=path, dsn="")


@pytest.fixture(autouse=True)
def _no_trace_file(tmp_path, monkeypatch):
    """
    把 trace 的输出目录换到 tmp：测试绝不该往仓库的 logs/ 里写东西
    （那会让 git status 永远脏，也会把测试的假问题混进真实的请求日志里）。
    """
    monkeypatch.setattr("ecom.orchestrator.utils.trace.LOGS_DIR", Path(tmp_path) / "logs")
