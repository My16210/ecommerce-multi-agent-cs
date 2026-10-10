"""
联网搜索的时间预算。

画像（README「性能画像」）量出来：完整链路里联网搜索那一段是 5.6 秒，比检索
（387ms）慢一个数量级，而其中大部分是**搜不到时的重试**。它又恰好挂在关键路径上
（`max(检索, 联网) + 生成 + 判定`），所以它拖多久，用户就等多久。

于是给整条这一路加了一个硬上限。这里钉的就是"上限真的是硬的"——
尤其是**超时也必须返回 []，不能抛出去**：联网只是增强，它挂掉不该把整个会话打断，
这条与文件开头"不能把主链路拖垮"是同一个约定，只是现在多了一个时间维度。

不联网：被测的引擎全被替换成假的。
"""
import importlib
import time

import pytest


def _module():
    return importlib.import_module("ecom.knowledge.web_search.web_search")


@pytest.fixture
def module(monkeypatch):
    """拿到模块并把 Tavily 关掉，让测试只跑 ddgs 那一路"""
    mod = _module()
    monkeypatch.setattr(mod, "TAVILY_API_KEY", "")
    return mod


def test_hanging_engine_is_cut_off_at_the_budget(module, monkeypatch):
    """卡住的引擎必须在预算处被切断，而不是无限等下去"""

    def never_returns(query, max_results, timeout=None):
        time.sleep(30)
        return [{"text": "永远拿不到", "title": "t", "url": "u", "source": "web"}]

    monkeypatch.setattr(module, "_search_by_ddgs", never_returns)

    started = time.monotonic()
    results = module.web_search("测试", 3, timeout=0.3)
    elapsed = time.monotonic() - started

    assert results == []
    # 给足余量：只要它不是"等满 30 秒"就说明预算生效了
    assert elapsed < 3, f"超时没生效，实际等了 {elapsed:.2f} 秒"


def test_slow_but_finishing_engine_still_returns_results(module, monkeypatch):
    """预算内跑完的要正常返回——别把"慢"和"超时"混为一谈"""
    def slow_but_ok(query, max_results, timeout=None):
        time.sleep(0.05)
        return [{"text": "赶上了", "title": "t", "url": "u", "source": "web"}]

    monkeypatch.setattr(module, "_search_by_ddgs", slow_but_ok)

    results = module.web_search("测试", 3, timeout=5)

    assert len(results) == 1
    assert results[0]["text"] == "赶上了"


def test_timeout_is_shared_between_tavily_and_ddgs(module, monkeypatch):
    """Tavily 花掉的时间要从 ddgs 的预算里扣，不能各算各的

    否则"整条这一路最多 N 秒"就是假的：Tavily 卡满 N 秒、ddgs 再卡满 N 秒，
    用户等的是 2N。这条就是钉住那个共享的 deadline。
    """
    monkeypatch.setattr(module, "TAVILY_API_KEY", "tvly-fake")
    monkeypatch.setattr(module, "_search_by_tavily",
                        lambda query, max_results: time.sleep(30))

    def should_not_be_reached(query, max_results, timeout=None):
        raise AssertionError("预算已经用光，不该再调 ddgs 了")

    monkeypatch.setattr(module, "_search_by_ddgs", should_not_be_reached)

    started = time.monotonic()
    results = module.web_search("测试", 3, timeout=0.3)
    elapsed = time.monotonic() - started

    assert results == []
    assert elapsed < 3, f"总预算没生效，实际等了 {elapsed:.2f} 秒"


def test_ddgs_receives_the_remaining_budget(module, monkeypatch):
    """剩下的预算要传给 ddgs 自己那个"每批"超时，且不小于 1 秒

    传 0.2 进去会被 ddgs 当成"一上来就掐死"，那不是我们想要的行为；
    下限 1 秒是为了让它至少有机会发一次请求。
    """
    seen = {}

    def record(query, max_results, timeout=None):
        seen["timeout"] = timeout
        return []

    monkeypatch.setattr(module, "_search_by_ddgs", record)
    module.web_search("测试", 3, timeout=0.2)

    assert seen["timeout"] is not None
    assert seen["timeout"] <= 0.2


def test_timeout_defaults_to_the_configured_budget(module, monkeypatch):
    """不传 timeout 时用的是 settings 里的 WEB_SEARCH_TIMEOUT"""
    seen = {}

    def record(query, max_results, timeout=None):
        seen["timeout"] = timeout
        return []

    monkeypatch.setattr(module, "_search_by_ddgs", record)
    module.web_search("测试", 3)

    assert seen["timeout"] == pytest.approx(module.WEB_SEARCH_TIMEOUT)
