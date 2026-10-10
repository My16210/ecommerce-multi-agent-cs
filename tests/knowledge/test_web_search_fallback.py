"""
联网搜索的降级路径。

Tavily 是可选依赖（pyproject.toml 里的 search extra）。「.env 里配了
TAVILY_API_KEY、但包没装」是最容易踩的一步：默认的 ModuleNotFoundError
只说"没有这个模块"，不说怎么办，用户很容易以为 Tavily 服务挂了。

这里钉住两件事：报错要说清楚怎么修；以及在报错之后**照样**降级到 ddgs，
不能因为一个可选的增强把联网这一路整个丢掉。

不联网：ddgs 那一路被替换掉了。
"""
import importlib
import sys

import pytest


def _module():
    """ecom/knowledge/web_search/__init__.py 把同名函数导了出去，所以要直接取子模块"""
    return importlib.import_module("ecom.knowledge.web_search.web_search")


def test_missing_tavily_package_gives_an_actionable_message(monkeypatch):
    # 把 sys.modules 里的名字设成 None，import 就会抛 ImportError——
    # 这样不管本机装没装 tavily，测的都是"没装"那条路
    monkeypatch.setitem(sys.modules, "tavily", None)
    module = _module()
    monkeypatch.setattr(module, "TAVILY_API_KEY", "tvly-fake")

    with pytest.raises(RuntimeError, match="uv sync --extra search"):
        module._search_by_tavily("测试", 3)


def test_web_search_still_falls_back_to_ddgs(monkeypatch):
    """Tavily 不可用只是少一路增强，不能把联网整个搞没"""
    monkeypatch.setitem(sys.modules, "tavily", None)
    module = _module()
    monkeypatch.setattr(module, "TAVILY_API_KEY", "tvly-fake")
    monkeypatch.setattr(
        module,
        "_search_by_ddgs",
        lambda query, max_results, timeout=None: [
            {"text": "兜底结果", "title": "t", "url": "u", "source": "web"}
        ],
    )

    results = module.web_search("测试", 2)

    assert len(results) == 1
    assert results[0]["text"] == "兜底结果"


def test_ddgs_failure_returns_empty_list(monkeypatch):
    """两路都挂也不能抛异常——联网只是增强"""

    def boom(query, max_results, timeout=None):
        raise RuntimeError("ddgs 也挂了")

    module = _module()
    monkeypatch.setattr(module, "TAVILY_API_KEY", "")
    monkeypatch.setattr(module, "_search_by_ddgs", boom)

    assert module.web_search("测试", 2) == []
