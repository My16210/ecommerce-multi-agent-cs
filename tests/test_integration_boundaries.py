"""
跨仓库接入的边界。

这一组测的不是功能，是**接缝**——本项目靠"运行时引用两个 sibling 仓库"来复用它们，
接缝上出问题的表现往往不是报错，而是行为悄悄不对。所以接缝要单独钉住。
"""
import re
import sys
import types
from pathlib import Path

import pytest

from cs.settings import ROOT
from cs.workers.knowledge import _assert_imports_came_from


def test_we_do_not_take_over_the_sibling_repos_top_level_names():
    """
    同一进程里，顶层包名就是全局命名空间：谁先占到谁说了算，**而且失败时不会
    提示你是撞名了**。

    真出过事：本项目原先把配置放在 `config/settings.py`，和 rag_knowledge_base
    一样——听起来是"保持一致"，实际是把 rag 的 16 个 `from config.settings import ...`
    全部劫持到了我们这份上，报出来的是 `ImportError: cannot import name 'LOGS_PATH'`，
    而 LOGS_PATH 这件事本身和故障原因一点关系都没有。

    所以这条测试守的是：本项目永远不拥有 `src/` 和 `config/` 这两个名字。
    想再加顶层包时，**先想想另一个仓库有没有同名的**。
    """
    # 这两个名字属于 sibling 仓库（rag_knowledge_base 用 src/ 和 config/，
    # sales_agent 用 src/）。本项目一个都不能占。
    for name in ("src", "config"):
        assert not (ROOT / name / "__init__.py").exists(), (
            f"本项目不应该有顶层包 {name}/——会和 sibling 仓库撞名。"
            "配置在 cs/settings.py，业务代码在 cs/ 下面。"
        )


def test_our_own_package_is_still_there():
    # 上面那条要是被"把 cs/ 也删了"的方式满足，就没意义了。
    assert (ROOT / "cs" / "__init__.py").exists()


def test_the_origin_guard_catches_a_hijacked_module(monkeypatch, tmp_path):
    """
    光有"不许撞名"的约定不够——约定是给人看的，运行时要能自己发现。
    这条测 `_assert_imports_came_from`：把 config.settings 换成"来自别处"的模块，
    它必须立刻喊出来。
    """
    hijacked = types.ModuleType("config.settings")
    hijacked.__file__ = str(tmp_path / "elsewhere" / "settings.py")
    monkeypatch.setitem(sys.modules, "config.settings", hijacked)

    with pytest.raises(RuntimeError, match="顶层包名撞了"):
        _assert_imports_came_from(Path("/nonexistent/rag_knowledge_base"))


def test_the_origin_guard_stays_quiet_when_nothing_is_loaded(monkeypatch):
    # 没 import 过就说明没法判断，放行——不要在"知识库还没接进来"的时候误报。
    monkeypatch.setitem(sys.modules, "config.settings", None)
    monkeypatch.setitem(sys.modules, "src.core.graph", None)
    _assert_imports_came_from(Path("/nonexistent/rag_knowledge_base"))


def test_the_analytics_worker_goes_through_a_process_boundary():
    # 数据分析 worker 不 import sales_agent 的任何模块，只起子进程。
    # 这条测的是"它别哪天被人改成 import 了"——那时两个 src/ 就会在同一个进程里撞上。
    source = (ROOT / "cs" / "workers" / "analytics.py").read_text(encoding="utf-8")
    assert "subprocess" in source
    # 只看真正的 import 语句：这个文件的文档字符串里**必然**会出现
    # "from src.x import y"——它讲的就是撞名这件事，不该被自己的说明文字绊倒。
    assert not re.search(r"^\s*(from|import)\s+src\b", source, re.M)
