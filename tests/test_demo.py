"""
公网演示的四条闸：邀请码 / 限流 / 每日 token 预算 / 只读策略。

这一组测试守的不是"功能好不好用"，而是**失效时倒向哪一边**：
  · 没配邀请码 → 拒绝所有人，而不是放行所有人；
  · 预算账本打不开 → 拒绝，而不是继续花；
  · 演示模式关掉 → 所有闸都不生效（本地和 CI 不能被它挡住）；
  · 禁用的 worker 从任意名单里摘掉（路由、回环、健康检查共用同一份规则）。

最后一条最容易写错：只从"这一轮的计划"里删名字，回环重路由时会从全量 worker
里重新挑——数据分析就又回来了，而它执行的正是模型生成的代码。
"""
import pytest

from ecom import demo


@pytest.fixture
def demo_on(monkeypatch, tmp_path):
    monkeypatch.setattr(demo, "DEMO_MODE", True)
    monkeypatch.setattr(demo, "DEMO_INVITE_CODE", "let-me-in")
    monkeypatch.setattr(demo, "DEMO_RATE_LIMIT_PER_MINUTE", 2)
    monkeypatch.setattr(demo, "DEMO_DAILY_TOKEN_BUDGET", 100)
    monkeypatch.setattr(demo, "DEMO_USAGE_DB_PATH", tmp_path / "usage.sqlite")
    demo.reset_rate_limits()
    yield
    demo.reset_rate_limits()


# ---------------------------------------------------------------------------
# 总开关：关掉时一切都该是"放行"
# ---------------------------------------------------------------------------
def test_every_guard_is_a_no_op_when_demo_mode_is_off(monkeypatch, tmp_path):
    monkeypatch.setattr(demo, "DEMO_MODE", False)
    monkeypatch.setattr(demo, "DEMO_USAGE_DB_PATH", tmp_path / "usage.sqlite")

    assert demo.check_invite("") is True
    assert demo.writes_allowed() is True
    assert demo.filter_workers(["knowledge", "order", "analytics"]) == [
        "knowledge", "order", "analytics",
    ]
    for _ in range(50):
        assert demo.allow_request("anyone") is True
    assert demo.budget_exhausted() is False
    # 没开演示模式时记账也不该在磁盘上留下一个账本
    assert demo.add_tokens({"in": 10, "out": 20}) == 0
    assert not (tmp_path / "usage.sqlite").exists()


# ---------------------------------------------------------------------------
# 邀请码：fail closed
# ---------------------------------------------------------------------------
def test_the_invite_code_must_match_exactly(demo_on):
    assert demo.check_invite("let-me-in") is True
    assert demo.check_invite("let-me-in ") is False
    assert demo.check_invite("") is False


def test_demo_mode_without_a_configured_code_rejects_everyone(demo_on, monkeypatch):
    """**这条是刻意的。**

    "忘了配邀请码"最坏的默认值是"谁都进得来"——那和一个没有门的门卫是同一种
    东西，而且看不出来。所以这里倒向拒绝。
    """
    monkeypatch.setattr(demo, "DEMO_INVITE_CODE", "")

    assert demo.invite_configured() is False
    assert demo.check_invite("") is False
    assert demo.check_invite("anything") is False


def test_a_non_ascii_invite_code_does_not_explode(demo_on, monkeypatch):
    """中文邀请码是很自然会设出来的东西；比较前不 encode 会抛 TypeError。"""
    monkeypatch.setattr(demo, "DEMO_INVITE_CODE", "演示口令")

    assert demo.check_invite("演示口令") is True
    assert demo.check_invite("别的口令") is False


# ---------------------------------------------------------------------------
# 限流：每个 key 一个滑动窗口
# ---------------------------------------------------------------------------
def test_a_client_is_limited_to_the_configured_rate(demo_on):
    assert demo.allow_request("a") is True
    assert demo.allow_request("a") is True
    assert demo.allow_request("a") is False


def test_one_client_hitting_the_limit_does_not_block_another(demo_on):
    assert demo.allow_request("a") is True
    assert demo.allow_request("a") is True
    assert demo.allow_request("a") is False

    assert demo.allow_request("b") is True


def test_the_rate_limit_can_be_disabled_with_zero(demo_on, monkeypatch):
    monkeypatch.setattr(demo, "DEMO_RATE_LIMIT_PER_MINUTE", 0)

    assert all(demo.allow_request("a") for _ in range(50))


# ---------------------------------------------------------------------------
# 每日 token 预算：磁盘账本，重启也算数
# ---------------------------------------------------------------------------
def test_tokens_are_summed_until_the_budget_runs_out(demo_on):
    assert demo.budget_exhausted() is False

    assert demo.add_tokens({"in": 40, "out": 20}) == 60
    assert demo.tokens_used_today() == 60
    assert demo.budget_remaining() == 40
    assert demo.budget_exhausted() is False

    assert demo.add_tokens({"in": 30, "out": 10}) == 40
    assert demo.budget_exhausted() is True
    assert demo.budget_remaining() == 0


def test_the_ledger_is_shared_across_reconnects(demo_on):
    """每次记账都新开连接——这模拟的是"另一个进程/重启之后再算"。

    如果预算只活在进程内变量里，多副本各算一份，实际花掉的是 N 倍。
    """
    demo.add_tokens({"in": 60, "out": 0})
    # 这里没有任何"同一个连接"的假设：tokens_used_today 会重新打开账本
    assert demo.tokens_used_today() == 60


def test_empty_usage_does_not_create_a_row(demo_on):
    assert demo.add_tokens({}) == 0
    assert demo.add_tokens({"in": 0, "out": 0}) == 0
    assert demo.add_tokens(None) == 0
    assert demo.tokens_used_today() == 0


def test_a_broken_ledger_fails_closed(demo_on, monkeypatch):
    """成本闸失效时倒向"不花钱"那一边。

    反过来的话，磁盘一出问题，预算就会在没人知道的情况下被烧穿——而演示环境
    停掉是立刻看得见的。
    """
    def _boom():
        raise OSError("disk gone")

    monkeypatch.setattr(demo, "_connect", _boom)

    assert demo.budget_exhausted() is True
    assert demo.add_tokens({"in": 10, "out": 10}) == 0   # 记账失败只打日志，不抛


# ---------------------------------------------------------------------------
# 只读策略
# ---------------------------------------------------------------------------
def test_demo_mode_is_read_only_and_has_no_analytics_worker(demo_on):
    assert demo.writes_allowed() is False
    assert demo.worker_allowed("knowledge") is True
    assert demo.worker_allowed("order") is True
    assert demo.worker_allowed("analytics") is False
    assert demo.filter_workers(["analytics", "order", "knowledge"]) == [
        "order", "knowledge",
    ]


def test_the_ledger_uses_wal_so_readers_and_writers_do_not_block_each_other(demo_on):
    """账本是 API 线程 / Streamlit 重跑 / CLI 共写的。

    默认的 rollback journal 下，一次读会挡住写；并发时表现为 "database is
    locked"，而记账失败只会打日志——预算于是被悄悄少记。WAL + busy_timeout
    让短暂的锁冲突变成等待。
    """
    import sqlite3

    demo.add_tokens({"in": 1, "out": 0})
    with sqlite3.connect(str(demo.DEMO_USAGE_DB_PATH)) as conn:
        assert str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal"
