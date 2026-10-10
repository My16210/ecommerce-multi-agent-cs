"""图结构与路由的离线测试（不需要 API key）。

build_graph() 会导入所有节点，因此需要装齐 langgraph / langchain-* 才能收集。
"""

import pytest

from ecom.analytics.core.state import make_initial_state


def _state(**overrides):
    state = make_initial_state("测试问题")
    state.update(overrides)
    return state


def test_after_execute_routes_to_self_correct_on_error():
    from ecom.analytics.core.graph import after_execute

    assert after_execute(_state(exec_error="boom", retry_count=0)) == "self_correct"
    assert after_execute(_state(exec_error="boom", retry_count=2)) == "self_correct"


def test_after_execute_gives_up_after_max_retry():
    from ecom.analytics.core.graph import MAX_EXEC_RETRY, after_execute

    assert after_execute(_state(exec_error="boom", retry_count=MAX_EXEC_RETRY)) == "report"


def test_after_execute_reviews_only_when_needed():
    from ecom.analytics.core.graph import after_execute

    assert after_execute(_state(need_review=True, review_round=0)) == "reviewer"
    # 简单问题直达报告，不调 Qwen
    assert after_execute(_state(need_review=False, review_round=0)) == "report"


def test_after_execute_review_quota_exhausted_goes_to_report():
    from ecom.analytics.core.graph import MAX_REVIEW_ROUNDS, after_execute

    assert after_execute(_state(need_review=True, review_round=MAX_REVIEW_ROUNDS)) == "report"


def test_after_review_routes():
    from ecom.analytics.core.graph import MAX_REVIEW_ROUNDS, after_review

    assert after_review(_state(review_passed=True, review_round=1)) == "report"
    assert after_review(_state(review_passed=False, review_round=1)) == "self_correct"
    # 审查额度用尽：带风险出报告，不再回环
    assert after_review(_state(review_passed=False, review_round=MAX_REVIEW_ROUNDS)) == "report"


def test_routing_loops_are_bounded():
    """两个环各自被独立计数器约束，最坏轮数有界，不会死循环。"""
    from ecom.analytics.core.graph import (
        MAX_EXEC_RETRY,
        MAX_REVIEW_ROUNDS,
        after_execute,
        after_review,
    )

    # 执行环：retry_count 递增到 MAX_EXEC_RETRY 后必然离开
    route = after_execute(_state(exec_error="boom", retry_count=MAX_EXEC_RETRY))
    assert route == "report"

    # 审查环：review_round 递增到 MAX_REVIEW_ROUNDS 后必然离开
    assert after_review(_state(review_passed=False, review_round=MAX_REVIEW_ROUNDS)) == "report"

    # 审查驱动的重跑仍然受执行环计数约束
    assert after_execute(
        _state(exec_error="boom", retry_count=MAX_EXEC_RETRY, need_review=True, review_round=1)
    ) == "report"


def test_graph_compiles():
    pytest.importorskip("langgraph")
    from ecom.analytics.core.graph import build_graph

    assert build_graph() is not None
