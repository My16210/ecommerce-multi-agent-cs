

from ecom.analytics.core.state import SalesState, make_initial_state


def test_make_initial_state_has_every_key():
    state = make_initial_state("哪个地区卖得最好？")
    # 初始状态必须覆盖 SalesState 的全部字段，避免节点里 .get 取到意料之外的默认值
    assert set(state.keys()) == set(SalesState.__annotations__.keys())


def test_make_initial_state_defaults():
    state = make_initial_state("测试问题")
    assert state["question"] == "测试问题"
    assert state["exec_error"] is None
    assert state["retry_count"] == 0
    assert state["review_round"] == 0
    assert state["review_passed"] is False
    assert state["need_review"] is False
    assert state["chart_generated"] is False
    assert state["chart_path"] == ""
    assert state["total_tokens"] == 0
    assert state["messages"] == []


def test_make_initial_state_is_fresh_each_call():
    a = make_initial_state("a")
    b = make_initial_state("b")
    a["messages"].append({"role": "用户", "content": "x"})
    assert b["messages"] == []  # 不能共用同一个 list
