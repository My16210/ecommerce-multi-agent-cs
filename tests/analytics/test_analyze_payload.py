"""守子进程那行 JSON 里 review_passed 的三态语义。

真跑出来的现象：问「总销售额是多少？」（简单问题，分类器判定不需要审查），
答案 1846604 完全正确，但 analyze 吐出的 review_passed 是 false——
因为它是 make_initial_state 里的初始值，审查压根没跑。
上层据此打了一条"结果可能不可靠"的告警、界面弹了黄条。
一个正确的回答被自己的系统举报了。

所以这里守的是：**"没审查"不能长得像"审查没过"**，但也不能被当成"审查通过"。
"""
from ecom.analytics.cli.analyze import review_outcome


def test_reviewed_and_passed():
    assert review_outcome({"review_passed": True, "review_round": 1}) is True


def test_reviewed_and_failed():
    assert review_outcome({"review_passed": False, "review_round": 1}) is False


def test_never_reviewed_is_none_not_false():
    """这就是那个 bug 的形状：round=0、passed 是初始值 False。"""
    assert review_outcome({"review_passed": False, "review_round": 0}) is None


def test_missing_keys_are_treated_as_never_reviewed():
    assert review_outcome({}) is None


def test_failed_then_exhausted_rounds_is_still_false():
    """审了、没过、额度用尽——这时必须报 False，不能因为"循环结束了"就洗白。"""
    assert review_outcome({"review_passed": False, "review_round": 2}) is False


def test_none_is_not_the_same_as_false():
    """三态的意义：下游 `is False` 的告警不该被 None 触发。"""
    never, failed = review_outcome({"review_round": 0}), review_outcome({"review_passed": False, "review_round": 1})
    assert never is None and failed is False
    assert (never is False) is False  # is False 严格比较，None 不会误命中
