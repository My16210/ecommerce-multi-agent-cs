"""路由：解析守卫 / 规则兜底 / 置信度补位 / 排除已试 / token 记账。"""
import pytest

from cs.core.router import CONFIDENCE_FLOOR, _rule_route, parse_route, route
from tests.conftest import ExplodingLLM, FakeLLM


# ---------------------------------------------------------------------------
# parse_route：把模型的自由输出收拾成可信的形状
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "content",
    [
        '{"workers": ["order"], "reason": "问订单", "confidence": 0.9}',
        '```json\n{"workers": ["order"], "reason": "问订单", "confidence": 0.9}\n```',
        '好的，我的判断是：{"workers": ["order"], "reason": "问订单", "confidence": 0.9}',
    ],
)
def test_parse_accepts_bare_fenced_and_padded_json(content):
    parsed = parse_route(content)
    assert parsed["plan"] == ["order"]
    assert parsed["confidence"] == pytest.approx(0.9)
    assert parsed["parse_error"] is False


def test_parse_drops_unknown_worker_names_instead_of_guessing():
    # 模型编了个 "shipping" 出来。**丢**，不猜成 order——
    # 猜错等于调错单元，比少调一个更贵。
    parsed = parse_route('{"workers": ["shipping", "order", "order"], "confidence": 1}')
    assert parsed["plan"] == ["order"]


def test_parse_clamps_percent_style_confidence():
    assert parse_route('{"workers": ["order"], "confidence": 95}')["confidence"] == pytest.approx(0.95)
    assert parse_route('{"workers": ["order"], "confidence": 500}')["confidence"] == 1.0
    assert parse_route('{"workers": ["order"], "confidence": -3}')["confidence"] == 0.0


def test_parse_treats_missing_or_bogus_confidence_as_zero():
    # 0 = "不知道怎么信它"，于是下游会补一个 knowledge。凭空相信才是错的。
    assert parse_route('{"workers": ["order"]}')["confidence"] == 0.0
    assert parse_route('{"workers": ["order"], "confidence": "高"}')["confidence"] == 0.0


def test_parse_reports_failure_without_raising():
    for garbage in ["", "我建议交给订单系统", "[1, 2, 3]", '{"workers": "order"}']:
        parsed = parse_route(garbage)
        assert parsed["plan"] == []
        assert parsed["parse_error"] is True


# ---------------------------------------------------------------------------
# 规则兜底
# ---------------------------------------------------------------------------
def test_rules_pick_up_order_id_and_multi_intent():
    assert _rule_route("订单 100002 到哪了") == ["order"]
    plan = _rule_route("订单 100002 到哪了，顺便问下退货政策")
    assert set(plan) == {"order", "knowledge"}
    assert _rule_route("哪个地区卖得最好") == ["analytics"]


def test_rules_stay_silent_on_small_talk():
    # 没命中就是没命中。**别硬塞一个 worker**——那是 fallback 层的事。
    assert _rule_route("你好呀") == []


# ---------------------------------------------------------------------------
# route：三层优先级
# ---------------------------------------------------------------------------
def test_llm_plan_is_used_when_confidence_is_high():
    result = route("订单 100002 到哪了", llm=FakeLLM('{"workers": ["order"], "confidence": 0.95}'))
    assert result["plan"] == ["order"]
    assert result["source"] == "llm"


def test_low_confidence_appends_knowledge_as_a_safety_net():
    # 代价不对称：漏调一个用户看得出来，多调一个用户看不出来。所以低置信时补。
    result = route("这个东西能不能退", llm=FakeLLM('{"workers": ["order"], "confidence": 0.3}'))
    assert "knowledge" in result["plan"]
    assert result["confidence"] < CONFIDENCE_FLOOR


def test_falls_back_to_rules_when_output_is_unparseable():
    result = route("订单 100002 到哪了", llm=FakeLLM("我不知道该给谁"))
    assert result["plan"] == ["order"]
    assert result["source"] == "rule"


def test_falls_back_to_rules_when_the_model_itself_is_broken():
    # 这一条是"规则兜底"存在的全部理由：模型这条路整个不可用。
    result = route("订单 100002 到哪了", llm=ExplodingLLM())
    assert result["plan"] == ["order"]
    assert result["source"] == "rule"


def test_defaults_to_knowledge_when_nothing_matches():
    result = route("随便聊聊", llm=FakeLLM("不是 JSON"))
    assert result["plan"] == ["knowledge"]
    assert result["source"] == "fallback"


def test_tried_workers_are_excluded_so_the_loop_cannot_spin():
    # LLM 还是说 order，但 order 试过了 → 计划必须变空。
    # 空计划会被 aggregate 读成"没得再试了"，回环就此停住。
    result = route("订单 100002 到哪了", tried=["order"], llm=FakeLLM('{"workers": ["order"], "confidence": 1.0}'))
    assert result["plan"] == []


def test_low_confidence_does_not_re_add_an_already_tried_worker():
    result = route("订单 100002 到哪了", tried=["knowledge"], llm=FakeLLM('{"workers": ["order"], "confidence": 0.2}'))
    assert result["plan"] == ["order"]


def test_retry_tokens_all_counted():
    # 三次都没解析出来 → 三次调用都要记账。只记最后一次会让成本表静静少算。
    result = route("订单 100002 到哪了", llm=FakeLLM("坏输出"))
    assert result["tokens"] == {"in": 30, "out": 15}
