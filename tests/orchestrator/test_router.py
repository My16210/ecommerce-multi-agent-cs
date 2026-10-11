"""路由：解析守卫 / 规则兜底 / 置信度补位 / 排除已试 / token 记账。"""
import pytest

from ecom.orchestrator.core.router import CONFIDENCE_FLOOR, _rule_route, parse_route, route
from tests.orchestrator.conftest import ExplodingLLM, FakeLLM


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
    assert _rule_route("订单 52644 到哪了") == ["order"]
    plan = _rule_route("订单 52644 到哪了，顺便问下退货政策")
    assert set(plan) == {"order", "knowledge"}
    assert _rule_route("哪个地区卖得最好") == ["analytics"]


def test_the_order_id_rule_matches_the_five_digit_ids_we_actually_have():
    """裸订单号（一个关键词都没有）也要能路由到 order。

    这条守的是**路由层和抽取层对"订单号是几位"的理解必须一致**。DataCo 的订单号
    最大 77204，只有 5 位；而这条规则原来是 `\\d{6,}`——对真实订单**一笔都命中
    不了**，同时会把发票号、长流水号这种 6 位以上的数字路由成"查订单"。最麻烦的
    是两处各自看都"对"：路由认得出的，抽取层认不出。

    **现在两处是同一个 `ORDER_ID_RE`（ecom/orchestrator/utils/ids.py），不可能再分叉。**
    这条测试因此从"盯着两个常量别漂"变成了"盯着那个唯一的形状别再被改坏"——
    它仍然值得留着，因为它同时钉住了 5 位门槛和 `(?<!\\w)` 这两件事的行为。

    代价也记在这里：门槛定在 5 位，12 位的发票号会被顺带路由到 order。它属于
    "宁可多问一次"的方向——order worker 查不到会说"查不到订单号 XXX"，
    而不是给出一个错的答案。
    """
    assert _rule_route("52644") == ["order"]
    # 4 位不行：那多半是年份。"2025 年买的还能退吗"里没有订单号。
    assert _rule_route("2025") == []
    # 客户号是 C 开头，不该被当成订单号（路由和抽取两层的 `(?<!\\w)` 是同一个理由）
    assert _rule_route("客户C20755") == []


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
