"""路由：解析守卫 / 规则兜底 / 置信度补位 / 排除已试 / token 记账。"""
import pytest

from ecom import demo
from ecom.orchestrator.core.router import (
    CONFIDENCE_FLOOR,
    _cold_plan,
    _rule_route,
    parse_route,
    route,
)
from ecom.orchestrator.utils.ids import ORDER_ID_RE
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
    # 客户号是 C 开头，里面的 20755 **不能**被当成订单号（`(?<!\\w)` 挡的就是这个）。
    # 但"客户 C20755 买了什么"确实是订单问题，由单独的客户号规则选 order——
    # 这两件事分开断言，免得以后有人把订单号正则放宽到能吃下客户号。
    assert ORDER_ID_RE.search("客户C20755") is None
    assert _rule_route("客户C20755") == ["order"]


def test_rules_stay_silent_on_small_talk():
    # 没命中就是没命中。**别硬塞一个 worker**——那是 fallback 层的事。
    assert _rule_route("你好呀") == []


@pytest.mark.parametrize(
    "question, expected",
    [
        # 客户号：没有订单号、也没有订单词，但问的就是某个客户的订单。
        ("客户 C20755 买了什么", ["order"]),
        # 「我要退货」是**要办**；「退货政策」是**问规定**。
        ("我要退货", ["order"]),
        ("退货政策是什么", ["knowledge"]),
        # 报表 = 把数据汇总出来，不特指某一单。
        ("帮我做个销售报表", ["analytics"]),
        # 「发货方式」是统计维度；「发货延迟是谁的责任」问规则；
        # 只有"这一单发没发"才是订单问题。
        ("哪种发货方式最容易延迟", ["analytics"]),
        ("发货延迟是谁的责任", ["knowledge"]),
        ("发货了吗", ["order"]),
        # 「能退吗」是「能不能退」的口语变体：既要查单也要政策。
        ("52644 能退吗", ["order", "knowledge"]),
    ],
)
def test_rules_cover_the_phrasings_that_used_to_fall_through(question, expected):
    """规则层的**词宽窄**，不是"关键词越多越好"。

    这些句子原来要么掉进 fallback、要么被别的 worker 的宽词抢走。判定方式
    和评测一致：什么都没命中时走 `_cold_plan()`，所以这里也照它算。
    """
    assert set(_rule_route(question) or _cold_plan()) == set(expected)


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


# ---------------------------------------------------------------------------
# 跨轮指代：路由也得知道这句话在说上一轮的哪件事
# ---------------------------------------------------------------------------
def _history_with_52644() -> list:
    return [
        {"role": "user", "content": "订单 52644 到哪了"},
        {"role": "assistant", "content": "订单 52644 的物流：目的地 San Salvador。"},
    ]


def test_the_router_prompt_carries_the_previous_turn():
    """LLM 那一路必须看得见上文——没有它，「那能退吗」在分类器眼里是一句没主语的话。

    断言落在"假模型收到的 prompt 里有什么"上：这是能验的**可观察契约**，
    不必去猜模型的判断。
    """
    fake = FakeLLM('{"workers": ["order"], "confidence": 0.9}')
    route("那能退吗", llm=fake, chat_history=_history_with_52644())
    prompt = fake.calls[0][1][1]
    assert "52644" in prompt
    assert "用户" in prompt


def test_the_rule_layer_carries_the_entity_for_a_followup():
    """真的出过事的那条路：LLM 吐了个空计划，兜底规则必须自己接得上。

    （README「已知限制」记的那个失败，就是发生在这一层——所以光给 LLM 加
      上文不够，`_rule_route` 也得携带实体。）
    """
    result = route("那能退吗", llm=FakeLLM("不是 JSON"), chat_history=_history_with_52644())
    # 「能退吗」本身也是政策问题（knowledge），所以这一轮是 order + knowledge；
    # 断集合而不是断顺序：两个 worker 谁先跑不影响结果，规则表里谁先命中却
    # 会随加词变化，钉死顺序只会让这条测试变成"词表快照"。
    assert set(result["plan"]) == {"order", "knowledge"}
    assert result["source"] == "rule"


def test_a_followup_without_an_anaphora_word_does_not_summon_the_order_worker():
    """**门槛管的是"接着上一轮说"，不是"上一轮说过订单"。**

    少了这条，「退货政策是什么」会因为在订单问题后面出现而顺带带上 order——
    一个纯政策问题被拉去查订单，正是"往关键词表里加词做指代消解"的典型翻车。
    """
    result = route("退货政策是什么", llm=FakeLLM("不是 JSON"), chat_history=_history_with_52644())
    assert "order" not in result["plan"]


def test_routing_without_history_stays_single_turn():
    # 不传 chat_history = 单轮。行为必须和加指代消解之前**一模一样**：
    # 「那能退吗」没有上文就不该冒出 order。
    result = route("那能退吗", llm=FakeLLM("不是 JSON"))
    assert "order" not in result["plan"]


# ---------------------------------------------------------------------------
# 演示模式：数据分析连"被选中"都不该发生
# ---------------------------------------------------------------------------
def test_demo_mode_drops_analytics_from_an_llm_plan_and_falls_back(monkeypatch):
    """只从计划里删名字不够：删空了要退回知识库，否则用户只得到一句"转人工"。

    这里刻意让 LLM 给高置信度——如果过滤写在置信度补位之前，它会把 analytics
    删掉之后什么都不补，计划为空。
    """
    monkeypatch.setattr(demo, "DEMO_MODE", True)

    result = route(
        "哪个地区卖得最好",
        llm=FakeLLM('{"workers": ["analytics"], "confidence": 0.95}'),
    )

    assert result["plan"] == ["knowledge"]
    assert "演示环境" in result["reason"]


def test_demo_mode_filters_the_rule_layer_too(monkeypatch):
    """LLM 挂了走规则层——规则层同样会命中 analytics，过滤必须在最终出口。"""
    monkeypatch.setattr(demo, "DEMO_MODE", True)

    result = route("哪个地区卖得最好", llm=ExplodingLLM())

    assert result["plan"] == ["knowledge"]
    assert result["source"] == "rule"


def test_demo_mode_does_not_offer_analytics_in_the_prompt(monkeypatch):
    """提示词里不列一个已经关掉的单元：让模型去选它只会白跑一次。"""
    monkeypatch.setattr(demo, "DEMO_MODE", True)
    fake = FakeLLM('{"workers": ["knowledge"], "confidence": 0.9}')

    route("随便聊聊", llm=fake)

    prompt = fake.calls[0][1][1]
    assert "analytics" not in prompt
    assert "knowledge" in prompt


def test_demo_mode_keeps_all_workers_in_the_plan_when_they_are_allowed(monkeypatch):
    monkeypatch.setattr(demo, "DEMO_MODE", True)

    result = route(
        "订单 52644 到哪了，顺便问下退货政策",
        llm=FakeLLM('{"workers": ["order", "knowledge"], "confidence": 0.9}'),
    )

    assert result["plan"] == ["order", "knowledge"]


def test_demo_mode_does_not_readd_an_already_tried_knowledge(monkeypatch):
    """回环时不能为了"补位"把已经试过的知识库再拉一遍——那就是原地打转。"""
    monkeypatch.setattr(demo, "DEMO_MODE", True)

    result = route(
        "哪个地区卖得最好",
        tried=["knowledge"],
        llm=FakeLLM('{"workers": ["analytics"], "confidence": 0.95}'),
    )

    assert result["plan"] == []
