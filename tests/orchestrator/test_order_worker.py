"""订单：业务规则的边界 / 规则优先的意图抽取 / "办不了"不是"坏了"。"""
import pytest

from ecom.orchestrator.tools.order_tools import (
    OrderNotFound,
    ReturnNotAllowed,
    create_return,
    get_returns,
    list_customer_orders,
    query_order,
)
from ecom.orchestrator.workers.order import _rule_extract, extract_intent, run_order
from tests.orchestrator.conftest import ExplodingLLM


@pytest.fixture
def wired(db, monkeypatch):
    """
    把三个模块里的 get_db 都指到临时库，seed 变成空操作，模型句柄换成"调用即抛"。

    换掉 `ecom.orchestrator.workers.order.get_llm` 是**必须的**：.env 里有真 key 的话，
    run_order 在规则认不全时会真的发一次 API 请求——测试就变成了联网测试，
    而且会在别人机器上因为没 key 而红。
    """
    # 两处取库的地方都要换：工具层自己取一次，run_order 开头的幂等 seed 取一次。
    # （order.py 自己不再 import get_db，所以没有第三处。）
    monkeypatch.setattr("ecom.orchestrator.tools.order_tools.get_db", lambda: db)
    monkeypatch.setattr("ecom.orchestrator.db.seed.get_db", lambda: db)
    monkeypatch.setattr("ecom.orchestrator.workers.order.seed", lambda *a, **k: 0)
    monkeypatch.setattr("ecom.orchestrator.workers.order.get_llm", lambda *a, **k: ExplodingLLM())
    return db


# ---------------------------------------------------------------------------
# 工具层
# ---------------------------------------------------------------------------
def test_query_order_returns_the_order_and_its_logistics(db):
    data = query_order("100002", db)
    assert data["order"]["status"] == "运输中"
    assert data["order"]["amount"] == pytest.approx(159.0)
    assert data["logistics"]["carrier"] == "中通"


def test_an_order_without_logistics_is_not_an_error(db):
    # 已取消的订单本来就没有运单。物流为 None 是业务事实，不是数据缺失。
    data = query_order("100005", db)
    assert data["logistics"] is None


def test_unknown_order_id_raises_rather_than_returning_empty(db):
    # 返回空 dict 的话，调用方迟早会把它当成"一笔状态为空的订单"用掉。
    with pytest.raises(OrderNotFound):
        query_order("999999", db)


@pytest.mark.parametrize("order_id", ["100002", "100003"])
def test_returnable_statuses_accept_a_request(order_id, db):
    result = create_return(order_id, "不想要了", db)
    assert result["status"] == "待审核"
    assert result["return_id"].startswith("R")
    assert get_returns(order_id, db)[0]["return_id"] == result["return_id"]


@pytest.mark.parametrize("order_id", ["100005", "100007"])
def test_unreturnable_statuses_are_rejected(order_id, db):
    # 已取消 = 没货可退；已下单（未发货）= 走取消订单，不走退货。
    with pytest.raises(ReturnNotAllowed):
        create_return(order_id, "不想要了", db)


def test_duplicate_open_requests_are_blocked(db):
    create_return("100002", "第一次", db)
    with pytest.raises(ReturnNotAllowed) as exc:
        create_return("100002", "第二次", db)
    assert "重复提交" in str(exc.value)
    assert len(get_returns("100002", db)) == 1


def test_list_customer_orders_is_newest_first(db):
    rows = list_customer_orders("C001", db)
    assert [r["order_id"] for r in rows] == ["100002", "100001"]


# ---------------------------------------------------------------------------
# 意图抽取：规则优先
# ---------------------------------------------------------------------------
def test_rules_recognize_the_order_id_and_the_action():
    intent = _rule_extract("订单 100002 到哪了")
    assert intent["order_id"] == "100002"
    assert intent["action"] == "tracking"


def test_a_year_is_not_an_order_id():
    # "2025" 只有 4 位，够不上 6 位；把年份当订单号会让 LLM 那一路显得比规则准。
    assert _rule_extract("2025 年买的那个还有保修吗")["order_id"] == ""


def test_an_explicit_demand_beats_an_incidental_policy_mention():
    # 「我要退货，顺便问下政策」是两条需求，用户点名的那个照办；
    # 被挡掉的只是"纯问政策"。闸门不能闸过界。
    assert _rule_extract("订单 100002 我要退货，顺便问下政策")["action"] == "create_return"
    assert _rule_extract("订单 100002 能不能退，政策是怎么写的")["action"] != "create_return"


def test_rules_answer_without_touching_the_model():
    # 常见问法零模型调用——这是"规则优先"最直接的收益。
    never = ExplodingLLM()
    intent = extract_intent("订单 100002 到哪了", llm=never)
    assert intent["source"] == "rule"
    assert intent["action"] == "tracking"
    assert never.calls == 0


def test_a_dead_model_does_not_take_down_the_rule_path():
    # 认出了动作但缺订单号 → 会去问模型 → 模型挂了 → 必须退回规则结果，
    # 而不是把异常抛给上层。
    intent = extract_intent("我的订单到哪了", llm=ExplodingLLM())
    assert intent["source"] == "rule"
    assert intent["action"] == "tracking"
    assert intent["order_id"] == ""


# ---------------------------------------------------------------------------
# worker 层
# ---------------------------------------------------------------------------
def test_tracking_question_is_answered_from_the_database(wired):
    result = run_order("订单 100002 到哪了")
    assert result["insufficient"] is False
    assert "中通" in result["answer"]
    assert "派送中" in result["answer"]


def test_an_order_that_was_never_shipped_says_so_instead_of_faking_tracking(wired):
    result = run_order("订单 100007 到哪了")
    assert "还没" in result["answer"] or "尚未" in result["answer"]
    assert result["insufficient"] is False


def test_not_found_is_a_business_answer_not_a_failure(wired):
    # ok=False 是"坏了"。订单查不到是"办不了"——它必须是一条正常的答复，
    # 否则监控上分不清"业务规则拦下了多少"和"系统坏了几次"。
    result = run_order("订单 999999 到哪了")
    assert result["insufficient"] is False
    assert "999999" in result["answer"]


def test_a_refused_return_explains_why(wired):
    result = run_order("订单 100005 我要退货")
    assert result["insufficient"] is False
    assert "不能申请退货" in result["answer"]


def test_a_missing_order_id_asks_for_it_instead_of_guessing(wired):
    result = run_order("我这单到哪了")
    assert result["insufficient"] is True
    assert "订单号" in result["answer"]


def test_an_unrecognizable_request_says_so(wired):
    result = run_order("你好")
    assert result["insufficient"] is True
    assert "没有明确的订单操作" in result["answer"]


# ---------------------------------------------------------------------------
# 写操作的边界（这一组是从一次真实事故里长出来的）
# ---------------------------------------------------------------------------
def test_asking_about_the_policy_does_not_file_a_return(wired):
    """
    真实事故：一句「订单 100002 到哪了，顺便问下你们的退货政策是什么」里
    有"退货"两个字，被规则判成 create_return，**真的往库里插了一条退货申请**。
    用户只是问政策，系统却替他办了件事——这是这个 worker 能犯的最严重的错，
    所以它值得两条独立的测试。
    """
    result = run_order("订单 100002 到哪了，顺便问下你们的退货政策是什么")
    assert get_returns("100002", wired) == []          # 库里没多出东西
    assert "物流" in result["answer"]                   # 该答的那半照答
    assert result["insufficient"] is False


def test_a_policy_flavoured_return_question_asks_before_writing(wired):
    result = run_order("订单 100002 能不能退")
    assert result["insufficient"] is True
    assert get_returns("100002", wired) == []
    assert "确认申请退货" in result["answer"]


def test_a_clear_self_demand_still_goes_through(wired):
    # 闸门不能闸过界：明确说"我要退"的必须照办，否则就成了过度保守。
    result = run_order("订单 100002 我要退货")
    assert "待审核" in result["answer"]
    assert len(get_returns("100002", wired)) == 1


def test_a_successful_return_is_only_registered_not_approved(wired):
    result = run_order("订单 100002 我要退货")
    assert "待审核" in result["answer"]
    # 决定权在人手里——模型不该有"通过退货"的能力
    assert get_returns("100002", wired)[0]["status"] == "待审核"
