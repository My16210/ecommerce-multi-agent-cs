"""订单：业务规则的边界 / 规则优先的意图抽取 / "办不了"不是"坏了"。

【这些订单号是哪来的】

不是编的，是从 `data/orchestrator/seed/orders.csv` 里挑的（那两份 CSV 由
`derive_seed.py` 从 DataCo 公开数据集派生）。挑的是几笔**特征对自己有用**的订单：

    52644  COMPLETE        ￥629.92  目的地 San Salvador  配送 Late delivery
    51592  COMPLETE        ￥319.92  可退货的第二个样本
    55676  PENDING_PAYMENT            不可退货
    75939  PENDING         ￥327.75   不可退货
    75935  CANCELED                   配送被取消 → **没有物流行**（不是数据缺失）

写死具体值仍然是故意的：`assert amount == 629.92` 一眼能看懂。变的是这个值的
来源——它现在来自真实数据，不是我拍脑袋定的。代价是换数据时这些断言要跟着重挑，
这一步不能省：断言和 CSV 脱钩之后，测试会继续绿，而它守的东西已经不是原来那个了。
"""
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

# 特征订单（见模块开头）。改数据时先来这里。
COMPLETE_ORDER = "52644"
COMPLETE_ORDER_2 = "51592"
PENDING_PAYMENT_ORDER = "55676"
PENDING_ORDER = "75939"
CANCELED_ORDER = "75935"  # 没有物流行
DEMO_CUSTOMER = "C9995"   # 演示账号，名下有 6 笔


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
    data = query_order(COMPLETE_ORDER, db)
    assert data["order"]["status"] == "COMPLETE"
    assert data["order"]["amount"] == pytest.approx(629.92)
    assert data["logistics"]["status"] == "Late delivery"


def test_we_do_not_pretend_to_know_the_carrier(db):
    """承运商和运单号在公开数据里根本不存在，留空不是"没灌全"。

    这条锁的是**不许编**：哪天有人为了"看起来完整"给这几列填上顺丰/ZT…，
    这条会红。它比它看起来重要——数据里没有的东西一旦被补上，下游的
    「引用不存在的事实」检查也就失去意义了。
    """
    logistics = query_order(COMPLETE_ORDER, db)["logistics"]
    assert logistics["carrier"] is None
    assert logistics["tracking_no"] is None
    # 连发货地都没有：数据里只有目的地（Order City 是 Destination city）
    assert logistics["from_city"] is None
    assert logistics["to_city"]


def test_an_order_without_logistics_is_not_an_error(db):
    # 配送被取消的订单本来就没有运单。物流为 None 是业务事实，不是数据缺失。
    data = query_order(CANCELED_ORDER, db)
    assert data["logistics"] is None
    assert data["order"]["status"] == "CANCELED"


def test_unknown_order_id_raises_rather_than_returning_empty(db):
    # 返回空 dict 的话，调用方迟早会把它当成"一笔状态为空的订单"用掉。
    with pytest.raises(OrderNotFound):
        query_order("999999", db)


@pytest.mark.parametrize("order_id", [COMPLETE_ORDER, COMPLETE_ORDER_2])
def test_returnable_statuses_accept_a_request(order_id, db):
    result = create_return(order_id, "不想要了", db)
    assert result["status"] == "待审核"
    assert result["return_id"].startswith("R")
    assert get_returns(order_id, db)[0]["return_id"] == result["return_id"]


@pytest.mark.parametrize("order_id", [CANCELED_ORDER, PENDING_ORDER])
def test_unreturnable_statuses_are_rejected(order_id, db):
    # 已取消 = 没货可退；待处理 = 还没履约，用户想做的其实是"取消"不是"退货"。
    with pytest.raises(ReturnNotAllowed):
        create_return(order_id, "不想要了", db)


def test_only_the_completed_status_is_returnable(db):
    """可退货状态只有 COMPLETE 一个，而且是有依据的少放行。

    DataCo 的官方字段字典没写 CLOSED 和 COMPLETE 的区别，所以 CLOSED 不进这个集合——
    没有依据的映射就是编，编错的代价是受理了一笔本不该受理的退货。这条测试把
    "我们只放行有把握的那一个"钉住，免得后来人顺手把 CLOSED 加进去让演示好看点。
    """
    from ecom.orchestrator.tools.order_tools import RETURNABLE_STATUS

    assert RETURNABLE_STATUS == {"COMPLETE"}


def test_duplicate_open_requests_are_blocked(db):
    create_return(COMPLETE_ORDER, "第一次", db)
    with pytest.raises(ReturnNotAllowed) as exc:
        create_return(COMPLETE_ORDER, "第二次", db)
    assert "重复提交" in str(exc.value)
    assert len(get_returns(COMPLETE_ORDER, db)) == 1


def test_list_customer_orders_is_newest_first(db):
    rows = list_customer_orders(DEMO_CUSTOMER, db)
    assert [r["order_id"] for r in rows] == ["55676", "52644", "51592", "37623", "32903", "16700"]
    # 按下单时间倒序，不是按订单号大小：这两者在真实数据里**不一致**
    # （16700 是最早的一笔，却是最小的订单号之一）。旧数据里两者恰好同向，
    # 所以那时候这条断言其实没测出排序用的是哪个字段。
    assert rows[0]["created_at"] > rows[-1]["created_at"]


# ---------------------------------------------------------------------------
# 意图抽取：规则优先
# ---------------------------------------------------------------------------
def test_rules_recognize_the_order_id_and_the_action():
    intent = _rule_extract(f"订单 {COMPLETE_ORDER} 到哪了")
    assert intent["order_id"] == COMPLETE_ORDER
    assert intent["action"] == "tracking"


def test_a_year_is_not_an_order_id():
    # "2025" 只有 4 位，够不上 5 位；把年份当订单号会让 LLM 那一路显得比规则准。
    assert _rule_extract("2025 年买的那个还有保修吗")["order_id"] == ""


def test_a_customer_id_is_not_mistaken_for_an_order_id():
    """客户号 C20755 里那串数字不许被当成订单号。

    客户号和订单号在源数据里都是纯数字，靠一个 C 前缀分开（见 derive_seed.py）。
    这里两件事一起锁：只有客户号时订单号必须为空；两者同现时取的是订单号。
    """
    assert _rule_extract("客户C9995的订单到哪了")["order_id"] == ""
    assert _rule_extract("客户 C9995 的订单 52644 到哪了")["order_id"] == "52644"


def test_order_ids_shorter_than_five_digits_are_not_recognized():
    """一条**已知边界**，写成测试是为了别让它悄悄变成"我们支持所有订单号"。

    DataCo 的订单号是 1~5 位（最小 1，最大 77204，不补零）。正则只能取 5 位起步——
    放宽到 4 位，"2025 年"就会被当成订单号。所以 1~4 位的订单号在对话里认不出来：
    库里查得到（`query_order("5895")` 正常返回），但用户说不出口。
    这条边界写在 README 的"明确不做"里，不是漏做。
    """
    assert _rule_extract("订单 5895 到哪了")["order_id"] == ""
    assert _rule_extract("订单 5895 到哪了")["action"] == "tracking"


def test_an_explicit_demand_beats_an_incidental_policy_mention():
    # 「我要退货，顺便问下政策」是两条需求，用户点名的那个照办；
    # 被挡掉的只是"纯问政策"。闸门不能闸过界。
    q = f"订单 {COMPLETE_ORDER}"
    assert _rule_extract(f"{q} 我要退货，顺便问下政策")["action"] == "create_return"
    assert _rule_extract(f"{q} 能不能退，政策是怎么写的")["action"] != "create_return"


def test_rules_answer_without_touching_the_model():
    # 常见问法零模型调用——这是"规则优先"最直接的收益。
    never = ExplodingLLM()
    intent = extract_intent(f"订单 {COMPLETE_ORDER} 到哪了", llm=never)
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
    result = run_order(f"订单 {COMPLETE_ORDER} 到哪了")
    assert result["insufficient"] is False
    assert "San Salvador" in result["answer"]
    assert "Late delivery" in result["answer"]
    # 没有的信息不许出现在回答里——包括那个难看的 "None"
    assert "None" not in result["answer"]


def test_an_order_that_was_never_shipped_says_so_instead_of_faking_tracking(wired):
    result = run_order(f"订单 {CANCELED_ORDER} 到哪了")
    assert "没有物流信息" in result["answer"]
    assert result["insufficient"] is False


def test_the_status_answer_does_not_claim_an_order_is_unpaid(wired):
    """`paid_at` 为空**不等于**"这笔订单没付钱"。

    旧版这里写着 `if paid_at else '（未支付）'`，那时候空值只有一种含义。
    换成真实数据后它有了第二种含义：**数据集根本没有支付时间这一列**。
    同一句话里的空值有两种意思，就不能用同一个说法兜——这条锁住"宁可不说，
    也不说错"，因为「未支付」是一句会让人去追问客服的假话。
    """
    result = run_order(f"订单 {COMPLETE_ORDER} 怎么样了")
    assert "COMPLETE" in result["answer"]
    assert "未支付" not in result["answer"]
    assert "支付时间" not in result["answer"]


def test_not_found_is_a_business_answer_not_a_failure(wired):
    # ok=False 是"坏了"。订单查不到是"办不了"——它必须是一条正常的答复，
    # 否则监控上分不清"业务规则拦下了多少"和"系统坏了几次"。
    result = run_order("订单 999999 到哪了")
    assert result["insufficient"] is False
    assert "999999" in result["answer"]


def test_a_refused_return_explains_why(wired):
    result = run_order(f"订单 {CANCELED_ORDER} 我要退货")
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
    真实事故：一句「订单 <id> 到哪了，顺便问下你们的退货政策是什么」里有"退货"
    两个字，被规则判成 create_return，**真的往库里插了一条退货申请**。
    用户只是问政策，系统却替他办了件事——这是这个 worker 能犯的最严重的错，
    所以它值得两条独立的测试。
    """
    result = run_order(f"订单 {COMPLETE_ORDER} 到哪了，顺便问下你们的退货政策是什么")
    assert get_returns(COMPLETE_ORDER, wired) == []   # 库里没多出东西
    assert "物流" in result["answer"]                  # 该答的那半照答
    assert result["insufficient"] is False


def test_a_policy_flavoured_return_question_asks_before_writing(wired):
    result = run_order(f"订单 {COMPLETE_ORDER} 能不能退")
    assert result["insufficient"] is True
    assert get_returns(COMPLETE_ORDER, wired) == []
    assert "确认申请退货" in result["answer"]


def test_a_clear_self_demand_still_goes_through(wired):
    # 闸门不能闸过界：明确说"我要退"的必须照办，否则就成了过度保守。
    result = run_order(f"订单 {COMPLETE_ORDER} 我要退货")
    assert "待审核" in result["answer"]
    assert len(get_returns(COMPLETE_ORDER, wired)) == 1


def test_a_successful_return_is_only_registered_not_approved(wired):
    result = run_order(f"订单 {COMPLETE_ORDER} 我要退货")
    assert "待审核" in result["answer"]
    # 决定权在人手里——模型不该有"通过退货"的能力
    assert get_returns(COMPLETE_ORDER, wired)[0]["status"] == "待审核"


# ---------------------------------------------------------------------------
# 跨轮指代：**读可以携带，写不可以**（见 order.py 模块 docstring）
# ---------------------------------------------------------------------------
def _history_with_the_order() -> list:
    """上一轮用户提过订单号。这就是 README「已知限制」里描述的那个场景。"""
    return [
        {"role": "user", "content": f"订单 {COMPLETE_ORDER} 到哪了"},
        {"role": "assistant", "content": f"订单 {COMPLETE_ORDER} 的物流：目的地 San Salvador。"},
    ]


def test_a_followup_inherits_the_order_id_for_a_read():
    # 「它到哪了」里没有订单号，主语在上一轮。规则这条路就该接得上，不必去问模型。
    intent = extract_intent("它到哪了", llm=ExplodingLLM(), chat_history=_history_with_the_order())
    assert intent["action"] == "tracking"
    assert intent["order_id"] == COMPLETE_ORDER
    assert intent["order_id_source"] == "context"
    assert intent["source"] == "rule"


def test_the_carried_order_id_is_tagged_with_where_it_came_from():
    """来源标记是写操作那道闸的全部依据，所以三种情况都要钉住。

    不标来源，`run_order` 就分不开"用户自己说的号"和"从上下文猜的号"。
    """
    from_question = extract_intent(f"订单 {COMPLETE_ORDER} 到哪了", llm=ExplodingLLM())
    from_context = extract_intent("那能退吗", llm=ExplodingLLM(), chat_history=_history_with_the_order())
    neither = extract_intent("你好", llm=ExplodingLLM())
    assert from_question["order_id_source"] == "question"
    assert from_context["order_id_source"] == "context"
    assert neither["order_id_source"] == ""


def test_a_read_followup_is_answered_for_the_carried_order(wired):
    # 携带来的号在**读**这条路上是照常用的——答错了用户当场看得出来。
    result = run_order("它到哪了", chat_history=_history_with_the_order())
    assert result["insufficient"] is False
    assert COMPLETE_ORDER in result["answer"]
    assert "San Salvador" in result["answer"]


def test_a_carried_order_id_is_never_used_to_file_a_return(wired):
    """**这一组里最重要的一条。**

    「那我要退货」携带来 52644，但写操作**不许**用猜来的号：拿它去提交退货申请，
    等于把"一条已知限制"升级成"替用户对错误订单提交退货"，而且**它已经落库了**。
    所以它必须退回"请提供订单号"，等用户显式给出。
    """
    result = run_order("那我要退货", chat_history=_history_with_the_order())
    assert result["insufficient"] is True
    assert "订单号" in result["answer"]
    assert get_returns(COMPLETE_ORDER, wired) == []   # 库里没多出东西


def test_a_policy_followup_names_the_carried_order_but_still_asks_first(wired):
    """README「已知限制」那一幕修好之后的样子。

    「订单 52644 到哪了」→「那能退吗」：第二轮认得出主语是 52644（答案里点名它），
    但**仍然止步在"确认申请退货"**——指代消解修的是"听不听得懂"，不是"该不该动手"。

    （`wired` 把模型换成"调用即抛"，所以走的是"规则认不出动作"那条分支。）
    """
    result = run_order("那能退吗", chat_history=_history_with_the_order())
    assert result["insufficient"] is True
    assert COMPLETE_ORDER in result["answer"]
    assert get_returns(COMPLETE_ORDER, wired) == []


def test_a_model_that_calls_the_followup_a_return_still_names_the_order(wired, monkeypatch):
    """**上一条测不到的那条路，这条专门守。**

    真模型（实测）会把「那能退吗」判成 `create_return`——不是认不出动作，是
    直接认成了写操作。那时候"写操作不许用猜来的号"这道闸如果卡在入口，就会把
    澄清问句里的号也一起清掉，答案退化成"如果要为**某一笔订单**提交退货申请"，
    等于白做了指代消解。

    所以闸卡在**真正落库前那一行**：这条路上答案要点名 52644，而且库仍然不变。
    """
    from tests.orchestrator.conftest import FakeLLM

    monkeypatch.setattr(
        "ecom.orchestrator.workers.order.get_llm",
        lambda *a, **k: FakeLLM('{"action": "create_return", "order_id": "", "reason_text": "问退货"}'),
    )
    result = run_order("那能退吗", chat_history=_history_with_the_order())
    assert COMPLETE_ORDER in result["answer"]
    assert result["insufficient"] is True
    assert get_returns(COMPLETE_ORDER, wired) == []
