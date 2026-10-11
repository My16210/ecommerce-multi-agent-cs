"""
退货审核（人那一侧）的测试。用现成的 `db` fixture（灌好种子数据的临时库）。

这个文件里最要紧的两条，都是"**写错了不会报错**"的那种：

1. `test_a_rejected_order_can_apply_again` —— 重复拦截条件写宽一格，被拒绝过
   的订单就再也申请不了，而界面上没有任何一句话解释这件事。
2. `test_an_approved_return_blocks_a_second_application` —— 重复拦截条件写窄
   一格（只比"待审核"），审完就能再申请一次，**结果是两笔退款**。

两条方向相反，所以必须各写一条：只写其中一条的话，另一种写法照样全绿。
"""
import sqlite3

import pytest

from ecom.orchestrator.admin import returns
from ecom.orchestrator.tools.order_tools import (
    APPROVED_RETURN_STATUS,
    OPEN_RETURN_STATUS,
    REJECTED_RETURN_STATUS,
    ReturnNotAllowed,
    create_return,
)


def _orders_that_can_be_returned(db, n: int = 1) -> list:
    """从种子库里挑 n 笔 COMPLETE 的订单。不写死订单号——数据换了它就会烂。

    **要几笔就取几笔**：同一笔订单只能有一份未审的申请（那是业务规则），
    所以一个用例想要两份申请，就得拿两个不同的订单号。
    """
    rows = db.query(
        "SELECT order_id FROM orders WHERE status = 'COMPLETE' ORDER BY order_id LIMIT ?", (n,)
    )
    assert len(rows) == n, f"种子库里 COMPLETE 的订单不足 {n} 笔，这个 fixture 的前提不成立了"
    return [r["order_id"] for r in rows]


def _open_return(db, order_id=None) -> dict:
    order_id = order_id or _orders_that_can_be_returned(db)[0]
    return create_return(order_id, "不想要了", db)


# ——— 列表与统计 ———

def test_a_fresh_application_shows_up_as_pending(db):
    made = _open_return(db)

    pending = returns.list_returns(status=OPEN_RETURN_STATUS, db=db)

    assert [r["return_id"] for r in pending] == [made["return_id"]]
    # 还没审的，审核痕迹那几个字段应当是 None（而不是空串或"未审核"这种假值）
    assert pending[0]["reviewed_at"] is None
    assert pending[0]["reviewer"] is None


def test_stats_counts_every_status_even_when_zero(db):
    """三个状态**都要出现**，哪怕一个都没有。少一个的话，界面上那一栏会消失，
    看起来像"这个状态不存在"，而真实原因是"现在没有"。"""
    _open_return(db)

    assert returns.stats(db) == {OPEN_RETURN_STATUS: 1, APPROVED_RETURN_STATUS: 0,
                                 REJECTED_RETURN_STATUS: 0}


def test_listing_an_unknown_status_is_an_error_not_an_empty_list(db):
    """拼错状态时**抛错**。静默返回空列表会让人以为"没有这种申请"，
    而真实原因是查询条件本身就写错了。"""
    with pytest.raises(ValueError):
        returns.list_returns(status="待審核", db=db)   # 注意是繁体字


def test_listing_by_status_filters(db):
    first, second = _orders_that_can_be_returned(db, 2)
    kept = _open_return(db, first)
    other = _open_return(db, second)
    returns.review_return(other["return_id"], REJECTED_RETURN_STATUS, db=db)

    pending = returns.list_returns(status=OPEN_RETURN_STATUS, db=db)
    everything = returns.list_returns(db=db)

    assert [r["return_id"] for r in pending] == [kept["return_id"]]
    assert len(everything) == 2


# ——— 下结论 ———

def test_approving_changes_the_status_and_leaves_a_trace(db):
    made = _open_return(db)

    result = returns.review_return(
        made["return_id"], APPROVED_RETURN_STATUS, note="客户提供了照片", reviewer="张三", db=db
    )

    assert result["ok"] and not result["already"]
    row = returns.get_return(made["return_id"], db)
    assert row["status"] == APPROVED_RETURN_STATUS
    assert row["reviewer"] == "张三"
    assert row["review_note"] == "客户提供了照片"
    assert row["reviewed_at"]
    assert returns.stats(db)[APPROVED_RETURN_STATUS] == 1


def test_rejecting_also_leaves_a_trace(db):
    made = _open_return(db)

    returns.review_return(made["return_id"], REJECTED_RETURN_STATUS, db=db)

    row = returns.get_return(made["return_id"], db)
    assert row["status"] == REJECTED_RETURN_STATUS
    assert row["decision"] == REJECTED_RETURN_STATUS
    # 没写备注 / 没署名也要能落下（存 None，不是空串）
    assert row["review_note"] is None
    assert row["reviewer"] is None


def test_a_note_can_be_omitted_without_breaking_the_trace(db):
    made = _open_return(db)
    returns.review_return(made["return_id"], APPROVED_RETURN_STATUS, db=db)
    assert returns.get_return(made["return_id"], db)["reviewed_at"]


# ——— 幂等：审过的不能再审 ———

def test_reviewing_twice_does_not_overwrite_the_first_decision(db):
    """人的结论是**证据**，不是可以被后一次点击改写的草稿。"""
    made = _open_return(db)
    returns.review_return(
        made["return_id"], APPROVED_RETURN_STATUS, note="第一次", reviewer="张三", db=db
    )

    again = returns.review_return(
        made["return_id"], REJECTED_RETURN_STATUS, note="第二次", reviewer="李四", db=db
    )

    assert again["ok"] and again["already"] is True
    row = returns.get_return(made["return_id"], db)
    assert row["status"] == APPROVED_RETURN_STATUS, "后一次把前一次的结论改掉了"
    assert row["review_note"] == "第一次", "备注被覆盖了"
    assert row["reviewer"] == "张三"


def test_the_same_decision_twice_is_also_a_no_op(db):
    """点了两次"通过"（网络卡了、界面没刷新）不该报错，也不该多一条痕迹。"""
    made = _open_return(db)
    returns.review_return(made["return_id"], APPROVED_RETURN_STATUS, db=db)

    again = returns.review_return(made["return_id"], APPROVED_RETURN_STATUS, db=db)

    assert again["already"] is True
    assert db.count("return_reviews") == 1


def test_reviewing_a_missing_application_is_a_normal_result(db):
    """"审一个不存在的申请"是业务结果，不是程序故障——所以返回 ok=False
    而不是抛异常（抛了的话 CLI 会打一整段 traceback）。"""
    result = returns.review_return("R不存在", APPROVED_RETURN_STATUS, db=db)

    assert result["ok"] is False
    assert "没有这个退货申请" in result["message"]


def test_a_bogus_decision_is_rejected_loudly(db):
    """decision 只能是两个值之一。**"待审核"也在这里被挡掉**——那是申请的
    初始状态，不是一个决定，用它调等于"把结论改回没审"。"""
    made = _open_return(db)

    with pytest.raises(ValueError):
        returns.review_return(made["return_id"], OPEN_RETURN_STATUS, db=db)
    with pytest.raises(ValueError):
        returns.review_return(made["return_id"], "通过了", db=db)


# ——— 和工具层那条边界的接缝（这个文件里最要紧的两条）———

def test_an_approved_return_blocks_a_second_application(db):
    """**审完之后不能对同一笔订单再申请一次。**

    这是最容易写错的地方：`create_return` 原来只比"待审核"，而审完之后这条
    申请已经不是待审核了——重复拦截会**静默失效**，用户再提交一次就能拿到
    第二笔退款。测试通过之后也是绿的，只有查账时才发现。
    """
    order_id = _orders_that_can_be_returned(db)[0]
    first = create_return(order_id, "第一次", db)
    returns.review_return(first["return_id"], APPROVED_RETURN_STATUS, db=db)

    with pytest.raises(ReturnNotAllowed) as exc:
        create_return(order_id, "再来一次", db)
    assert "已经通过" in str(exc.value)

    assert db.count("return_requests") == 1, "第二笔申请被放进去了"


def test_a_rejected_order_can_apply_again(db):
    """**被拒绝过的订单必须还能再申请。**

    反方向的错：把拦截条件写成"存在任何申请就拦"，用户就会撞上一个
    "你被拒绝过一次，所以永远不能再申请"的系统——而没有任何一句话说明它。
    被拒绝是可以补充材料再来的，这是业务，不是漏洞。
    """
    order_id = _orders_that_can_be_returned(db)[0]
    first = create_return(order_id, "第一次", db)
    returns.review_return(first["return_id"], REJECTED_RETURN_STATUS, db=db)

    second = create_return(order_id, "补充了照片，再申请一次", db)

    assert second["return_id"] != first["return_id"]
    assert second["status"] == OPEN_RETURN_STATUS
    assert db.count("return_requests") == 2


def test_a_pending_application_still_blocks(db):
    """原来就有的行为不能因为这次改动丢掉。"""
    order_id = _orders_that_can_be_returned(db)[0]
    create_return(order_id, "第一次", db)

    with pytest.raises(ReturnNotAllowed):
        create_return(order_id, "再来一次", db)


def test_reviewing_does_not_touch_the_order(db):
    """**批准退货不改订单状态。**

    两件事：① 订单状态是数据源原值，改它就是在编造数据；
    ② "退货通过"和"订单状态变化"本来就是两回事——真上线时订单状态由发货/
    签收那些流程推着走，不是由退货审核来写的。
    """
    order_id = _orders_that_can_be_returned(db)[0]
    before = db.query("SELECT * FROM orders WHERE order_id = ?", (order_id,))[0]
    made = create_return(order_id, "不想要了", db)

    returns.review_return(made["return_id"], APPROVED_RETURN_STATUS, db=db)

    after = db.query("SELECT * FROM orders WHERE order_id = ?", (order_id,))[0]
    assert after == before


def test_a_failed_review_leaves_nothing_behind(db):
    """事务的意义：两条写要么都成、要么都不成。

    这里逼出"痕迹写不进去"的情形——`return_reviews.return_id` 是主键，
    先手工占住它，那条 INSERT 必然违反约束。**要验的是 UPDATE 有没有被回滚**：
    回滚了状态就还是待审核（列表页看得见、还能再审一次）；没回滚就变成一个
    "已通过但没有任何人批过"的申请，而且它再也不能被审了（幂等会拦住）。
    """
    made = _open_return(db)
    db.execute(
        "INSERT INTO return_reviews (return_id, decision, reviewer, note, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (made["return_id"], "占位", None, None, "2026-01-01 00:00:00"),
    )

    with pytest.raises(sqlite3.IntegrityError):
        returns.review_return(made["return_id"], APPROVED_RETURN_STATUS, db=db)

    row = returns.get_return(made["return_id"], db)
    assert row["status"] == OPEN_RETURN_STATUS, "UPDATE 没有被一起回滚"
