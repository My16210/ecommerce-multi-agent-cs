"""
退货审核：**人**对一个申请下结论，并把结论留痕。

调用方有两个，共用下面这些函数（不各写一份）：
    ecom/orchestrator/cli/returns.py   人敲的命令行
    admin.py（仓库根）                  后台界面

【"批准"到底做了什么——这条必须写在最显眼的地方】
只**登记人的决定**。具体是三件事：
    1. `return_requests.status` 从 待审核 → 已通过 / 已拒绝；
    2. 往 `return_reviews` 插一行：谁、什么时候、什么决定、什么备注；
    3. 两步在**同一个事务**里（见 `Database.transaction`）。
**不做**的两件事，写死在这里：
    · **不执行退款。** 没有支付网关可调，假装退款比不做更坏。
    · **不改 `orders.status`。** 退货通过不等于订单状态变化，而且那个字段是
      DataCo 的原值，改它就是在编造数据。

【幂等：审过的再审一次，不覆盖】
和 `ecom/knowledge/review/queue.py` 那边「人工定了就不再回退」是同一条语义——
人的结论是**证据**，不是可以被后一次点击改写的草稿。第二次调用返回
`already=True`，note / reviewer 一个字段都不动。要改结论得先删掉那行痕迹
（那是另一件事，这里不提供入口）。

【为什么 `list_returns` 要 join `return_reviews`】
列表页要显示"谁审的、什么时候、备注写了什么"。分成两次查也行，但那样列表和
详情会**用两条不同的代码路径读同一件事**——迟早一条改了另一条没改。
"""
from datetime import datetime

from ecom.auth import settings as auth_settings
from ecom.auth import store as auth_store
from ecom.orchestrator.db.engine import Database, get_db
from ecom.orchestrator.tools.order_tools import (
    APPROVED_RETURN_STATUS,
    OPEN_RETURN_STATUS,
    REJECTED_RETURN_STATUS,
    RETURN_STATUSES,
)

# 人能下的两个结论。**"待审核"不在里面**：那是申请的初始状态，不是一个决定。
# 再审一次 = 把人已经下过的结论改掉，而按上面的幂等规则，这件事做不了。
DECISIONS = (APPROVED_RETURN_STATUS, REJECTED_RETURN_STATUS)

_MAX_NOTE = 255
_MAX_REVIEWER = 64


def _db(db: Database = None) -> Database:
    return db or get_db()


def list_returns(status: str = None, limit: int = 50, db: Database = None) -> list:
    """
    退货申请列表，最近提交的在前，**带上审核痕迹**（没审过的那些字段是 None）。

    `status` 传 None 表示不过滤。传一个不在 `RETURN_STATUSES` 里的值**抛错**
    ——静默返回空列表会让人以为"这个状态没有申请"，而真实原因是拼错了。
    """
    if status is not None and status not in RETURN_STATUSES:
        raise ValueError(f"未知的退货状态：{status!r}，只有 {RETURN_STATUSES}")

    sql = (
        "SELECT r.return_id, r.order_id, r.reason, r.status, r.created_at, "
        "       v.decision, v.reviewer, v.note AS review_note, v.created_at AS reviewed_at "
        "FROM return_requests r "
        "LEFT JOIN return_reviews v ON v.return_id = r.return_id "
    )
    params: tuple = ()
    if status is not None:
        sql += "WHERE r.status = ? "
        params = (status,)
    sql += "ORDER BY r.created_at DESC, r.return_id DESC LIMIT ?"
    return _db(db).query(sql, params + (int(limit),))


def get_return(return_id: str, db: Database = None) -> dict:
    """单个申请（含审核痕迹）。查不到返回 None——调用方要判空。"""
    rows = _db(db).query(
        "SELECT r.return_id, r.order_id, r.reason, r.status, r.created_at, "
        "       v.decision, v.reviewer, v.note AS review_note, v.created_at AS reviewed_at "
        "FROM return_requests r "
        "LEFT JOIN return_reviews v ON v.return_id = r.return_id "
        "WHERE r.return_id = ?",
        (str(return_id).strip(),),
    )
    return rows[0] if rows else None


def stats(db: Database = None) -> dict:
    """各状态各有多少条。后台列表页的顶栏用它。"""
    counts = dict.fromkeys(RETURN_STATUSES, 0)
    for row in _db(db).query(
        "SELECT status, COUNT(*) AS n FROM return_requests GROUP BY status"
    ):
        counts[row["status"]] = int(row["n"])
    return counts


def review_return(
    return_id: str,
    decision: str,
    note: str = None,
    reviewer: str = None,
    db: Database = None,
) -> dict:
    """
    对一个退货申请下结论。

    **这是一个"编排函数"，不是 CLI 命令**：它不打印任何东西，只返回结果字典，
    好让命令行和后台界面共用同一份逻辑（照 `knowledge/cli/review.approve_item`
    的形状）。返回：

        {"ok": bool, "already": bool, "return_id", "status", "message"}

    `ok=False` 的两种情形（都没抛异常，因为"审一个不存在的申请"是个正常的
    业务结果，不是程序故障）：申请号查不到 / decision 不合法。
    """
    if decision not in DECISIONS:
        raise ValueError(f"decision 只能是 {DECISIONS}，收到 {decision!r}")

    db = _db(db)
    current = get_return(return_id, db)
    if current is None:
        return {
            "ok": False,
            "already": False,
            "return_id": return_id,
            "status": None,
            "message": f"没有这个退货申请：{return_id}",
        }

    if current["status"] != OPEN_RETURN_STATUS:
        # **不覆盖。** 人的结论是证据，不是草稿：第二次点击返回 already=True，
        # 一个字段都不动（包括 note）。想改结论得先删掉那行痕迹——那是另一件事。
        return {
            "ok": True,
            "already": True,
            "return_id": current["return_id"],
            "status": current["status"],
            "message": (
                f"这个申请已经审过了（当前：{current['status']}），"
                f"本次没有改动。审核备注保持原样。"
            ),
        }

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    class _LostRace(Exception):
        """状态在"读"和"写"之间被别人改了。用它把整笔事务回滚掉。"""

    try:
        # 两条写**必须原子**：只改状态不留痕迹，日后没人说得清是谁批的；
        # 只留痕迹不改状态，列表页会一直把它显示成待审核。
        counts = db.transaction([
            (
                # 第三个条件是**第二道闸**，堵的是 check-then-act 的竞态：
                # 上面 `get_return` 读到"待审核"之后、这条 UPDATE 执行之前，
                # 另一个审核人（后台界面和 CLI 完全可能同时在跑）先把结论下了。
                # 少了它，后手会把人已经下的结论覆盖掉——而按上面的幂等规则，
                # 那本来是不允许的，只是因为读了旧值才绕了过去。
                "UPDATE return_requests SET status = ? WHERE return_id = ? AND status = ?",
                (decision, current["return_id"], OPEN_RETURN_STATUS),
            ),
            (
                "INSERT INTO return_reviews "
                "(return_id, decision, reviewer, note, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    current["return_id"],
                    decision,
                    (reviewer or "").strip()[:_MAX_REVIEWER] or None,
                    (note or "").strip()[:_MAX_NOTE] or None,
                    now,
                ),
            ),
        ])
        if counts[0] == 0:
            # 更新一行都没命中 = 状态已经不是待审核了。**必须抛出去**，
            # 而不是在这里 return：return 会正常提交，那条 review 痕迹就落下了
            # ——变成"有痕迹、但状态没被这次改动"，正是这个事务要防的东西。
            raise _LostRace
    except _LostRace:
        latest = get_return(return_id, db) or {}
        return {
            "ok": True,
            "already": True,
            "return_id": return_id,
            "status": latest.get("status"),
            "message": f"就在你确认的这一刻，这个申请已经被别人审成「{latest.get('status')}」了。"
                       f"本次没有改动，请刷新看最新的结论。",
        }

    if auth_settings.AUTH_ENABLED:
        # 审计只记录"谁下了什么结论"，不重复 reviewer 字段——那个在
        # return_reviews 里已经是证据。开关没开时不写 auth.db：本地演示
        # 和测试不该凭空多一个凭证库。
        auth_store.audit(
            reviewer or "unknown",
            "return.review",
            current["return_id"],
            decision,
        )

    return {
        "ok": True,
        "already": False,
        "return_id": current["return_id"],
        "status": decision,
        "message": f"退货申请 {current['return_id']} 已{decision}。"
                   f"（只登记决定，不执行退款、不改订单状态）",
    }
