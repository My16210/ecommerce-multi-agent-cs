"""
订单工具集：查单、查物流、申请退货、查某客户的订单。

【为什么这些是"工具"而不是"Agent"】
它们没有决策循环：输入确定、输出确定、不需要根据中间结果改变下一步。
硬套一个 agent 壳子只会多一层提示词和一次模型调用，换来的是不确定的输出。
判断标准就一条——**有没有"看结果再决定"的循环**。数据分析有（报错要改代码重跑），
所以它是 agent；查单没有，所以它是工具。

【业务规则写在这里，不写在提示词里】
"哪些状态能退货""重复申请怎么办"是**规则**，规则要能被测试、能被审计。
交给模型去判断，就会出现"同一个订单今天能退明天不能退"。所以这里用代码写死，
模型只负责把用户的话翻译成函数调用（intent），不负责判断能不能办。
"""
import uuid
from datetime import datetime

from cs.db.engine import Database, get_db

# 能申请退货的状态。已下单（还没发货）走的是"取消订单"不是退货；
# 已取消的订单没有货可退。这两条边界都是业务事实，不是漏写。
RETURNABLE_STATUS = {"已发货", "运输中", "已签收"}

# 同一订单不允许同时有两个待审核的退货申请——重复提交大多是用户没看到回执，
# 放过去就会出现两笔退款。
OPEN_RETURN_STATUS = "待审核"


class OrderNotFound(Exception):
    """订单号在库里查不到。和"查到但状态不允许"要分开——用户需要知道是哪种。"""


class ReturnNotAllowed(Exception):
    """订单存在，但按规则不能退货。"""


def _db(db: Database = None) -> Database:
    return db or get_db()


def query_order(order_id: str, db: Database = None) -> dict:
    """
    查一笔订单，连同它的物流（没有则为 None）。

    返回 {"order": {...}, "logistics": {...} | None}。
    """
    db = _db(db)
    rows = db.query("SELECT * FROM orders WHERE order_id = ?", (str(order_id).strip(),))
    if not rows:
        raise OrderNotFound(f"查不到订单号 {order_id}")

    order = rows[0]
    logistics_rows = db.query("SELECT * FROM logistics WHERE order_id = ?", (order["order_id"],))
    return {"order": order, "logistics": logistics_rows[0] if logistics_rows else None}


def get_order_status(order_id: str, db: Database = None) -> dict:
    """只取订单状态与金额那一层，不拉物流明细。"""
    return query_order(order_id, db)["order"]


def list_customer_orders(customer_id: str, db: Database = None) -> list:
    """某客户名下的全部订单，按下单时间倒序（最近的在前）。"""
    db = _db(db)
    return db.query(
        "SELECT * FROM orders WHERE customer_id = ? ORDER BY created_at DESC",
        (str(customer_id).strip(),),
    )


def get_returns(order_id: str, db: Database = None) -> list:
    """某订单已有的退货申请。"""
    db = _db(db)
    return db.query(
        "SELECT * FROM return_requests WHERE order_id = ? ORDER BY created_at DESC",
        (str(order_id).strip(),),
    )


def create_return(order_id: str, reason: str, db: Database = None) -> dict:
    """
    提交退货申请。**只登记，不通过**——最终决定权在人（审核队列），
    这一点和 rag 项目里"模型判定不进向量库、要人审"是同一条原则。

    返回 {"return_id", "order_id", "status", "created_at", "note"}。
    三种情况抛异常：订单不存在 / 状态不可退 / 已有待审核申请。
    抛异常而不是返回 {"ok": False}，是为了让调用方**没法忽略失败**——
    返回一个"看起来像结果"的东西，最后一定会有人把它当成功用了。
    """
    db = _db(db)
    order = get_order_status(order_id, db)

    if order["status"] not in RETURNABLE_STATUS:
        raise ReturnNotAllowed(
            f"订单 {order_id} 当前状态是「{order['status']}」，不能申请退货"
            f"（可申请的状态：{'、'.join(sorted(RETURNABLE_STATUS))}）"
        )

    for existing in get_returns(order_id, db):
        if existing["status"] == OPEN_RETURN_STATUS:
            raise ReturnNotAllowed(
                f"订单 {order_id} 已经有一笔在审核中的退货申请"
                f"（{existing['return_id']}），请勿重复提交"
            )

    return_id = "R" + uuid.uuid4().hex[:10].upper()
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    db.execute(
        "INSERT INTO return_requests (return_id, order_id, reason, status, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (return_id, order["order_id"], reason.strip()[:255], OPEN_RETURN_STATUS, created_at),
    )
    return {
        "return_id": return_id,
        "order_id": order["order_id"],
        "status": OPEN_RETURN_STATUS,
        "created_at": created_at,
        "note": "申请已提交，等待人工审核。审核结果会另行通知。",
    }
