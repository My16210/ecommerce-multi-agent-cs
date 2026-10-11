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

from ecom.orchestrator.db.engine import Database, get_db

# 能申请退货的状态。**只有 COMPLETE**——订单已完成，货在客户手里，才谈得上退。
#
# 为什么不把 CLOSED 也放进来：DataCo 的官方字段字典对 `Order Status` 只列了取值，
# 没写 CLOSED 和 COMPLETE 的区别。**没有依据的映射就是编**，而这里编错的代价是
# 系统替用户提交了一笔本不该受理的退货。所以宁可少放行一档——被拦下的用户还能
# 找人工客服，被放行的错误申请是要花钱的。写操作的边界该往保守那边偏。
#
# 为什么 PROCESSING / PENDING 不行：那些订单还没履约完，"退"无从谈起，用户想做的
# 其实是**取消**。取消和退货是两条流程，这里不能替用户换一个操作。
#
# （旧版这行是 `{"已发货", "运输中", "已签收"}`，配合手写的 5 个中文状态。
#  换成真实数据后那 5 个状态没了：DataCo 的订单状态和配送状态是两个维度，
#  "运输中"是配送状态里的说法，不是订单状态。口径收窄是事实，写在这里。）
RETURNABLE_STATUS = {"COMPLETE"}

# 状态的中文注释，只用在给人看的报错和回答里，**不参与任何判断**。
# 判断一律比上面的英文集合——注释一旦参与判断，它就变成了第二个真相来源。
STATUS_GLOSS = {
    "COMPLETE": "订单已完成",
    "CLOSED": "订单已关闭",
    "PROCESSING": "处理中",
    "PENDING": "待处理",
    "PENDING_PAYMENT": "待支付",
    "ON_HOLD": "已挂起",
    "PAYMENT_REVIEW": "支付审核中",
    "SUSPECTED_FRAUD": "疑似欺诈",
    "CANCELED": "已取消",
}


def describe_status(status: str) -> str:
    """`COMPLETE` → `COMPLETE（订单已完成）`。查不到的中文注释就不加括号。"""
    gloss = STATUS_GLOSS.get(status)
    return f"{status}（{gloss}）" if gloss else status

# ——— 退货申请的状态 ———
#
# 这三个值是**中文**，和上面那 9 个英文状态是两个方向，理由写在 schema.sql：
# `orders.status` 的读者是代码和模型，它是数据源原值，改一个就多一个没有依据的
# 映射；退货状态是我们自己造的、读者是人（后台界面、给用户的话术）。
# 一句话：**数据源给的字段保留原值，我们自己造的状态用中文。**
#
# 这里是取值域的**唯一来源**。审核那边（ecom/orchestrator/admin/returns.py）
# 从这里 import，不在那边再抄一份——两份状态清单只会在"加一个状态"时漂。
OPEN_RETURN_STATUS = "待审核"
APPROVED_RETURN_STATUS = "已通过"
REJECTED_RETURN_STATUS = "已拒绝"
RETURN_STATUSES = (OPEN_RETURN_STATUS, APPROVED_RETURN_STATUS, REJECTED_RETURN_STATUS)

# 挡住重复申请的两种状态。**"已通过"必须也在里面，而且它是最要紧的一个**：
# 审完之后这笔申请已经不是"待审核"了，如果只比 待审核，同一笔订单就能被再申请
# 一次——放过去就是**两笔退款**，而且是在"看起来一切正常"的情况下发生的。
#
# **"已拒绝"刻意不在里面**：被拒绝之后应当允许重新申请（补了理由、时间过了、
# 换了客服）。少了这一条，用户会撞上一个"你被拒绝过，所以永远不能再申请"的
# 系统，而界面上没有任何一句话说明这件事。
BLOCKING_RETURN_STATUSES = {OPEN_RETURN_STATUS, APPROVED_RETURN_STATUS}


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
        allowed = "、".join(describe_status(s) for s in sorted(RETURNABLE_STATUS))
        raise ReturnNotAllowed(
            f"订单 {order_id} 当前状态是「{describe_status(order['status'])}」，"
            f"不能申请退货（可申请的状态：{allowed}）"
        )

    for existing in get_returns(order_id, db):
        if existing["status"] in BLOCKING_RETURN_STATUSES:
            if existing["status"] == APPROVED_RETURN_STATUS:
                # 状态已经通过时，错误话术必须换一句：对用户说"请勿重复提交"
                # 会让他以为申请还卡在审核里，然后一直等一个不会来的结果。
                raise ReturnNotAllowed(
                    f"订单 {order_id} 的退货申请（{existing['return_id']}）已经通过，"
                    f"无需重复申请"
                )
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
