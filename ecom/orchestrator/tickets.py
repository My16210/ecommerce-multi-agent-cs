"""
人工工单：转人工之后真正有人接手的那一段。

【这一层补的是什么】
以前 `decision="human"` 只是换了一句固定话术，用户说完「转人工」之后什么也
不会发生。现在转人工会落一张工单：谁、什么时候、因为什么、排在第几、谁认领、
谁回复、什么时候关掉。**没有工单的"转人工"是话术，不是功能。**

【状态机只有三档，而且只往前走】
    open  ──认领──►  claimed  ──关闭──►  closed

不设 "reopened"：关掉之后再求人工会新建一张工单。重开一张旧单会让"这张单
到底被处理了几次"变成一个没法从状态里读出来的问题。

【一个会话同时最多一张未关闭的工单】
由部分唯一索引 `idx_tickets_live_session` 保证，不是靠应用层先查后插——
两个请求同时转人工时，先查后插两边都会读不到、然后各插一条。

【人工回复直接写进会话的 messages 表（role='agent'）】
不另开 ticket_messages，三个理由：
1. 用户重新打开会话时走的是同一个 `transcript()`，人工说过的话必须和机器
   说过的话按时间排在一起；两张表就要在读取侧做一次归并，归并的顺序和删除
   的完整性都多一个出错点。
2. 删会话时 `_delete_rows` 一次就删干净，不会留下"正文删了、工单里还有"。
3. `history()` 把 agent 映射成 assistant 喂回模型，人工说过的话自然成为
   下一轮机器答复的上下文——这正是接手的人希望发生的。

【工单和身份层绑定】
工单必须有 `owner_id`（principal id）：没有账号就没有"谁在排队"这件事，
`open_ticket` 直接返回 None，API 层据此显示"当前环境未开通人工客服"。
"""
import secrets
import sqlite3
from datetime import datetime

from ecom.orchestrator import session

STATUSES = ("open", "claimed", "closed")
MAX_REPLY_CHARS = 2000


class TicketError(Exception):
    """工单操作不合法（状态不对、被别人认领、内容为空……）。API 翻成 409/422。"""


class TicketNotFound(TicketError):
    """工单不存在。API 翻成 404，不暴露"存在但没权限"。"""


class TicketConflict(TicketError):
    """状态冲突（已关闭、已被别人认领）。API 翻成 409。"""


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _new_ticket_id() -> str:
    """人能念出来的工单号：T + 日期 + 6 位。用户要把它报给客服。"""
    return "T" + datetime.now().strftime("%y%m%d") + "-" + secrets.token_hex(3).upper()


def _row(row) -> dict:
    return dict(row) if row is not None else None


def _position(conn, row) -> int:
    """
    排队位置：**在它前面还有几张没被认领的单**。已经有人接手的单是 0。

    用自增 `id` 排序而不是 `created_at`：后者只精确到秒，同一秒里转进来的
    两张单会拿到同一个位置，而"你前面还有几个人"必须是一个确定的数。
    """
    if row["status"] != "open":
        return 0
    return conn.execute(
        "SELECT COUNT(*) AS n FROM tickets WHERE status = 'open' AND id < ?",
        (row["id"],),
    ).fetchone()["n"]


def _with_position(conn, row) -> dict:
    data = _row(row)
    data["queue_position"] = _position(conn, row)
    return data


def open_ticket(session_id: str, owner_id: str, reason: str = "") -> dict | None:
    """
    给一个会话开人工工单；同一会话已有未关闭的工单就原样返回它。

    幂等是刻意的：用户连说两遍「转人工」、或者机器连续两轮都给出 human，
    排队里都不该多出第二张单。返回 **None** 表示"这件事在这个环境里做不了"
    （没开身份层 / 没传 owner / 会话不属于他），由调用方决定怎么告诉用户。
    """
    sid = str(session_id or "").strip()
    owner = str(owner_id or "").strip()
    if not sid or not owner:
        return None

    meta = session.get_session(sid)
    if meta is None or (meta.get("owner_id") or "") != owner:
        return None

    with session.transaction() as conn:
        existing = conn.execute(
            "SELECT * FROM tickets WHERE session_id = ? AND status != 'closed' "
            "ORDER BY id DESC LIMIT 1",
            (sid,),
        ).fetchone()
        if existing is not None:
            return _with_position(conn, existing)

        now = _now()
        ticket_id = _new_ticket_id()
        try:
            conn.execute(
                "INSERT INTO tickets "
                "(ticket_id, session_id, owner_id, status, reason, created_at, updated_at) "
                "VALUES (?, ?, ?, 'open', ?, ?, ?)",
                (ticket_id, sid, owner, str(reason or "")[:500], now, now),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            # 另一个请求同时开了单（唯一索引挡住了）。重读，把先到的那张给它。
            conn.rollback()
            existing = conn.execute(
                "SELECT * FROM tickets WHERE session_id = ? AND status != 'closed' "
                "ORDER BY id DESC LIMIT 1",
                (sid,),
            ).fetchone()
            if existing is None:  # 理论上不可达；真出现就按"开不了"处理
                return None
            return _with_position(conn, existing)

        row = conn.execute(
            "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)
        ).fetchone()
        return _with_position(conn, row)


def ticket_for_session(session_id: str) -> dict | None:
    """
    这个会话最近的一张工单（**包括已关闭的**）。

    已关闭也要返回：用户重新打开会话时，界面要能显示"上一次人工服务已结束"，
    而不是像什么都没发生过。没有工单返回 None。
    """
    sid = str(session_id or "").strip()
    if not sid:
        return None
    with session.transaction() as conn:
        row = conn.execute(
            "SELECT * FROM tickets WHERE session_id = ? ORDER BY id DESC LIMIT 1",
            (sid,),
        ).fetchone()
        return _with_position(conn, row) if row is not None else None


def get_ticket(ticket_id: str) -> dict | None:
    tid = str(ticket_id or "").strip()
    if not tid:
        return None
    with session.transaction() as conn:
        row = conn.execute(
            "SELECT * FROM tickets WHERE ticket_id = ?", (tid,)
        ).fetchone()
        return _with_position(conn, row) if row is not None else None


def list_tickets(
    status: str = None,
    limit: int = 50,
    viewer_id: str = None,
    is_admin: bool = False,
) -> list:
    """
    队列。`status=None` / `"all"` 表示全部；否则只认 open/claimed/closed。

    排队看的是**最老的在前**（先来先服务）；已关闭的看最近关闭的在前。

    `viewer_id` 不为空时按客服收口：**排队中的所有人可见**（要先看前因才决定
    认不认领），**进行中 / 已结束的只有自己认领或关闭的可见**。`is_admin=True`
    不受限制。`viewer_id=None` 表示不加范围（CLI / 单元测试用），API 永远传
    principal id，不会走到这里的分支。
    """
    filter_status = str(status or "").strip().lower()
    if filter_status in ("", "all"):
        filter_status = None
    elif filter_status not in STATUSES:
        raise TicketError(f"status 只能是 {STATUSES} 或 all")
    limit = max(1, min(int(limit or 50), 200))

    where_parts: list[str] = []
    args: list = []
    if filter_status:
        where_parts.append("t.status = ?")
        args.append(filter_status)
    if viewer_id is not None and not is_admin:
        # 排队中的公开；接手过的（claimed_by）和关过的（closed_by）才算自己的。
        where_parts.append(
            "(t.status = 'open' OR t.claimed_by = ? OR t.closed_by = ?)"
        )
        args.extend([str(viewer_id), str(viewer_id)])
    where = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
    args.append(limit)

    order = "t.id DESC" if filter_status == "closed" else "t.id ASC"
    with session.transaction() as conn:
        rows = conn.execute(
            f"""
            SELECT t.*,
                   COALESCE((
                       SELECT content FROM messages m
                       WHERE m.session_id = t.session_id AND m.role = 'user'
                       ORDER BY m.id ASC LIMIT 1
                   ), '') AS title
            FROM tickets t
            {where}
            ORDER BY {order}
            LIMIT ?
            """,
            tuple(args),
        ).fetchall()
        return [_with_position(conn, row) for row in rows]


def can_view(ticket: dict, viewer_id: str, is_admin: bool = False) -> bool:
    """
    这张工单的上下文（transcript）能不能给这个人看。

    · admin：全部可见；
    · 排队中的：所有客服可见——不看前因没法判断该不该接；
    · 进行中 / 已结束：只有认领人（或执行关闭的人）可见。

    列表已经按同一口径过滤；这个函数守的是**直接拿 ticket_id 来查详情**的路径，
    两条路必须一致，否则列表看不见、详情却能读到。
    """
    if is_admin:
        return True
    if ticket is None or not viewer_id:
        return False
    if ticket.get("status") == "open":
        return True
    return viewer_id in {ticket.get("claimed_by"), ticket.get("closed_by")}


def claim_ticket(ticket_id: str, agent_id: str, agent_name: str) -> dict:
    """
    认领一张工单。**只有 open 的单能被认领**；重复认领自己那张是幂等的。

    被别人认领的单直接冲突，不提供"抢过来"——那会让"谁在服务这个用户"
    变成要看时间线的判断题。
    """
    tid = str(ticket_id or "").strip()
    aid = str(agent_id or "").strip()
    if not tid or not aid:
        raise TicketNotFound(f"没有这个工单：{ticket_id}")

    with session.transaction() as conn:
        row = conn.execute(
            "SELECT * FROM tickets WHERE ticket_id = ?", (tid,)
        ).fetchone()
        if row is None:
            raise TicketNotFound(f"没有这个工单：{tid}")
        if row["status"] == "closed":
            raise TicketConflict("工单已关闭")
        if row["status"] == "claimed":
            if row["claimed_by"] == aid:
                return _with_position(conn, row)
            raise TicketConflict(f"工单已被 {row['claimed_by_name'] or '其他客服'} 认领")

        now = _now()
        conn.execute(
            "UPDATE tickets SET status = 'claimed', claimed_by = ?, claimed_by_name = ?, "
            "claimed_at = ?, updated_at = ? WHERE ticket_id = ? AND status = 'open'",
            (aid, str(agent_name or "")[:64], now, now, tid),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM tickets WHERE ticket_id = ?", (tid,)
        ).fetchone()
        return _with_position(conn, row)


def reply_ticket(ticket_id: str, agent_id: str, agent_name: str, text: str) -> dict:
    """
    在工单里回一句。**先认领才能回**：没认领就回等于两个客服同时对着一个人说话。

    回复写进会话的 messages 表（role='agent'）。返回 `{"message_id": n}`。
    """
    tid = str(ticket_id or "").strip()
    aid = str(agent_id or "").strip()
    content = str(text or "").strip()
    if not content:
        raise TicketError("回复内容不能为空")
    if len(content) > MAX_REPLY_CHARS:
        raise TicketError(f"回复太长（上限 {MAX_REPLY_CHARS} 字）")

    with session.transaction() as conn:
        row = conn.execute(
            "SELECT * FROM tickets WHERE ticket_id = ?", (tid,)
        ).fetchone()
        if row is None:
            raise TicketNotFound(f"没有这个工单：{tid}")
        if row["status"] == "closed":
            raise TicketConflict("工单已关闭")
        if row["status"] != "claimed" or row["claimed_by"] != aid:
            raise TicketConflict("请先认领这张工单")

    # 写会话消息走 session 自己的锁/连接，和上面的检查不是同一次事务。
    # 中间被别人关掉工单的窗口极小，而且后果只是"关单后又多了一句回复"，
    # 不值得为此把两张表的写入挤进一个跨模块事务里。
    #
    # `ensure_session` 在会话不存在时**会新建一个**：不先确认的话，删掉会话后
    # 又有人回复，会把一个空会话原地复活，里面只有客服这一句。
    if session.get_session(row["session_id"]) is None:
        raise TicketConflict("会话已被删除，工单无法回复")
    message_id = session.append_agent_message(
        row["session_id"], content, author=agent_name, owner_id=row["owner_id"]
    )
    with session.transaction() as conn:
        conn.execute(
            "UPDATE tickets SET updated_at = ? WHERE ticket_id = ?",
            (_now(), tid),
        )
        conn.commit()
    return {"message_id": message_id}


def close_ticket(ticket_id: str, agent_id: str, allow_override: bool = False) -> dict:
    """
    关掉工单。默认只有认领人能关（重复关是幂等的）。

    `allow_override=True` 只给 admin 的恢复路径用：认领人离职 / 账号被吊销 /
    忘了关，工单不能因此永远占着这个会话。旁路必须留审计（API 层负责写），
    不能让"谁对这张单负责"变成一个查不出来的问题。
    """
    tid = str(ticket_id or "").strip()
    aid = str(agent_id or "").strip()
    with session.transaction() as conn:
        row = conn.execute(
            "SELECT * FROM tickets WHERE ticket_id = ?", (tid,)
        ).fetchone()
        if row is None:
            raise TicketNotFound(f"没有这个工单：{tid}")
        if row["status"] == "closed":
            return _with_position(conn, row)
        if not allow_override and row["claimed_by"] != aid:
            raise TicketConflict("只有认领人可以关闭这张工单")
        now = _now()
        conn.execute(
            "UPDATE tickets SET status = 'closed', closed_at = ?, closed_by = ?, "
            "updated_at = ? WHERE ticket_id = ?",
            (now, aid, now, tid),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM tickets WHERE ticket_id = ?", (tid,)
        ).fetchone()
        return _with_position(conn, row)


def transfer_ticket(ticket_id: str, assignee: dict = None) -> dict:
    """
    把工单转给另一个客服；`assignee=None` 表示**放回队列**（清空认领）。

    只给 admin 用，API 层校验角色。转派会把 `claimed_at` 重置为现在——
    "这张单从什么时候开始归我"必须能从数据里读出来；放回队列则把三个
    claimed_* 字段一起清空，排队位置按 `id` 重新算。
    """
    tid = str(ticket_id or "").strip()
    with session.transaction() as conn:
        row = conn.execute(
            "SELECT * FROM tickets WHERE ticket_id = ?", (tid,)
        ).fetchone()
        if row is None:
            raise TicketNotFound(f"没有这个工单：{tid}")
        if row["status"] == "closed":
            raise TicketConflict("工单已关闭，不能转派")

        now = _now()
        if assignee is None:
            conn.execute(
                "UPDATE tickets SET status = 'open', claimed_by = NULL, "
                "claimed_by_name = NULL, claimed_at = NULL, updated_at = ? "
                "WHERE ticket_id = ?",
                (now, tid),
            )
        else:
            conn.execute(
                "UPDATE tickets SET status = 'claimed', claimed_by = ?, "
                "claimed_by_name = ?, claimed_at = ?, updated_at = ? "
                "WHERE ticket_id = ?",
                (
                    str(assignee.get("id") or ""),
                    str(assignee.get("name") or "")[:64],
                    now,
                    now,
                    tid,
                ),
            )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM tickets WHERE ticket_id = ?", (tid,)
        ).fetchone()
        return _with_position(conn, row)
