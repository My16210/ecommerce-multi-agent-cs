"""人工工单的存储层测试：排队、认领、回复、关闭、删除。

不联网、不调模型。这里钉住的都是**不报错但会算错**的语义：
  · 同一会话重复转人工只有一张活单；
  · 排队位置按先来后到，不按秒级时间戳；
  · 人工回复真的写进会话，并且 `history()` 能把它当 assistant 喂回模型；
  · 删会话会连工单一起删（隐私上是同一种数据）。
"""
import pytest

from ecom.orchestrator import session, tickets


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "SESSION_DB_PATH", str(tmp_path / "sessions.db"))
    session.reset_connection()
    yield session
    session.reset_connection()


def _owned_session(store, owner="p_demo", first="订单 52644 到哪了"):
    sid = store.ensure_session("", owner_id=owner)
    store.append_turn(sid, first, "答复", owner_id=owner)
    return sid


# ——— 开单 ———

def test_open_ticket_is_idempotent_for_the_same_session(store):
    sid = _owned_session(store)
    first = tickets.open_ticket(sid, "p_demo", "用户要求")
    second = tickets.open_ticket(sid, "p_demo", "又说了一遍")

    assert first["ticket_id"] == second["ticket_id"]
    assert len(tickets.list_tickets(status="open")) == 1


def test_open_ticket_refuses_someone_elses_session(store):
    sid = _owned_session(store, owner="p_demo")
    assert tickets.open_ticket(sid, "p_other", "") is None
    assert tickets.open_ticket("nope", "p_demo", "") is None


def test_queue_position_counts_the_older_open_tickets(store):
    """先来的排前面。用自增 id 而不是秒级时间戳——同一秒开的两张单也要有先后。"""
    s1 = _owned_session(store, first="第一")
    s2 = _owned_session(store, first="第二")
    s3 = _owned_session(store, first="第三")

    t1 = tickets.open_ticket(s1, "p_demo", "")
    t2 = tickets.open_ticket(s2, "p_demo", "")
    t3 = tickets.open_ticket(s3, "p_demo", "")

    assert (t1["queue_position"], t2["queue_position"], t3["queue_position"]) == (0, 1, 2)


def test_claiming_removes_the_ticket_from_the_queue(store):
    s1 = _owned_session(store, first="第一")
    s2 = _owned_session(store, first="第二")
    tickets.open_ticket(s1, "p_demo", "")
    second = tickets.open_ticket(s2, "p_demo", "")

    claimed = tickets.claim_ticket(second["ticket_id"], "p_agent", "小李")

    assert claimed["status"] == "claimed"
    assert claimed["claimed_by_name"] == "小李"
    assert claimed["queue_position"] == 0
    # 前一张还等着，所以它的位置不变；被认领的那张不再占排队位。
    assert len(tickets.list_tickets(status="open")) == 1


def test_claiming_someone_elses_ticket_conflicts(store):
    sid = _owned_session(store)
    ticket = tickets.open_ticket(sid, "p_demo", "")
    tickets.claim_ticket(ticket["ticket_id"], "p_agent", "小李")

    with pytest.raises(tickets.TicketConflict):
        tickets.claim_ticket(ticket["ticket_id"], "p_agent2", "小王")


def test_reply_requires_claiming_first(store):
    sid = _owned_session(store)
    ticket = tickets.open_ticket(sid, "p_demo", "")

    with pytest.raises(tickets.TicketConflict):
        tickets.reply_ticket(ticket["ticket_id"], "p_agent", "小李", "你好")


def test_reply_is_written_into_the_session_and_fed_back_to_the_model(store):
    """人工回复必须和机器答复落在同一条时间线上。

    `history()` 若原样返回 role='agent'，模型会收到一个它不认识的 role——
    换成严格的 provider 就会报错。所以这里同时钉住"写进去"和"映射成 assistant"。
    """
    sid = _owned_session(store)
    ticket = tickets.open_ticket(sid, "p_demo", "")
    tickets.claim_ticket(ticket["ticket_id"], "p_agent", "小李")

    tickets.reply_ticket(ticket["ticket_id"], "p_agent", "小李", "我是小李，正在帮你查。")

    rows = store.transcript(sid)
    assert rows[-1]["role"] == "agent"
    assert rows[-1]["author"] == "小李"
    assert rows[-1]["content"] == "我是小李，正在帮你查。"

    history = store.history(sid)
    assert history[-1] == {"role": "assistant", "content": "我是小李，正在帮你查。"}


def test_closing_is_only_for_the_claimant_and_is_idempotent(store):
    sid = _owned_session(store)
    ticket = tickets.open_ticket(sid, "p_demo", "")
    tickets.claim_ticket(ticket["ticket_id"], "p_agent", "小李")

    with pytest.raises(tickets.TicketConflict):
        tickets.close_ticket(ticket["ticket_id"], "p_agent2")

    closed = tickets.close_ticket(ticket["ticket_id"], "p_agent")
    assert closed["status"] == "closed"
    assert tickets.close_ticket(ticket["ticket_id"], "p_agent")["status"] == "closed"


def test_asking_for_a_human_after_closing_opens_a_new_ticket(store):
    sid = _owned_session(store)
    first = tickets.open_ticket(sid, "p_demo", "")
    tickets.claim_ticket(first["ticket_id"], "p_agent", "小李")
    tickets.close_ticket(first["ticket_id"], "p_agent")

    second = tickets.open_ticket(sid, "p_demo", "再次要求")

    assert second["ticket_id"] != first["ticket_id"]
    assert second["status"] == "open"


def test_ticket_for_session_returns_the_closed_one_too(store):
    """关掉之后不能"看起来像没转接过"：界面要能显示上一次服务已结束。"""
    sid = _owned_session(store)
    ticket = tickets.open_ticket(sid, "p_demo", "")
    tickets.claim_ticket(ticket["ticket_id"], "p_agent", "小李")
    tickets.close_ticket(ticket["ticket_id"], "p_agent")

    latest = tickets.ticket_for_session(sid)
    assert latest["ticket_id"] == ticket["ticket_id"]
    assert latest["status"] == "closed"


# ——— 删除与保留期 ———

def test_deleting_a_session_deletes_its_ticket(store):
    sid = _owned_session(store)
    tickets.open_ticket(sid, "p_demo", "")

    store.delete_session(sid)

    assert tickets.ticket_for_session(sid) is None
    assert tickets.list_tickets(status="all") == []


def test_agent_sees_open_queue_but_not_other_peoples_claimed_or_closed(store):
    sid_claimed = _owned_session(store, first="别人接手的")
    sid_closed = _owned_session(store, first="别人关掉的")
    sid_open = _owned_session(store, first="还在排队")

    claimed = tickets.open_ticket(sid_claimed, "p_demo", "")
    closed = tickets.open_ticket(sid_closed, "p_demo", "")
    opened = tickets.open_ticket(sid_open, "p_demo", "")

    tickets.claim_ticket(claimed["ticket_id"], "p_a", "小 A")
    tickets.claim_ticket(closed["ticket_id"], "p_a", "小 A")
    tickets.close_ticket(closed["ticket_id"], "p_a")

    # 排队中的所有人都要看见（不看前因没法决定接不接）。
    assert [t["ticket_id"] for t in tickets.list_tickets(
        status="open", viewer_id="p_b"
    )] == [opened["ticket_id"]]
    # 别人接手的、别人关掉的，列表里不出现。
    assert tickets.list_tickets(status="claimed", viewer_id="p_b") == []
    assert tickets.list_tickets(status="closed", viewer_id="p_b") == []
    # 自己的能看到，admin 不受限制。
    assert [t["ticket_id"] for t in tickets.list_tickets(
        status="claimed", viewer_id="p_a"
    )] == [claimed["ticket_id"]]
    assert len(tickets.list_tickets(status="all", viewer_id="p_a", is_admin=True)) == 3

    # 详情走的是另一条路，必须与列表同口径，否则绕过列表就能读全文。
    # 注意要拿**刷新后的行**：open_ticket 返回的是开单那一刻的快照。
    claimed = tickets.get_ticket(claimed["ticket_id"])
    closed = tickets.get_ticket(closed["ticket_id"])
    assert tickets.can_view(opened, "p_b") is True
    assert tickets.can_view(claimed, "p_b") is False
    assert tickets.can_view(claimed, "p_a") is True
    assert tickets.can_view(closed, "p_b") is False
    assert tickets.can_view(closed, "p_a") is True
    assert tickets.can_view(claimed, "p_b", is_admin=True) is True


def test_admin_can_force_close_an_unclaimed_ticket(store):
    sid = _owned_session(store)
    ticket = tickets.open_ticket(sid, "p_demo", "")

    with pytest.raises(tickets.TicketConflict):
        tickets.close_ticket(ticket["ticket_id"], "p_admin")

    closed = tickets.close_ticket(ticket["ticket_id"], "p_admin", allow_override=True)
    assert closed["status"] == "closed"
    assert closed["closed_by"] == "p_admin"
    assert closed["claimed_by"] is None


def test_transfer_reassigns_and_release_puts_it_back_in_the_queue(store):
    sid = _owned_session(store)
    ticket = tickets.open_ticket(sid, "p_demo", "")
    tickets.claim_ticket(ticket["ticket_id"], "p_a", "小 A")

    moved = tickets.transfer_ticket(
        ticket["ticket_id"], {"id": "p_b", "name": "小 B"}
    )
    assert moved["status"] == "claimed"
    assert moved["claimed_by"] == "p_b"
    assert moved["claimed_by_name"] == "小 B"
    assert moved["queue_position"] == 0

    # 原认领人已经不能再回 / 关。
    with pytest.raises(tickets.TicketConflict):
        tickets.reply_ticket(ticket["ticket_id"], "p_a", "小 A", "旧客服插话")
    with pytest.raises(tickets.TicketConflict):
        tickets.close_ticket(ticket["ticket_id"], "p_a")

    released = tickets.transfer_ticket(ticket["ticket_id"], None)
    assert released["status"] == "open"
    assert released["claimed_by"] is None
    assert released["claimed_by_name"] is None
    assert released["claimed_at"] is None
    assert released["queue_position"] == 0

    # 已经回到队列的单再放一次是幂等的，不该报错（admin 点重了很常见）。
    again = tickets.transfer_ticket(ticket["ticket_id"], None)
    assert again["status"] == "open"


def test_closed_ticket_cannot_be_transferred(store):
    sid = _owned_session(store)
    ticket = tickets.open_ticket(sid, "p_demo", "")
    tickets.claim_ticket(ticket["ticket_id"], "p_a", "小 A")
    tickets.close_ticket(ticket["ticket_id"], "p_a")

    with pytest.raises(tickets.TicketConflict):
        tickets.transfer_ticket(ticket["ticket_id"], {"id": "p_b", "name": "小 B"})


def test_purge_takes_the_tickets_with_it(store):
    from datetime import datetime, timedelta

    sid = _owned_session(store)
    tickets.open_ticket(sid, "p_demo", "")
    stamp = (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d %H:%M:%S")
    conn = store.connect()
    conn.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (stamp, sid))
    conn.commit()

    result = store.purge(older_than_days=30)

    assert result["tickets"] == 1
    assert tickets.list_tickets(status="all") == []
