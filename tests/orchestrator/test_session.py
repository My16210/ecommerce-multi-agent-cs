"""
会话存储的测试。

**整份文件不联网、不调模型**——它只碰 sqlite。

这里最要紧的一条是 `test_history_keeps_the_most_recent_turns`：截断方向写反
（留最早的几轮而不是最近的几轮）**不会报任何错**，只会让模型拿到一段很旧的
上文。这种错只能靠测试钉住，靠人看代码是看不出来的。
"""
import threading

import pytest

from ecom.orchestrator import session


@pytest.fixture
def store(tmp_path, monkeypatch):
    """每个用例一个独立的会话库文件。不清连接缓存就会串库。"""
    monkeypatch.setattr(session, "SESSION_DB_PATH", str(tmp_path / "sessions.db"))
    session.reset_connection()
    yield session
    session.reset_connection()


def _age(session_id: str, days: int, *, created: bool = False, updated: bool = True) -> None:
    """把会话的时间戳往前推 N 天，用来测过期。"""
    from datetime import datetime, timedelta

    stamp = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    conn = session.connect()
    if created:
        conn.execute("UPDATE sessions SET created_at = ? WHERE session_id = ?", (stamp, session_id))
    if updated:
        conn.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (stamp, session_id))
    conn.commit()


# ——— 会话号 ———

def test_a_blank_session_id_gets_replaced_by_a_real_one(store):
    sid = store.ensure_session("")
    assert sid
    assert store.get_session(sid) is not None


def test_session_ids_are_long_enough_to_be_credentials(store):
    """session_id 就是凭证（拿到它就能读会话），所以不能是可枚举的短串。

    `uuid4().hex[:12]` 只有 48 bit，撞库成本极低——trace.request_id 那种形状
    是给日志用的，不是给鉴权用的。这条测试同时钉住"长度"和"不重复"。
    """
    ids = {store.new_session_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(len(i) >= 30 for i in ids)


def test_ensure_session_is_idempotent(store):
    sid = store.ensure_session("mine")
    first = store.get_session(sid)["created_at"]
    store.ensure_session(sid)
    assert store.get_session(sid)["created_at"] == first


# ——— 读写历史 ———

def test_history_comes_back_in_chronological_order(store):
    sid = store.ensure_session("chat")
    store.append_turn(sid, "第一问", "第一答")
    store.append_turn(sid, "第二问", "第二答")

    assert store.history(sid) == [
        {"role": "user", "content": "第一问"},
        {"role": "assistant", "content": "第一答"},
        {"role": "user", "content": "第二问"},
        {"role": "assistant", "content": "第二答"},
    ]


def test_history_keeps_the_most_recent_turns(store):
    """**留最近的，不是最早的。**

    写反了（ASC + LIMIT 而不是 DESC + LIMIT + 反转）不会报错，只会让模型
    拿到一段很旧的上文——所以这条断言两边都钉：留下了哪些、丢掉了哪些。
    """
    sid = store.ensure_session("chat")
    for i in range(1, 6):  # 5 轮
        store.append_turn(sid, f"问{i}", f"答{i}")

    got = store.history(sid, max_turns=2)

    assert [m["content"] for m in got] == ["问4", "答4", "问5", "答5"]
    assert "问1" not in [m["content"] for m in got]


def test_history_of_an_unknown_session_is_empty_not_an_error(store):
    assert store.history("nobody") == []


def test_history_can_be_switched_off(store):
    """max_turns=0 就是"不带历史"，这条路径要能走通。"""
    sid = store.ensure_session("chat")
    store.append_turn(sid, "问", "答")
    assert store.history(sid, max_turns=0) == []


def test_two_sessions_do_not_bleed_into_each_other(store):
    a = store.ensure_session("a")
    b = store.ensure_session("b")
    store.append_turn(a, "A的问题", "A的回答")
    store.append_turn(b, "B的问题", "B的回答")

    assert [m["content"] for m in store.history(a)] == ["A的问题", "A的回答"]
    assert [m["content"] for m in store.history(b)] == ["B的问题", "B的回答"]


def test_an_empty_answer_is_still_recorded(store):
    """模型答不出来时 answer 是空串——那一轮还是要记的，
    否则下一个人看到的是一个悬空的问题。"""
    sid = store.ensure_session("chat")
    store.append_turn(sid, "问", "")
    assert len(store.history(sid)) == 2


def test_appending_creates_the_session_if_it_is_missing(store):
    store.append_turn("凭空来的", "问", "答")
    assert store.get_session("凭空来的") is not None


# ——— 删除 ———

def test_delete_removes_the_messages_too(store):
    sid = store.ensure_session("chat")
    store.append_turn(sid, "问", "答")

    assert store.delete_session(sid) == 2
    assert store.get_session(sid) is None
    assert store.history(sid) == []
    # 消息行是真删了，不是只删了会话那一行（否则会留下永远读不到的孤儿数据）
    left = store.connect().execute(
        "SELECT COUNT(*) AS n FROM messages WHERE session_id = ?", (sid,)
    ).fetchone()["n"]
    assert left == 0


def test_deleting_an_unknown_session_reports_zero(store):
    assert store.delete_session("不存在") == 0


# ——— 过期 ———

def test_purge_removes_only_stale_sessions(store):
    old = store.ensure_session("old")
    fresh = store.ensure_session("fresh")
    store.append_turn(old, "老问题", "老回答")
    store.append_turn(fresh, "新问题", "新回答")
    _age(old, days=40)

    result = store.purge(older_than_days=30)

    assert result == {"sessions": 1, "messages": 2, "memories": 0}
    assert store.get_session(old) is None
    assert store.get_session(fresh) is not None


def test_purge_measures_idle_time_not_age(store):
    """按**最后一次说话**算，不是按创建时间。

    一个持续聊了 40 天的会话，不该在第 31 天因为"建得太早"被删掉。
    """
    sid = store.ensure_session("chat")
    _age(sid, days=40, created=True, updated=False)  # 建得早，但刚刚还在说话

    assert store.purge(older_than_days=30)["sessions"] == 0
    assert store.get_session(sid) is not None


def test_purge_on_an_empty_store_is_a_no_op(store):
    assert store.purge(older_than_days=30) == {"sessions": 0, "messages": 0, "memories": 0}


# ——— 记忆（存储这一半；蒸馏/召回在 test_memory.py）———

def _fill(store, sid: str, turns: int) -> None:
    """给会话灌 `turns` 轮问答。消息 id 从 1 开始，一问一答两条。"""
    for i in range(turns):
        store.append_turn(sid, f"问{i}", f"答{i}")


def test_out_of_window_messages_stops_at_the_window_edge(store):
    """滑出窗口的才蒸馏，窗口内的不蒸馏。

    `HISTORY_TURNS` 默认 6，所以窗口是 12 条消息。这里按 `_window_size()`
    现算而不是写死 12——写死的话，谁把 `HISTORY_TURNS` 调一下就红一片，
    而那种红说明不了代码有问题。
    """
    window = store._window_size()
    turns = window // 2 + 4  # 比窗口多 4 轮，保证真有滑出去的
    total = turns * 2
    sid = store.ensure_session("chat")
    _fill(store, sid, turns)

    rows = store.out_of_window_messages(sid, limit=10**6)

    assert [r["id"] for r in rows] == list(range(1, total - window + 1))
    # 两段互补、不重叠：窗口内 + 窗口外 正好是全部消息，中间没有缝
    assert len(rows) + len(store.history(sid)) == total


def test_distilled_messages_are_not_offered_again(store):
    """水位线挡住已经蒸馏过的那一段——否则每跑一次 distill 都要从头付一次钱。"""
    window = store._window_size()
    turns = window // 2 + 4
    sid = store.ensure_session("chat")
    _fill(store, sid, turns)

    store.add_memory(sid, "摘要", "关键词", None, turns * 2 - window)

    assert store.out_of_window_messages(sid, limit=10**6) == []


def test_a_short_session_has_nothing_to_distill(store):
    """窗口都还没装满，所有消息都在窗口里——`distill` 说的「无可蒸馏」就是这种。"""
    sid = store.ensure_session("chat")
    _fill(store, sid, store._window_size() // 2)

    assert store.out_of_window_messages(sid, limit=10**6) == []


def test_writing_the_same_watermark_twice_keeps_one_row(store):
    """幂等由**数据库**保证，不是由 `_lock` 保证。

    那把锁是进程内的，而 `distill` 是独立进程的 CLI：cron 撞上手动执行时，
    两边会读到同一个水位线、蒸馏同一批消息、各插一条。`UNIQUE` + `INSERT OR
    IGNORE` 让第二次成为 no-op——**这条测试就是在钉那个约束真的建上了**。
    """
    sid = store.ensure_session("chat")
    _fill(store, sid, 10)

    assert store.add_memory(sid, "摘要一", "词", None, 8) is True
    assert store.add_memory(sid, "摘要二", "词", None, 8) is False

    rows = store.memories_for(sid)
    assert len(rows) == 1
    assert rows[0]["summary"] == "摘要一"  # 先到的那条留着，后来的被丢掉


def test_memories_come_back_oldest_first(store):
    """按时间正序读回来——它是一段经历，倒着读没有意义。"""
    sid = store.ensure_session("chat")
    _fill(store, sid, 20)
    store.add_memory(sid, "第一段", "", None, 8)
    store.add_memory(sid, "第二段", "", None, 16)

    assert [m["summary"] for m in store.memories_for(sid)] == ["第一段", "第二段"]


def test_delete_takes_the_memories_with_it(store):
    """删了会话却留着摘要 = 隐私上的假动作：用户以为删干净了，摘要还躺在库里。

    返回契约仍然是**消息条数**（`api.py` / CLI 在用它），所以这里同时钉住
    "删干净了"和"返回值没变"两件事。
    """
    sid = store.ensure_session("chat")
    _fill(store, sid, 10)
    store.add_memory(sid, "摘要", "", None, 8)

    assert store.delete_session(sid) == 20
    assert store.memories_for(sid) == []
    left = store.connect().execute(
        "SELECT COUNT(*) AS n FROM memories WHERE session_id = ?", (sid,)
    ).fetchone()["n"]
    assert left == 0


def test_purge_takes_the_memories_with_it(store):
    sid = store.ensure_session("chat")
    _fill(store, sid, 10)
    store.add_memory(sid, "摘要", "", None, 8)
    _age(sid, days=40)

    result = store.purge(older_than_days=30)

    assert result == {"sessions": 1, "messages": 20, "memories": 1}
    assert store.memories_for(sid) == []


def test_memories_are_scoped_to_their_own_session(store):
    """**这条表是会话内的，不是用户画像**——两个会话之间不许串。

    写法上没有任何东西拦着"按关键词跨会话捞摘要"，所以只能靠这条钉住。
    """
    a = store.ensure_session("a")
    b = store.ensure_session("b")
    _fill(store, a, 10)
    _fill(store, b, 10)
    store.add_memory(a, "A 的摘要", "", None, 8)

    assert [m["summary"] for m in store.memories_for(b)] == []


# ——— 并发 ———

def test_concurrent_writes_do_not_lose_messages(store):
    """`/chat` 是 `def`，FastAPI 把它丢进线程池——所以真的会并发写。

    原来那个进程内字典没有任何锁，这里是补上的。这条测试跑不跑得出问题
    取决于调度，所以它的价值是"**能失败**"：锁被删掉时它有机会红，
    而不是一个永远绿的摆设。
    """
    sid = store.ensure_session("chat")
    threads = [
        threading.Thread(target=store.append_turn, args=(sid, f"问{i}", f"答{i}"))
        for i in range(20)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(store.history(sid, max_turns=1000)) == 40


def test_session_ownership_is_exclusive(store):
    """先认领者赢；另一个 principal 拿同一个 session_id 必须被拒。"""
    sid = store.ensure_session("", owner_id="principal-a")
    assert store.get_session(sid)["owner_id"] == "principal-a"

    with pytest.raises(store.SessionAccessError):
        store.ensure_session(sid, owner_id="principal-b")

    # 同一个 principal 再来一次（比如下一轮）必须正常
    assert store.ensure_session(sid, owner_id="principal-a") == sid


def test_legacy_session_can_be_claimed_once(store):
    """老库里 owner_id='' 的会话：第一个 principal 可以认领，之后锁死。"""
    sid = store.ensure_session("legacy")
    assert store.get_session(sid)["owner_id"] == ""

    assert store.ensure_session(sid, owner_id="principal-a") == sid
    assert store.get_session(sid)["owner_id"] == "principal-a"
    with pytest.raises(store.SessionAccessError):
        store.ensure_session(sid, owner_id="principal-b")
