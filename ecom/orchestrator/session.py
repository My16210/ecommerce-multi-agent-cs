"""
会话存储（SQLite）：让多轮对话活过进程重启。

【为什么不复用 knowledge 那边已经跑通的 SqliteSaver】
`ecom/knowledge/cli/ask.py` 用 `SqliteSaver` + `thread_id` 把知识库图的**状态**
存进 `data/knowledge/chat_history.db`。看着应该直接搬过来统一，**但不该搬**，
四条理由，从硬到软：

1. **`chat_history` 在 supervisor 里是"节点输入"，不是"图状态"。**
   `supervisor.answer()` 把它塞进 `make_initial_state`，图**从不往它里面写**：
   没有 reducer、没有累积节点，每一轮都是调用方给一份新的。要让 checkpointer
   "接管"历史，得先往图里加一个累积节点或给这个通道配 reducer——那是在改图的
   语义，不是在加一个存储。
   （读它的是 router（指代消解）、order worker（抽实体）和 knowledge worker
   （生成答案）；**aggregate 刻意不读**。这里原来写的是"图只在 aggregate 节点
   读一次"，**说反了**——完整的更正见 core/state.py 的 `CSState` 注释。）
2. **`state.py` 已经警告过那条路有坑**：`results` 是 `Annotated[list, operator.add]`，
   注释写着「往 add-型通道写 `[]` 等于没写、跨轮累加、清不掉」。给 `chat_history`
   配同样的 reducer，会得到一条**只增不减、没法在开头裁剪**的通道——而多轮历史
   的正确行为恰恰是"留最近 N 轮、丢掉更早的"，与 add-型语义相反。
3. **checkpointer 持久化的是整份 state**，`results` / `round` / `tried` 也会跟着
   跨请求累积。这三个都是**每请求**语义（`tried` 是"本轮试过谁"，跨请求累积会让
   第二问直接没得可试）。持久化整份 state 从语义上就是错的。
4. **两个图不是同一个图。** 同一个 `thread_id` 命名空间里塞两张不同图的
   checkpoint，排查时只会互相误导。

**这件事的反面也要说清，否则上面的理由没有分量**：统一成一套的好处是
"会话"在整个项目里只有一个答案，新人只学一次。**什么条件下该反过来选**——
如果哪天 supervisor 需要在**图内部**做带记忆的决策（比如记住"这个用户上一轮
已经在等人工了"），那就不是"历史作为输入"而是"历史作为状态"，那时候
checkpointer 才是对的工具。现在没有这个需求。

所以这里存的是**扁平的消息行**：一问一答两条，读回来拼成 `chat_history`
喂给 `answer()`。图本身一行都不用改。

【session_id 是凭证，不是账号——这条必须写在最显眼的地方】
`workers/order.py` 的 `_guess_customer()` 是**从用户的话里猜客户号**的
（代码里标了警告）。也就是说这个系统里的"客户"是**未认证**的：没有登录态，
没有可核验的身份。因此：

    **拿到 session_id 就能读到这个会话的全部历史。**

它等价于一个 bearer token。具体影响三条：
  · 别把它当账号系统用。要谈"这个会话属于谁"，前提是先有登录态。
  · **没有身份，就没法按"账号"响应删除请求**，只能按 session_id 删。
  · 所以保留期必须**按时间**清（`purge`），不能指望"用户走了就可以删"。

【三个问题的答案（README 里说"要真做，得先想清楚"的就是这三条）】
  存哪     data/orchestrator/sessions.db，单文件、单节点、不跨副本。
           不引 Redis——那是"多副本怎么共享会话"的另一个问题。
  怎么删   `python -m ecom.orchestrator.cli.session purge`（按时间）
           / `delete <id>`（按会话）。CLI 和界面共用下面这些函数。
  合规     明文落盘，**不做**静态加密和字段级脱敏——那是另一个量级的事，
           这里能做的缓解只有两条：短保留 + 能删。别把它说成"合规了"。

【和 knowledge 那个库的另一个区别：这里只有 SQLite，没有 MySQL 分支】
会话是纯演示特性，接 MySQL 是过度设计；更重要的是**会话数据必须有自己的
文件和自己的保留策略**——如果走 `ORDER_DB_DSN`，接了 MySQL 就会把用户说过的话
写进共享的订单库，那是把两类数据混在一处。所以这个模块自己开 sqlite，
不走 `orchestrator.db.engine.Database`。
"""
import secrets
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

from ecom.orchestrator.settings import HISTORY_TURNS, SESSION_DB_PATH, SESSION_TTL_DAYS
from ecom.orchestrator.utils.logger import setup_logger

logger = setup_logger(__name__)


class SessionAccessError(Exception):
    """这个会话不属于当前 principal。API 把它翻成 404，不暴露"存在但没权限"。"""

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    owner_id   TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    author     TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at);
CREATE TABLE IF NOT EXISTS memories (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id         TEXT NOT NULL,
    summary            TEXT NOT NULL,
    keywords           TEXT NOT NULL,
    embedding          BLOB,
    through_message_id INTEGER NOT NULL,
    created_at         TEXT NOT NULL,
    UNIQUE(session_id, through_message_id)
);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id, id);
CREATE TABLE IF NOT EXISTS tickets (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id       TEXT NOT NULL UNIQUE,
    session_id      TEXT NOT NULL,
    owner_id        TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL CHECK (status IN ('open', 'claimed', 'closed')),
    reason          TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    claimed_by      TEXT,
    claimed_by_name TEXT,
    claimed_at      TEXT,
    closed_at       TEXT,
    closed_by       TEXT
);
-- 一个会话同时最多一张未关闭的工单：重复触发转人工不该在队列里堆出好几行。
CREATE UNIQUE INDEX IF NOT EXISTS idx_tickets_live_session
    ON tickets(session_id) WHERE status != 'closed';
CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status, created_at);
"""

_conn = None
# **这把锁是补上一个真缺陷，不是预防性加的。** api.py 原来那个进程内字典
# `_SESSIONS` 没有任何保护——`/chat` 是 `def`，FastAPI 把它丢进线程池，
# 两个请求同时写同一个 dict 是可能的。原来侥幸没炸是因为 dict 操作快且
# GIL 挡了一部分；换成 sqlite 之后没有这层侥幸了（跨线程用同一个连接
# 会直接抛 ProgrammingError）。
#
# 读也一起锁：这里进程内**共用同一个连接**，而 sqlite3 的连接对象本身
# 不是为并发设计的。WAL 解决的是**跨进程**（purge 的 CLI 和服务同时开着）
# 的读写并发，不是同一个连接内部的并发——两件事，别互相顶替。
_lock = threading.Lock()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def connect() -> sqlite3.Connection:
    """
    打开会话库，**进程内复用同一个连接**，建表脚本只在首次打开时跑一次。

    所以下面那些函数里不能再有 `finally: conn.close()`——那会把缓存连接
    关掉，下次调用拿到一个已关闭的连接。
    """
    global _conn
    if _conn is None:
        Path(SESSION_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(SESSION_DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # WAL：让"purge 命令在跑"和"服务在写"不互相阻塞。默认的 journal 模式下
        # 一个写者会挡住所有读者，而 purge 是一次全表扫描。
        conn.execute("PRAGMA journal_mode=WAL")
        # 万一还是撞上锁（比如宿主上另一个进程正持着），等 5 秒再报错，
        # 而不是立刻抛 "database is locked"。
        conn.execute("PRAGMA busy_timeout=5000")
        conn.executescript(SCHEMA)
        # 老库没有 owner_id：CREATE TABLE IF NOT EXISTS 不会改已存在的表。
        # 这里做一次幂等补列，而不是要求"删库重灌"——会话库是真数据，不能丢。
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(sessions)")}
        if "owner_id" not in columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN owner_id TEXT NOT NULL DEFAULT ''")
        # 人工客服答复也落在同一张 messages 表里（role='agent'），`author` 记是谁回的。
        # 合并成一张表而不是另开 ticket_messages：历史回看、导出、删除都只有一条路径，
        # 删会话时也不会漏掉"人工那几句还留着"。
        message_columns = {row["name"] for row in conn.execute("PRAGMA table_info(messages)")}
        if "author" not in message_columns:
            conn.execute("ALTER TABLE messages ADD COLUMN author TEXT NOT NULL DEFAULT ''")
        conn.commit()
        _conn = conn
    return _conn


def reset_connection() -> None:
    """
    丢掉缓存的连接，下次 connect() 重新按 SESSION_DB_PATH 打开。

    给测试用：每个用例都要把会话库指到自己的临时文件，不清缓存就会串库。
    """
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None


def new_session_id() -> str:
    """
    生成会话号。

    用 `secrets.token_urlsafe(24)` 而不是 `uuid4().hex[:12]`（trace.request_id
    那种形状）：**因为 session_id 是凭证**（见模块 docstring）。uuid4 截 12 位
    只剩 48 bit，可枚举；token_urlsafe(24) 是 192 bit，而且 URL 安全、能直接
    贴进浏览器或 curl。
    """
    return secrets.token_urlsafe(24)


def ensure_session(session_id: str, owner_id: str = "") -> str:
    """
    会话不存在就建一个。返回 session_id 本身，方便链式写。

    `owner_id` 非空时做归属检查：会话已有主且不是他 → `SessionAccessError`；
    还没有主（老库 / 未登录时建的）→ 认领。`owner_id` 为空表示"不检查"，
    给 CLI 和本地演示用——API 在鉴权模式下永远会传。
    """
    sid = (session_id or "").strip()
    if not sid:
        sid = new_session_id()
    owner = str(owner_id or "").strip()
    now = _now()
    with _lock:
        conn = connect()
        conn.execute(
            "INSERT INTO sessions (session_id, created_at, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(session_id) DO NOTHING",
            (sid, now, now),
        )
        if owner:
            row = conn.execute(
                "SELECT owner_id FROM sessions WHERE session_id = ?", (sid,)
            ).fetchone()
            current = (row["owner_id"] if row else "") or ""
            if current and current != owner:
                raise SessionAccessError(f"会话 {sid} 不属于当前 principal")
            if not current:
                cursor = conn.execute(
                    "UPDATE sessions SET owner_id = ? WHERE session_id = ? AND owner_id = ''",
                    (owner, sid),
                )
                # 进程内的锁挡不住另一个进程（CLI / 多 worker）同时认领：
                # 两个 principal 都读到 ''，先写的那条赢，后手 update 命中 0 行。
                # 这时必须再读一次确认，否则后手会以为自己认领成功。
                if cursor.rowcount == 0:
                    latest = conn.execute(
                        "SELECT owner_id FROM sessions WHERE session_id = ?", (sid,)
                    ).fetchone()
                    if ((latest["owner_id"] if latest else "") or "") != owner:
                        raise SessionAccessError(f"会话 {sid} 不属于当前 principal")
        conn.commit()
    return sid


def append_turn(session_id: str, question: str, answer: str, owner_id: str = "") -> None:
    """
    记一轮问答。一问一答两行，**同一时刻写入**——分成两次调用的话，
    中途挂掉就会留下一个只有提问没有回答的会话，读回来喂给模型时
    它看到的是一个悬空的问题。
    """
    sid = ensure_session(session_id, owner_id=owner_id)
    now = _now()
    with _lock:
        conn = connect()
        conn.executemany(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            [(sid, "user", question, now), (sid, "assistant", answer or "", now)],
        )
        conn.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (now, sid))
        conn.commit()


def history(session_id: str, max_turns: int = None) -> list:
    """
    取回最近 `max_turns` 轮历史，**按时间正序**，形状就是 `answer()` 要的
    `[{"role": "user"/"assistant", "content": ...}]`。

    截断方向是这里最容易写错的地方：要留**最近**的几轮，不是最早的几轮。
    所以 SQL 里是 `ORDER BY id DESC LIMIT n`（先拿到最新的），拿到之后再
    反转成正序——反过来写（ASC + LIMIT）不会报错，只会让模型拿到一段
    很旧的上文，而且完全看不出来。
    """
    sid = (session_id or "").strip()
    if not sid:
        return []
    turns = HISTORY_TURNS if max_turns is None else max_turns
    if turns <= 0:
        return []

    with _lock:
        rows = connect().execute(
            "SELECT role, content FROM messages WHERE session_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (sid, turns * 2),
        ).fetchall()
    # 人工客服的答复（role='agent'）对模型来说就是"上一轮的助手说过的话"。
    # **不把这个映射写在这里也不报错**：worker 会收到一个它不认识的 role，
    # 各家模型对未知 role 的处理从忽略到报错都有，属于"换模型才炸"的坑。
    return [
        {"role": "assistant" if r["role"] == "agent" else r["role"], "content": r["content"]}
        for r in reversed(rows)
    ]


# ---------------------------------------------------------------------------
# 会话内长期记忆的水位线切分（蒸馏/召回在 ecom/orchestrator/memory.py）
# ---------------------------------------------------------------------------
def _window_size() -> int:
    """
    "窗口"是多少条消息。`history()` 取回的就是这个范围，`out_of_window_messages()`
    取回的是它**之前**那一段——两者互补且不重叠，中间没有缝。

    抽成函数、不把 `HISTORY_TURNS * 2` 散在 SQL 里：这个算式一处分叉，就会出现
    "记忆覆盖了窗口内的消息"或"有一批消息永远蒸馏不到"，**两种都不报错**。
    """
    return HISTORY_TURNS * 2


def out_of_window_messages(session_id: str, limit: int) -> list:
    """
    已经滑出 `history()` 窗口、且还没被蒸馏过的消息，**从旧到新**，最多 `limit` 条。

    返回的行含 `id`——水位线（`through_message_id`）要取这一批最后一条的 id。

    两道界，方向相反：
      · 下界 `id > 已蒸馏到的最大 id`：蒸馏过的消息不再重复喂给模型。
      · 上界 `id <= 会话最大 id - 窗口宽度`：**还在窗口里的不蒸馏**。它们下一轮
        就会由 `history()` 原样喂给模型，再蒸馏一遍是重复付费，而且模型会读到
        一段自己有完整原文的删节版。
    """
    sid = (session_id or "").strip()
    if not sid or limit <= 0:
        return []
    with _lock:
        rows = connect().execute(
            """
            SELECT id, role, content FROM messages
            WHERE session_id = ?
              AND id > COALESCE(
                    (SELECT MAX(through_message_id) FROM memories WHERE session_id = ?), 0)
              AND id <= COALESCE(
                    (SELECT MAX(id) FROM messages WHERE session_id = ?), 0) - ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (sid, sid, sid, _window_size(), limit),
        ).fetchall()
    return [dict(r) for r in rows]


def add_memory(session_id: str, summary: str, keywords: str,
               embedding: bytes, through_message_id: int) -> bool:
    """
    写一条记忆。返回**这条是不是新写进去的**（False = 已有一条覆盖到同一处，跳过）。

    `INSERT OR IGNORE` + `UNIQUE(session_id, through_message_id)` **不是装饰**：
    上面那把 `_lock` 是**进程内**的，而蒸馏是独立进程的 CLI。两个蒸馏同时跑
    （cron 撞上手动执行）时，两边读到同一个水位线、蒸馏同一批消息、各插一条
    ——**WAL 解决的是 DB 锁，解决不了这种逻辑上的双重蒸馏**。这个约束让第二次
    插入变成 no-op。所以幂等由**数据库**保证，不是由锁保证。
    """
    with _lock:
        conn = connect()
        cursor = conn.execute(
            "INSERT OR IGNORE INTO memories "
            "(session_id, summary, keywords, embedding, through_message_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, summary, keywords, embedding, through_message_id, _now()),
        )
        conn.commit()
    return cursor.rowcount > 0


def memories_for(session_id: str, limit: int = None) -> list:
    """
    取这个会话沉淀下来的记忆，**按时间正序**（读起来像一段经历）。

    返回的行含 `embedding`（BLOB），形状由调用方解释——`session.py` 是
    "只有 SQLite 的笨存储"，它不认识向量（边界见模块开头）。
    """
    sid = (session_id or "").strip()
    if not sid:
        return []
    sql = "SELECT * FROM memories WHERE session_id = ? ORDER BY id ASC"
    args: list = [sid]
    if limit is not None:
        sql += " LIMIT ?"
        args.append(limit)
    with _lock:
        rows = connect().execute(sql, args).fetchall()
    return [dict(r) for r in rows]


def list_sessions(limit: int = 50, owner_id: str = None) -> list:
    """
    会话列表，最近更新的在前。带消息条数和第一条提问（界面拿它当标题）。

    `owner_id=None` 表示**不加归属过滤**，给 CLI 看全库用；传字符串则只返回
    这个 principal 的会话。**不要把空字符串当成"不过滤"**——匿名会话的
    owner_id 就是空串，那会把所有人的匿名会话混在一起。
    """
    where = ""
    args: list = []
    if owner_id is not None:
        where = "WHERE s.owner_id = ?"
        args.append(owner_id)
    args.append(limit)
    with _lock:
        rows = connect().execute(
            f"""
            SELECT s.session_id, s.created_at, s.updated_at, s.owner_id,
                   COUNT(m.id) AS messages,
                   COALESCE((
                       SELECT first.content FROM messages first
                       WHERE first.session_id = s.session_id AND first.role = 'user'
                       ORDER BY first.id ASC LIMIT 1
                   ), '') AS title
            FROM sessions s
            LEFT JOIN messages m ON m.session_id = s.session_id
            {where}
            GROUP BY s.session_id
            ORDER BY s.updated_at DESC
            LIMIT ?
            """,
            args,
        ).fetchall()
    return [dict(r) for r in rows]


def get_session(session_id: str) -> dict:
    """单个会话的元信息（不含消息正文）。查不到返回 None。"""
    with _lock:
        row = connect().execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    return dict(row) if row else None


def first_question(session_id: str) -> str:
    """
    这个会话的第一条用户提问。列表和明细都用它做标题。

    单独查而不是从 `transcript()` 里拿：明细只取最近 200 条，一个聊得很长的
    会话，第一条提问早就滑出去了——从窗口里取会让标题变成「未命名会话」。
    """
    sid = (session_id or "").strip()
    if not sid:
        return ""
    with _lock:
        row = connect().execute(
            "SELECT content FROM messages WHERE session_id = ? AND role = 'user' "
            "ORDER BY id ASC LIMIT 1",
            (sid,),
        ).fetchone()
    return row["content"] if row else ""


def transcript(session_id: str, max_messages: int = 200) -> list:
    """
    一个会话的消息明细，**按时间正序**，带 `id` 和时间戳。

    和 `history()` 的区别是用途：`history()` 是喂给模型的窗口（只留最近几轮、
    不带 id），这里是给人回看的历史（更长、带 id 方便前端做 key）。两者不要
    互相顶替——把 `transcript` 直接喂模型会悄悄把上下文窗口撑大。
    """
    sid = (session_id or "").strip()
    if not sid or max_messages <= 0:
        return []
    with _lock:
        rows = connect().execute(
            "SELECT id, role, content, created_at, author FROM messages "
            "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (sid, max_messages),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def append_agent_message(session_id: str, content: str, author: str, owner_id: str = "") -> int:
    """
    人工客服在一个会话里回一句。返回消息 id。

    写进 `messages`（role='agent'）而不是单独的工单消息表：用户重新打开会话时
    走的是同一个 `transcript()`，人工说过的话必须和机器说过的话一起出现；
    `history()` 会把它映射成 assistant 喂回模型。**写入前先做归属检查**——
    `ensure_session(owner_id=...)` 会拒绝别人的会话。
    """
    sid = ensure_session(session_id, owner_id=owner_id)
    now = _now()
    with _lock:
        conn = connect()
        cursor = conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at, author) "
            "VALUES (?, 'agent', ?, ?, ?)",
            (sid, content, now, str(author or "")[:64]),
        )
        conn.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (now, sid))
        conn.commit()
        return int(cursor.lastrowid)


@contextmanager
def transaction():
    """
    在会话库上持锁执行一组读写。

    工单模块和会话模块共用**同一个** sqlite 连接：sqlite3 的连接对象不是为
    并发设计的，各自持一把锁仍然可能同时用同一个连接。所以工单那边不自己
    开锁，而是通过这里拿连接——两把锁保护一个连接，等于没保护。
    """
    with _lock:
        yield connect()


def _delete_rows(conn, sid: str) -> int:
    """在已持锁的连接上删消息 / 记忆 / 工单 / 会话，返回删掉的消息条数。"""
    cursor = conn.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
    removed = cursor.rowcount
    conn.execute("DELETE FROM memories WHERE session_id = ?", (sid,))
    # 工单里有"这位用户找过人工、谁接待的、接待到哪一步"，和会话正文一样敏感。
    # 删会话却留着工单，和删了正文留着摘要同一种假动作。
    conn.execute("DELETE FROM tickets WHERE session_id = ?", (sid,))
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
    conn.commit()
    return removed


def delete_session(session_id: str) -> int:
    """
    删一个会话连同它的消息。返回删掉的**消息**条数（0 表示这个会话本来就不存在）。

    三条 DELETE 在同一个连接、同一次提交里——不靠外键级联，因为 SQLite
    默认不开 `PRAGMA foreign_keys`，写了 `ON DELETE CASCADE` 也不生效。
    依赖一个默认关闭的开关，是那种"今天对、换个环境就静默留垃圾"的写法。

    **memories 必须一起删**：记忆是这段对话的摘要，里面同样有用户说过的话。
    "删了会话却留着它的摘要"不是漏删一行——它是**隐私上的假动作**：用户以为
    删干净了，而摘要还躺在库里。返回契约仍然是**消息条数**（调用方在用它）。
    """
    sid = (session_id or "").strip()
    if not sid:
        return 0
    with _lock:
        return _delete_rows(connect(), sid)


def delete_session_owned(session_id: str, owner_id: str):
    """
    只删属于 `owner_id` 的会话。返回消息条数；不存在或不属于他返回 `None`。

    归属检查和删除在**同一把锁、同一个连接**里做。先在 API 层 `get_session`
    再删也能跑，但那是两次独立读取——删除权限的判断不该和删除动作之间留缝。
    """
    sid = (session_id or "").strip()
    owner = str(owner_id or "").strip()
    if not sid or not owner:
        return None
    with _lock:
        conn = connect()
        row = conn.execute(
            "SELECT owner_id FROM sessions WHERE session_id = ?", (sid,)
        ).fetchone()
        if row is None or (row["owner_id"] or "") != owner:
            return None
        return _delete_rows(conn, sid)


def purge(older_than_days: int = None) -> dict:
    """
    删掉太久没动静的会话。返回
    `{"sessions": n, "messages": m, "memories": k, "tickets": t}`。

    **memories 一起删、一起报**：它是这段对话的摘要，里面同样有用户说过的话。
    只删消息留着摘要，等于"用户以为删干净了，而摘要还躺在库里"。计数单列出来，
    是因为**没被报出来的删除，等于没发生**——CLI 只报消息数的话，这件事在
    输出里根本看不见。

    按 `updated_at`（最后一次说话的时间）算，不是 `created_at`——一个持续
    聊了 40 天的会话不该在第 31 天被删掉。

    **什么时候调它**：`api.py` / `app.py` 启动时一次，以及 CLI 显式调。
    刻意**不做**"读的时候顺手删"那种懒清理：那会让一个只读操作产生写副作用，
    出问题时（"我的会话怎么没了"）极难定位。代价是长驻进程不会自我清理，
    得外挂一行 cron——这条写进 README 的已知限制。
    """
    days = SESSION_TTL_DAYS if older_than_days is None else older_than_days
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    with _lock:
        conn = connect()
        # 顺序不能换：memories 和 messages 都要在删 sessions 之前删，因为
        # 它们的子查询依据的就是 sessions 里那批过期的 id。
        memories = conn.execute(
            "DELETE FROM memories WHERE session_id IN "
            "(SELECT session_id FROM sessions WHERE updated_at < ?)",
            (cutoff,),
        ).rowcount
        removed = conn.execute(
            "DELETE FROM messages WHERE session_id IN "
            "(SELECT session_id FROM sessions WHERE updated_at < ?)",
            (cutoff,),
        ).rowcount
        tickets = conn.execute(
            "DELETE FROM tickets WHERE session_id IN "
            "(SELECT session_id FROM sessions WHERE updated_at < ?)",
            (cutoff,),
        ).rowcount
        sessions = conn.execute(
            "DELETE FROM sessions WHERE updated_at < ?", (cutoff,)
        ).rowcount
        conn.commit()
    if sessions:
        logger.info(
            "会话清理：删掉 %s 个会话 / %s 条消息 / %s 条记忆 / %s 张工单（早于 %s）",
            sessions, removed, memories, tickets, cutoff,
        )
    return {
        "sessions": sessions,
        "messages": removed,
        "memories": memories,
        "tickets": tickets,
    }


def stats() -> dict:
    """会话总数、消息总数、记忆条数、以及最老/最新的会话时间。CLI 概览用。

    `memories` 单列出来，是因为它和 messages 是**两条独立增长曲线**：
    messages 随对话线性长，memories 只在有人跑 `distill` 时才长。两个数字
    放在一起看，就能一眼判断"蒸馏有没有在跑"。
    """
    with _lock:
        conn = connect()
        sessions = conn.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"]
        messages = conn.execute("SELECT COUNT(*) AS n FROM messages").fetchone()["n"]
        memories = conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"]
        tickets = conn.execute(
            "SELECT COUNT(*) AS n FROM tickets WHERE status != 'closed'"
        ).fetchone()["n"]
        oldest = conn.execute("SELECT MIN(created_at) AS t FROM sessions").fetchone()["t"]
        newest = conn.execute("SELECT MAX(updated_at) AS t FROM sessions").fetchone()["t"]
    return {
        "sessions": sessions,
        "messages": messages,
        "memories": memories,
        "open_tickets": tickets,
        "oldest": oldest,
        "newest": newest,
        "ttl_days": SESSION_TTL_DAYS,
        "db_path": str(SESSION_DB_PATH),
    }
