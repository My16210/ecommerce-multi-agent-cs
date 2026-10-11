"""
待审核队列（SQLite）
功能：存放「模型判定值得沉淀的提问」，等人来审

为什么中间要有这一层
--------------------
模型生成的答案不能直接进向量库。它一旦入库，下一轮检索就会把它当成
"权威资料"捞出来喂给模型，错误会被放大并固化成看起来不容置疑的知识——
也就是**知识库自污染**。所以中间必须隔一个队列 + 一次人工批准。

本模块只管存取，不做任何向量库操作。向量库的唯一写入口始终是 ingest.py。
"""
import hashlib
import json
import re
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from ecom.knowledge.settings import REVIEW_DB_PATH
from ecom.knowledge.utils.logger import setup_logger

logger = setup_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS question_review (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint   TEXT    NOT NULL UNIQUE,
    question      TEXT    NOT NULL,
    answer        TEXT,
    worthy        INTEGER NOT NULL DEFAULT 0,
    category      TEXT,
    score         REAL,
    reason        TEXT,
    kb_hits       INTEGER,
    kb_top_score  REAL,
    web_hits      INTEGER,
    sources_json  TEXT,
    ask_count     INTEGER NOT NULL DEFAULT 1,
    status        TEXT    NOT NULL DEFAULT 'pending',
    doc_path      TEXT,
    created_at    TEXT,
    updated_at    TEXT,
    reviewed_at   TEXT,
    review_note   TEXT
);
CREATE INDEX IF NOT EXISTS idx_status ON question_review(status);
"""

# 一条语句完成"新建或更新"。
#
# 原来写的是 SELECT-then-INSERT：两条语句之间有窗口，并发或重入时
# 会撞上 fingerprint 的 UNIQUE 约束直接抛 IntegrityError。upsert 由
# SQLite 在单条语句内完成检查与写入，没有这个窗口。
#
# DO UPDATE 里**刻意不含** status / doc_path / review_note / created_at /
# question——人工审过的结论不能被下一次提问推翻，created_at 是首次出现时间。
# 这条语义必须原样保住：它正是"人工审核"这一环有意义的前提。
UPSERT_SQL = """
INSERT INTO question_review (
    fingerprint, question, answer, worthy, category, score, reason,
    kb_hits, kb_top_score, web_hits, sources_json, created_at, updated_at
) VALUES (
    :fingerprint, :question, :answer, :worthy, :category, :score, :reason,
    :kb_hits, :kb_top_score, :web_hits, :sources_json, :created_at, :updated_at
)
ON CONFLICT(fingerprint) DO UPDATE SET
    answer       = excluded.answer,
    worthy       = excluded.worthy,
    category     = excluded.category,
    score        = excluded.score,
    reason       = excluded.reason,
    kb_hits      = excluded.kb_hits,
    kb_top_score = excluded.kb_top_score,
    web_hits     = excluded.web_hits,
    sources_json = excluded.sources_json,
    ask_count    = ask_count + 1,
    updated_at   = excluded.updated_at
RETURNING id, ask_count
"""

# 审核状态机：pending → approved | rejected（人工定了就不再回退）
STATUSES = ("pending", "approved", "rejected")

_conn = None

# 【为什么必须有一把锁】下面 `connect()` 用 `check_same_thread=False` 让**进程内
# 共用同一个连接**，但 sqlite3 的连接对象本身不是为并发设计的。而这里的调用方
# 是并发的：`admin.py`（人在审）和 app 那条路（模型在写）可能同时进来，Streamlit
# 每个会话又是自己的线程。
#
# 只开 `check_same_thread=False` 却不配锁，等于把"多线程共用一个连接"这个口子
# 主动开着——那样得到的不是"能并发"，是"偶尔炸"。
#
# 读也一起锁：共用的是同一个连接对象，读写不是两条独立的路。
# `ecom/orchestrator/session.py` 是同一类存储、同样的形状，那边一直是这么做的；
# 这里原来漏了。
_lock = threading.Lock()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def connect() -> sqlite3.Connection:
    """
    打开队列库，**进程内复用同一个连接**，建表脚本只在首次打开时跑一次。

    所以下面那些函数里不能再有 `finally: conn.close()`——那会把缓存连接
    关掉，下次调用拿到一个已关闭的连接。
    """
    global _conn
    if _conn is None:
        Path(REVIEW_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False：和 main.py 里 SqliteSaver 的用法一致。
        # 它只是关掉 sqlite 的"跨线程用同一连接就报错"这道自检，
        # **并发安全由上面那把 `_lock` 负责**，两者缺一不可。
        conn = sqlite3.connect(REVIEW_DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # WAL：让"人正在审"和"服务正在写"不互相阻塞。默认 journal 模式下
        # 一个写者会挡住所有读者。
        conn.execute("PRAGMA journal_mode=WAL")
        # 万一还是撞上锁（比如宿主上另一个进程正持着），等 5 秒再报错，
        # 而不是立刻抛 "database is locked"。
        conn.execute("PRAGMA busy_timeout=5000")
        conn.executescript(SCHEMA)
        conn.commit()
        _conn = conn
    return _conn


def reset_connection() -> None:
    """
    丢掉缓存的连接，下次 connect() 重新按 REVIEW_DB_PATH 打开。

    给测试用：每个用例都要把队列指到自己的临时库文件，不清缓存就会串库。
    """
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None


def fingerprint(question: str) -> str:
    """
    问题的指纹：去掉所有空白、转小写后取 sha1。

    同一个问题问十遍只留一行（ask_count 累加），
    否则队列会被重复提问刷屏。
    """
    normalized = re.sub(r"\s+", "", question or "").strip().lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def enqueue(question: str, result: dict, verdict: dict) -> tuple:
    """
    把一次提问的完整现场写进队列。

    返回 (row_id, is_new)。重复提问只更新、不新增。
    """
    contexts = result.get("contexts") or []
    # kb_top_score 存的是"最高相似度"，人审核时要看这个数判断知识库答没答好。
    # 混合检索开启后 score 已经是 RRF 融合分（量级 1/61），存进去会让人以为
    # 检索质量暴跌，所以优先取向量的原始余弦分（见 judge._format_signals 同样的处理）。
    scores = [
        c.get("vector_score", c.get("score", 0.0)) for c in contexts if isinstance(c, dict)
    ]
    now = _now()

    with _lock:
        conn = connect()
        row = conn.execute(
            UPSERT_SQL,
            {
                "fingerprint": fingerprint(question),
                "question": question,
                "answer": result.get("answer") or "",
                "worthy": 1 if verdict.get("worthy") else 0,
                "category": verdict.get("category"),
                "score": verdict.get("score"),
                "reason": verdict.get("reason"),
                "kb_hits": len(contexts),
                "kb_top_score": max(scores) if scores else 0.0,
                "web_hits": len(result.get("web_contexts") or []),
                "sources_json": json.dumps(result.get("sources") or [], ensure_ascii=False),
                "created_at": now,
                "updated_at": now,
            },
        ).fetchone()
        conn.commit()

    # 走冲突分支时 ask_count 一定已被 +1 成 >= 2，所以只有全新插入才是 1。
    # 这样不用再补一条 SELECT 就能判断是不是新行。
    return row["id"], row["ask_count"] == 1


def list_items(worthy_only: bool = True, status: str = "pending", limit: int = 50) -> list:
    """列出队列。worthy_only=False 时连模型判为无价值的也一起返回，用来看判定准不准"""
    sql = "SELECT * FROM question_review WHERE 1 = 1"
    params = []
    if status:
        sql += " AND status = ?"
        params.append(status)
    if worthy_only:
        sql += " AND worthy = 1"
    sql += " ORDER BY score DESC, id DESC LIMIT ?"
    params.append(limit)

    with _lock:
        rows = connect().execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def get_item(item_id: int) -> dict:
    with _lock:
        row = connect().execute(
            "SELECT * FROM question_review WHERE id = ?", (item_id,)
        ).fetchone()
    return dict(row) if row else None


def set_status(item_id: int, status: str, note: str = None, doc_path: str = None) -> bool:
    """改状态。返回是否命中了一行。doc_path 传 None 表示不动原值"""
    if status not in STATUSES:
        raise ValueError(f"未知状态 {status!r}，可选：{STATUSES}")

    with _lock:
        conn = connect()
        cursor = conn.execute(
            """
            UPDATE question_review SET
                status = ?, reviewed_at = ?, review_note = COALESCE(?, review_note),
                doc_path = COALESCE(?, doc_path)
            WHERE id = ?
            """,
            (status, _now(), note, doc_path, item_id),
        )
        conn.commit()
    return cursor.rowcount > 0


def stats() -> dict:
    """各状态的条数，供 CLI 概览用"""
    with _lock:
        rows = connect().execute(
            "SELECT status, COUNT(*) AS n FROM question_review GROUP BY status"
        ).fetchall()
    return {row["status"]: row["n"] for row in rows}


if __name__ == "__main__":
    total = connect().execute("SELECT COUNT(*) AS n FROM question_review").fetchone()["n"]
    print(f"待审核队列：{REVIEW_DB_PATH}")
    print(f"共 {total} 条，状态分布：{stats()}")
