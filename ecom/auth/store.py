"""principal 的存储：只存 token 的 SHA-256，不存明文。

【为什么是 bearer token，不是用户名密码】
这个项目有四个入口（CLI / Streamlit / HTTP / MCP），它们不是同一个信任域。
真正需要的是"谁在调用、能做什么、做过什么留痕"；一套登录页并不能回答这些，
反而会把身份绑死在唯一一个界面上。创建 principal 时打印一次 token，服务端
只保存它的哈希——和密码库同一条原则：**令牌泄漏时，库本身不是明文**。

【为什么角色是有序的，不是一堆布尔】
`admin` 能做的事包含 `agent` 能做的，`agent` 包含 `customer` 的。用
`customer < agent < admin` 一个序就够表达"至少是什么级别"，不需要为每个动作
维护一张权限表——这个规模下，权限表只会变成一张没人看得懂的矩阵。
"""
import hashlib
import secrets
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from ecom.auth import settings

# 测试会 monkeypatch 这个模块级变量，所以 connect() 里要现读，不要在导入时算死。
AUTH_DB_PATH = settings.AUTH_DB_PATH

ROLES = ("customer", "agent", "admin")
ROLE_RANK = {"customer": 1, "agent": 2, "admin": 3}

_MAX_NAME = 64
_MAX_DETAIL = 1000

SCHEMA = """
CREATE TABLE IF NOT EXISTS principals (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    role       TEXT NOT NULL CHECK (role IN ('customer', 'agent', 'admin')),
    token_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    revoked_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_principals_token ON principals(token_hash);
CREATE TABLE IF NOT EXISTS audit_log (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    at      TEXT NOT NULL,
    actor   TEXT NOT NULL,
    action  TEXT NOT NULL,
    target  TEXT NOT NULL DEFAULT '',
    detail  TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_audit_at ON audit_log(id);
"""


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _connect() -> sqlite3.Connection:
    path = Path(AUTH_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5)
    conn.row_factory = sqlite3.Row
    # 和 sessions.db 一样：WAL + busy_timeout，让 CLI 和 API 同时读写时不互相卡死。
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(SCHEMA)
    return conn


def _clean_name(name) -> str:
    value = str(name or "").strip()
    if not value or len(value) > _MAX_NAME or any(ch.isspace() for ch in value):
        raise ValueError(f"名字要去掉空白、1~{_MAX_NAME} 个字符")
    return value


def _public(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "role": row["role"],
        "created_at": row["created_at"],
        "revoked_at": row["revoked_at"],
    }


def create_principal(name: str, role: str = "customer", token: str = None) -> dict:
    """
    创建一个 principal，返回**含明文 token** 的字典。

    `token` 参数只给测试/导入用；生产路径应该让这里生成。返回值里的 token
    是它唯一一次出现的地方——调用方要负责交给对方，服务端不留。
    """
    clean_name = _clean_name(name)
    clean_role = str(role or "").strip().lower()
    if clean_role not in ROLES:
        raise ValueError(f"role 只能是 {ROLES}，收到 {role!r}")

    plain = str(token or "").strip() or secrets.token_urlsafe(settings.AUTH_TOKEN_BYTES)
    principal_id = "p_" + secrets.token_hex(8)
    now = _now()

    with closing(_connect()) as conn:
        try:
            conn.execute(
                "INSERT INTO principals (id, name, role, token_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (principal_id, clean_name, clean_role, _hash(plain), now),
            )
            conn.commit()
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"名字或 token 已存在：{clean_name}") from exc

    return {
        "id": principal_id,
        "name": clean_name,
        "role": clean_role,
        "created_at": now,
        "token": plain,
    }


def authenticate(token: str) -> dict:
    """
    用 token 换 principal；无效/已吊销返回 None。

    查的是哈希列（有索引），**不是**把明文拿来逐个比较——所以这里不存在
    "第一个字节不同就返回"的时序问题。
    """
    plain = str(token or "").strip()
    if not plain:
        return None
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT * FROM principals WHERE token_hash = ? AND revoked_at IS NULL",
            (_hash(plain),),
        ).fetchone()
    return _public(row) if row else None


def get_principal(identifier: str) -> dict:
    """按 id 或 name 查（不含哈希）。查不到返回 None。"""
    value = str(identifier or "").strip()
    if not value:
        return None
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT * FROM principals WHERE id = ? OR name = ?", (value, value)
        ).fetchone()
    return _public(row) if row else None


def list_principals() -> list:
    with closing(_connect()) as conn:
        rows = conn.execute("SELECT * FROM principals ORDER BY created_at, id").fetchall()
    return [_public(row) for row in rows]


def revoke_principal(identifier: str, actor: str = "") -> bool:
    """吊销一个 principal。返回是否真的改了一行（已吊销再吊销返回 False）。"""
    principal = get_principal(identifier)
    if principal is None or principal["revoked_at"]:
        return False
    with closing(_connect()) as conn:
        conn.execute(
            "UPDATE principals SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
            (_now(), principal["id"]),
        )
        conn.commit()
    audit(actor or "system", "principal.revoke", principal["name"], "")
    return True


def has_role(principal: dict, minimum: str) -> bool:
    """`principal` 是否至少是 `minimum` 这一级。未知角色按 0 处理（拒绝）。"""
    if not principal:
        return False
    return ROLE_RANK.get(principal.get("role"), 0) >= ROLE_RANK.get(minimum, 99)


def audit(actor: str, action: str, target: str = "", detail: str = "") -> int:
    """写一条审计。**失败不抛**：审计写不进去不该让业务操作失败，但要留日志。"""
    try:
        with closing(_connect()) as conn:
            cursor = conn.execute(
                "INSERT INTO audit_log (at, actor, action, target, detail) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    _now(),
                    str(actor or "system")[:255],
                    str(action or "")[:255],
                    str(target or "")[:255],
                    str(detail or "")[:_MAX_DETAIL],
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)
    except Exception:  # noqa: BLE001 —— 见 docstring
        import logging

        logging.getLogger(__name__).exception("写审计失败：%s / %s", actor, action)
        return 0


def recent_audit(limit: int = 50) -> list:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (int(limit),)
        ).fetchall()
    return [dict(row) for row in rows]
