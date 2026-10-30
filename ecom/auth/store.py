"""principal 的存储：token 只存 SHA-256，密码只存 scrypt 哈希。

【三种凭证，用途不同】
- **API token**：长期凭证，给脚本/服务用；创建时显示一次，可设过期天数，可轮换。
- **登录会话**：`/auth/login` 用用户名 + 密码换来的短期 token，默认 12 小时；
  它和 API token 走同一个 `authenticate()`，但过期时间短、可以单独清理。
- **密码**：只用于换登录会话；服务端存的是带盐的 scrypt 哈希，不是明文。

角色仍然只有三档、有序：`customer < agent < admin`。
"""
import hashlib
import hmac
import re
import secrets
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path

from ecom.auth import settings

# 测试会 monkeypatch 这个模块级变量，所以 connect() 里要现读，不要在导入时算死。
AUTH_DB_PATH = settings.AUTH_DB_PATH

ROLES = ("customer", "agent", "admin")
ROLE_RANK = {"customer": 1, "agent": 2, "admin": 3}

_MAX_NAME = 64
_MAX_DETAIL = 1000

# scrypt 参数：N=2^14 在 2026 年的机器上是"几十毫秒"量级，够挡住离线爆破的
# 成本，又不至于让一次登录明显卡住。参数会写进哈希串，将来调高不用迁移老密码。
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_LEN = 32

SCHEMA = """
CREATE TABLE IF NOT EXISTS principals (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    role          TEXT NOT NULL CHECK (role IN ('customer', 'agent', 'admin')),
    token_hash    TEXT NOT NULL UNIQUE,
    created_at    TEXT NOT NULL,
    revoked_at    TEXT,
    expires_at    TEXT,
    password_hash TEXT,
    customer_id   TEXT
);
CREATE INDEX IF NOT EXISTS idx_principals_token ON principals(token_hash);
CREATE TABLE IF NOT EXISTS auth_tokens (
    token_hash   TEXT PRIMARY KEY,
    principal_id TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    expires_at   TEXT NOT NULL,
    revoked_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_auth_tokens_principal ON auth_tokens(principal_id);
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


def _now() -> datetime:
    return datetime.now()


def _stamp(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _connect() -> sqlite3.Connection:
    path = Path(AUTH_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(SCHEMA)
    # 老库补列：CREATE TABLE IF NOT EXISTS 不会改已存在的表。幂等，可以每次跑。
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(principals)")}
    if "expires_at" not in columns:
        conn.execute("ALTER TABLE principals ADD COLUMN expires_at TEXT")
    if "password_hash" not in columns:
        conn.execute("ALTER TABLE principals ADD COLUMN password_hash TEXT")
    if "customer_id" not in columns:
        conn.execute("ALTER TABLE principals ADD COLUMN customer_id TEXT")
    conn.commit()
    return conn


def _clean_name(name) -> str:
    value = str(name or "").strip()
    if not value or len(value) > _MAX_NAME or any(ch.isspace() for ch in value):
        raise ValueError(f"名字要去掉空白、1~{_MAX_NAME} 个字符")
    return value


def _clean_customer(value) -> str:
    """客户号：`C` + 3 位以上数字。空值表示"没绑定客户身份"。"""
    text = str(value or "").strip().upper()
    if not text:
        return ""
    if not re.fullmatch(r"C\d{3,}", text):
        raise ValueError("客户号应形如 C20755")
    return text


def _public(row: sqlite3.Row) -> dict:
    keys = set(row.keys())
    return {
        "id": row["id"],
        "name": row["name"],
        "role": row["role"],
        "created_at": row["created_at"],
        "revoked_at": row["revoked_at"] if "revoked_at" in keys else None,
        "expires_at": row["expires_at"] if "expires_at" in keys else None,
        "customer_id": row["customer_id"] if "customer_id" in keys else None,
    }


def _expiry(days: float = 0) -> str:
    return _stamp(_now() + timedelta(days=float(days))) if days else None


def _alive(expires_at: str) -> bool:
    return not expires_at or expires_at > _stamp(_now())


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_LEN,
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt_hex, digest_hex = str(stored).split("$")
        if scheme != "scrypt":
            return False
        digest = hashlib.scrypt(
            str(password).encode("utf-8"),
            salt=bytes.fromhex(salt_hex),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(bytes.fromhex(digest_hex)),
        )
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (TypeError, ValueError):
        return False


def create_principal(
    name: str,
    role: str = "customer",
    token: str = None,
    expires_days: float = 0,
    customer_id: str = "",
) -> dict:
    """
    创建一个 principal，返回**含明文 token** 的字典。

    `expires_days=0` 表示不过期（给本地/长期服务 token）；大于 0 就是到期时间。
    token 只在返回值里出现一次，服务端只存 SHA-256。
    `customer_id` 非空时，这个 principal 的订单查询会**锁定到这个客户**，
    不再从用户的话里猜客户号。
    """
    clean_name = _clean_name(name)
    clean_role = str(role or "").strip().lower()
    if clean_role not in ROLES:
        raise ValueError(f"role 只能是 {ROLES}，收到 {role!r}")
    clean_customer = _clean_customer(customer_id)

    plain = str(token or "").strip() or secrets.token_urlsafe(settings.AUTH_TOKEN_BYTES)
    principal_id = "p_" + secrets.token_hex(8)
    now = _stamp(_now())
    expires_at = _expiry(expires_days)

    with closing(_connect()) as conn:
        try:
            conn.execute(
                "INSERT INTO principals "
                "(id, name, role, token_hash, created_at, expires_at, customer_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    principal_id,
                    clean_name,
                    clean_role,
                    _hash(plain),
                    now,
                    expires_at,
                    clean_customer or None,
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"名字或 token 已存在：{clean_name}") from exc

    return {
        "id": principal_id,
        "name": clean_name,
        "role": clean_role,
        "created_at": now,
        "expires_at": expires_at,
        "customer_id": clean_customer or None,
        "token": plain,
    }


def set_customer(identifier: str, customer_id: str) -> None:
    """把 principal 绑定/解绑到一个客户号。绑定之后订单查询不再靠猜。"""
    principal = get_principal(identifier)
    if principal is None:
        raise ValueError(f"没有这个 principal：{identifier}")
    clean = _clean_customer(customer_id)
    with closing(_connect()) as conn:
        conn.execute(
            "UPDATE principals SET customer_id = ? WHERE id = ?",
            (clean or None, principal["id"]),
        )
        conn.commit()
    audit(principal["name"], "principal.set_customer", principal["name"], clean or "unbound")


def authenticate(token: str) -> dict:
    """
    用 token 换 principal；无效/已吊销/已过期返回 None。

    先查长期 API token，再查登录会话 token。两者都是哈希查找，不存在
    "第一个字节不同就返回"的时序问题。
    """
    plain = str(token or "").strip()
    if not plain:
        return None
    digest = _hash(plain)
    now = _stamp(_now())
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT * FROM principals "
            "WHERE token_hash = ? AND revoked_at IS NULL "
            "AND (expires_at IS NULL OR expires_at > ?)",
            (digest, now),
        ).fetchone()
        if row:
            return _public(row)

        row = conn.execute(
            "SELECT p.* FROM auth_tokens t JOIN principals p ON p.id = t.principal_id "
            "WHERE t.token_hash = ? AND t.revoked_at IS NULL AND t.expires_at > ? "
            "AND p.revoked_at IS NULL "
            "AND (p.expires_at IS NULL OR p.expires_at > ?)",
            (digest, now, now),
        ).fetchone()
    return _public(row) if row else None


def rotate_principal(identifier: str, expires_days: float = 0) -> dict:
    """给 principal 换一枚新 API token；旧 token 和所有登录会话立即失效。"""
    principal = get_principal(identifier)
    if principal is None or principal["revoked_at"]:
        raise ValueError(f"没有可轮换的 principal：{identifier}")
    plain = secrets.token_urlsafe(settings.AUTH_TOKEN_BYTES)
    now = _stamp(_now())
    expires_at = _expiry(expires_days)
    with closing(_connect()) as conn:
        conn.execute(
            "UPDATE principals SET token_hash = ?, created_at = ?, expires_at = ? "
            "WHERE id = ?",
            (_hash(plain), now, expires_at, principal["id"]),
        )
        conn.execute(
            "UPDATE auth_tokens SET revoked_at = ? WHERE principal_id = ? AND revoked_at IS NULL",
            (now, principal["id"]),
        )
        conn.commit()
    audit(principal["name"], "principal.rotate", principal["name"], "旧 token 已失效")
    return {**principal, "token": plain, "created_at": now, "expires_at": expires_at}


def set_password(identifier: str, password: str) -> None:
    """设置/重置登录密码。密码不返回、不落日志。"""
    principal = get_principal(identifier)
    if principal is None:
        raise ValueError(f"没有这个 principal：{identifier}")
    if len(str(password or "")) < 8:
        raise ValueError("密码至少 8 个字符")
    with closing(_connect()) as conn:
        conn.execute(
            "UPDATE principals SET password_hash = ? WHERE id = ?",
            (_hash_password(password), principal["id"]),
        )
        conn.commit()
    audit(principal["name"], "principal.set_password", principal["name"], "")


def verify_password(name: str, password: str) -> dict:
    """用户名 + 密码 → principal；任何一步不对都返回 None（不区分原因）。"""
    value = str(name or "").strip()
    if not value or not password:
        return None
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT * FROM principals WHERE name = ? AND revoked_at IS NULL", (value,)
        ).fetchone()
    if row is None or not row["password_hash"]:
        return None
    if not _alive(row["expires_at"]):
        return None
    return _public(row) if _verify_password(password, row["password_hash"]) else None


def issue_session(principal: dict, ttl_hours: float = None) -> dict:
    """登录成功后发一枚短期 token。默认 12 小时，返回明文一次。"""
    hours = settings.AUTH_SESSION_TTL_HOURS if ttl_hours is None else float(ttl_hours)
    plain = secrets.token_urlsafe(settings.AUTH_TOKEN_BYTES)
    now = _stamp(_now())
    expires_at = _stamp(_now() + timedelta(hours=hours))
    with closing(_connect()) as conn:
        conn.execute(
            "INSERT INTO auth_tokens (token_hash, principal_id, created_at, expires_at) "
            "VALUES (?, ?, ?, ?)",
            (_hash(plain), principal["id"], now, expires_at),
        )
        conn.commit()
    audit(principal["name"], "auth.login", principal["name"], f"expires {expires_at}")
    return {"token": plain, "expires_at": expires_at}


def revoke_token(token: str) -> bool:
    """
    吊销一枚**登录会话** token，返回是否真的吊销了一条。

    长期 API token（`principals.token_hash`）不在这里处理：它是账号的根凭证，
    吊销它等于把该账号所有设备一起踢下线，应该走 `rotate_principal` 显式轮换。
    登出只结束当前这枚短期 token。
    """
    plain = str(token or "").strip()
    if not plain:
        return False
    now = _stamp(_now())
    with closing(_connect()) as conn:
        cursor = conn.execute(
            "UPDATE auth_tokens SET revoked_at = ? WHERE token_hash = ? AND revoked_at IS NULL",
            (now, _hash(plain)),
        )
        conn.commit()
        return cursor.rowcount > 0


def purge_sessions() -> int:
    """删掉过期/已吊销的登录会话，返回删除条数。启动时和 CLI 都可以调。"""
    now = _stamp(_now())
    with closing(_connect()) as conn:
        cursor = conn.execute(
            "DELETE FROM auth_tokens WHERE expires_at <= ? OR revoked_at IS NOT NULL", (now,)
        )
        conn.commit()
        return cursor.rowcount


def get_principal(identifier: str) -> dict:
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
    """吊销一个 principal，并把它名下所有登录会话一起吊销。"""
    principal = get_principal(identifier)
    if principal is None or principal["revoked_at"]:
        return False
    now = _stamp(_now())
    with closing(_connect()) as conn:
        conn.execute(
            "UPDATE principals SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
            (now, principal["id"]),
        )
        conn.execute(
            "UPDATE auth_tokens SET revoked_at = ? WHERE principal_id = ? AND revoked_at IS NULL",
            (now, principal["id"]),
        )
        conn.commit()
    audit(actor or "system", "principal.revoke", principal["name"], "")
    return True


def has_role(principal: dict, minimum: str) -> bool:
    if not principal:
        return False
    return ROLE_RANK.get(principal.get("role"), 0) >= ROLE_RANK.get(minimum, 99)


def audit(actor: str, action: str, target: str = "", detail: str = "") -> int:
    """写一条审计。失败不抛：审计写不进去不该让业务操作失败，但要留日志。"""
    try:
        with closing(_connect()) as conn:
            cursor = conn.execute(
                "INSERT INTO audit_log (at, actor, action, target, detail) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    _stamp(_now()),
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
