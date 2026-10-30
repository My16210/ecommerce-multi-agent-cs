"""principal 存储的契约测试：创建、认证、吊销、角色、审计。"""
import hashlib
import sqlite3

import pytest

from ecom.auth import store


@pytest.fixture(autouse=True)
def auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "AUTH_DB_PATH", tmp_path / "auth.db")


def test_create_authenticate_and_revoke():
    created = store.create_principal("alice", "admin")
    assert created["token"]
    assert created["role"] == "admin"

    principal = store.authenticate(created["token"])
    assert principal["name"] == "alice"
    assert principal["role"] == "admin"
    assert "token" not in principal
    assert store.has_role(principal, "admin")
    assert store.has_role(principal, "customer")

    assert store.revoke_principal("alice")
    assert store.authenticate(created["token"]) is None
    assert store.revoke_principal("alice") is False


def test_token_is_only_stored_as_a_hash():
    created = store.create_principal("bob")
    with sqlite3.connect(store.AUTH_DB_PATH) as conn:
        row = conn.execute(
            "SELECT token_hash FROM principals WHERE name = 'bob'"
        ).fetchone()
    assert row[0] == hashlib.sha256(created["token"].encode()).hexdigest()
    assert created["token"] not in row[0]


def test_duplicate_name_and_bad_role_are_rejected():
    store.create_principal("carol")
    with pytest.raises(ValueError):
        store.create_principal("carol")
    with pytest.raises(ValueError):
        store.create_principal("dave", "root")
    with pytest.raises(ValueError):
        store.create_principal("  ", "customer")


def test_list_never_exposes_tokens():
    store.create_principal("eve", "agent")
    rows = store.list_principals()
    assert rows[0]["name"] == "eve"
    assert all("token" not in row and "token_hash" not in row for row in rows)


def test_role_is_ordered():
    customer = {"role": "customer"}
    assert not store.has_role(customer, "agent")
    assert store.has_role(customer, "customer")
    assert not store.has_role({}, "customer")


def test_audit_is_written_and_readable():
    store.audit("alice", "return.approve", "R0000000001", "ok")
    rows = store.recent_audit()
    assert rows[0]["actor"] == "alice"
    assert rows[0]["action"] == "return.approve"
    assert rows[0]["target"] == "R0000000001"


def test_password_login_issues_short_lived_session():
    created = store.create_principal("frank", "agent")
    store.set_password("frank", "correct horse battery")

    assert store.verify_password("frank", "wrong") is None
    principal = store.verify_password("frank", "correct horse battery")
    assert principal["role"] == "agent"

    issued = store.issue_session(principal, ttl_hours=1)
    assert store.authenticate(issued["token"])["name"] == "frank"

    # 0 小时 = 立即过期：登录会话必须是短期凭证，不能退化成永久的
    expired = store.issue_session(principal, ttl_hours=0)
    assert store.authenticate(expired["token"]) is None
    assert created["token"]  # 长期 API token 仍然有效
    assert store.authenticate(created["token"])["name"] == "frank"


def test_expired_api_token_is_rejected():
    created = store.create_principal("grace", expires_days=-1)
    assert store.authenticate(created["token"]) is None


def test_rotate_invalidates_the_old_token_and_sessions():
    created = store.create_principal("heidi", "admin")
    store.set_password("heidi", "long enough password")
    principal = store.verify_password("heidi", "long enough password")
    session = store.issue_session(principal, ttl_hours=1)

    rotated = store.rotate_principal("heidi")
    assert rotated["token"] != created["token"]
    assert store.authenticate(created["token"]) is None
    assert store.authenticate(session["token"]) is None
    assert store.authenticate(rotated["token"])["role"] == "admin"


def test_revoke_also_kills_sessions():
    created = store.create_principal("ivan")
    store.set_password("ivan", "long enough password")
    principal = store.verify_password("ivan", "long enough password")
    session = store.issue_session(principal, ttl_hours=1)

    assert store.revoke_principal("ivan")
    assert store.authenticate(created["token"]) is None
    assert store.authenticate(session["token"]) is None


def test_password_is_hashed_with_scrypt():
    store.create_principal("judy")
    store.set_password("judy", "a-very-secret-password")
    with sqlite3.connect(store.AUTH_DB_PATH) as conn:
        stored = conn.execute(
            "SELECT password_hash FROM principals WHERE name = 'judy'"
        ).fetchone()[0]
    assert stored.startswith("scrypt$")
    assert "a-very-secret-password" not in stored


def test_customer_binding():
    created = store.create_principal("kim", "customer", customer_id="C20755")
    assert store.authenticate(created["token"])["customer_id"] == "C20755"

    store.set_customer("kim", "")
    assert store.authenticate(created["token"])["customer_id"] is None

    with pytest.raises(ValueError):
        store.set_customer("kim", "not-a-customer")


def test_set_password_revokes_existing_sessions_but_not_the_api_token():
    """改密码必须让旧登录会话立刻失效。

    否则密码泄露后，受害者改了密码，已经拿到会话 token 的人还能继续用到
    12 小时过期。长期 API token 是另一条根凭证，由 rotate 显式轮换。
    """
    created = store.create_principal("kate", "agent")
    store.set_password("kate", "first long password")
    principal = store.verify_password("kate", "first long password")
    old_session = store.issue_session(principal, ttl_hours=1)

    store.set_password("kate", "second long password")

    assert store.authenticate(old_session["token"]) is None
    assert store.authenticate(created["token"])["name"] == "kate"
    assert store.verify_password("kate", "first long password") is None
    assert store.verify_password("kate", "second long password")["name"] == "kate"


def test_unknown_expired_and_passwordless_users_still_pay_the_hash_cost(monkeypatch):
    """用户名枚举不能靠耗时做：不存在 / 过期 / 没密码也要跑一次 scrypt。

    这里不测墙钟（CI 上不稳），只钉住"确实调用了 `_verify_password` 且传入
    的是固定假哈希"。任何人把提前 return 加回来都会让这条变红。
    """
    calls = []
    real = store._verify_password

    def spy(password, stored):  # noqa: ANN001
        calls.append((password, stored))
        return real(password, stored)

    monkeypatch.setattr(store, "_verify_password", spy)

    assert store.verify_password("nobody", "whatever") is None
    assert calls[-1] == ("whatever", store._DUMMY_PASSWORD_HASH)

    store.create_principal("leo", expires_days=-1)
    store.set_password("leo", "expired but set")
    assert store.verify_password("leo", "expired but set") is None
    assert calls[-1] == ("expired but set", store._DUMMY_PASSWORD_HASH)

    store.create_principal("mia")
    assert store.verify_password("mia", "no password set") is None
    assert calls[-1] == ("no password set", store._DUMMY_PASSWORD_HASH)

    assert store.verify_password("", "") is None
    assert calls[-1] == ("", store._DUMMY_PASSWORD_HASH)


def test_purge_audit_removes_only_rows_past_retention():
    store.audit("alice", "old.event", "R1", "")
    store.audit("alice", "new.event", "R2", "")
    with sqlite3.connect(store.AUTH_DB_PATH) as conn:
        conn.execute(
            "UPDATE audit_log SET at = '2000-01-01 00:00:00' WHERE action = 'old.event'"
        )
        conn.commit()

    assert store.purge_audit(older_than_days=30) == 1
    actions = [row["action"] for row in store.recent_audit()]
    assert actions == ["new.event"]

    # <= 0 表示关闭自动清理，不能把审计全删掉。
    assert store.purge_audit(older_than_days=0) == 0
    assert [row["action"] for row in store.recent_audit()] == ["new.event"]
