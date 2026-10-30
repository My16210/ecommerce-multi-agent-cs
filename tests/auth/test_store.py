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
