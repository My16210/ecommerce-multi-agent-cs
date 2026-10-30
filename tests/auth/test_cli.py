"""CLI 的冒烟测试：创建 -> 认证 -> 列表 -> 吊销。"""
import json

from ecom.auth import cli, store


def test_cli_create_prints_token_once(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(store, "AUTH_DB_PATH", tmp_path / "auth.db")
    assert cli.main(["create", "--name", "alice", "--role", "admin"]) == 0
    out = capsys.readouterr().out
    assert "TOKEN" in out
    token = out.strip().splitlines()[-1]
    principal = store.authenticate(token)
    assert principal["name"] == "alice"
    assert principal["role"] == "admin"


def test_cli_list_and_revoke(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(store, "AUTH_DB_PATH", tmp_path / "auth.db")
    created = store.create_principal("bob", "customer")

    assert cli.main(["list"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["name"] == "bob"
    assert "token" not in rows[0]

    assert cli.main(["revoke", "bob"]) == 0
    assert store.authenticate(created["token"]) is None
    assert cli.main(["revoke", "bob"]) == 1
