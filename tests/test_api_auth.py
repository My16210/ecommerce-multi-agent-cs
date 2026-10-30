"""身份层接入 API 的测试：token、角色、会话归属。

默认的 `AUTH_ENABLED=0` 行为由 `tests/test_api.py` 守着；这里只验打开之后
多出来的那三条：**没 token 401、不是自己的会话 404、trace 要 admin**。
"""
import pytest

pytest.importorskip("fastapi", reason="需要 uv sync --extra api")
pytest.importorskip("httpx", reason="TestClient 需要 httpx（dev 依赖组）")

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
from ecom.auth import settings as auth_settings  # noqa: E402
from ecom.auth import store as auth_store  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api.session, "SESSION_DB_PATH", str(tmp_path / "sessions.db"))
    monkeypatch.setattr(api.trace, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(auth_store, "AUTH_DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(auth_settings, "AUTH_ENABLED", True)
    monkeypatch.setattr(api, "API_KEY", "")
    api.session.reset_connection()

    monkeypatch.setattr(
        api,
        "answer",
        lambda question, history: {
            "request_id": "req-auth",
            "answer": f"答复：{question}",
            "sources": [],
            "decision": "answer",
            "plan": ["knowledge"],
            "tokens": {"in": 1, "out": 1},
        },
    )
    monkeypatch.setattr(
        api,
        "answer_stream",
        lambda question, history: iter(
            [
                {"type": "start", "request_id": "req-auth"},
                {"type": "delta", "text": "答复"},
                {
                    "type": "final",
                    "state": {
                        "request_id": "req-auth",
                        "answer": "答复",
                        "sources": [],
                        "decision": "answer",
                        "plan": ["knowledge"],
                        "tokens": {},
                    },
                    "elapsed_ms": 1,
                },
            ]
        ),
    )

    alice = auth_store.create_principal("alice", "admin")
    bob = auth_store.create_principal("bob", "customer")

    with TestClient(api.app) as c:
        c.alice = alice
        c.bob = bob
        yield c

    api.session.reset_connection()


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_chat_requires_a_token(client):
    assert client.post("/chat", json={"question": "hi"}).status_code == 401
    assert client.get("/trace").status_code == 401


def test_chat_and_stream_work_with_a_token(client):
    r = client.post("/chat", json={"question": "hi"}, headers=_auth(client.bob["token"]))
    assert r.status_code == 200
    assert r.json()["answer"] == "答复：hi"

    r = client.post(
        "/chat/stream", json={"question": "hi"}, headers=_auth(client.bob["token"])
    )
    assert r.status_code == 200
    assert "event-stream" in r.headers["content-type"]


def test_a_session_belongs_to_its_principal(client):
    r = client.post("/chat", json={"question": "hi"}, headers=_auth(client.bob["token"]))
    sid = r.json()["session_id"]
    assert api.session.get_session(sid)["owner_id"] == client.bob["id"]

    # 另一个 principal 拿同一个 session_id：按 404 处理，不暴露"存在但没权限"
    r = client.post(
        "/chat",
        json={"question": "偷看", "session_id": sid},
        headers=_auth(client.alice["token"]),
    )
    assert r.status_code == 404


def test_trace_requires_admin(client):
    assert client.get("/trace", headers=_auth(client.bob["token"])).status_code == 403
    # admin 没有记录时是 404，不是 403
    assert client.get("/trace", headers=_auth(client.alice["token"])).status_code == 404


def test_health_reports_auth_mode(client):
    body = client.get("/health").json()
    assert body["auth"] is True
