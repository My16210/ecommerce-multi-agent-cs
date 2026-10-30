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
    carol = auth_store.create_principal("carol", "agent")

    with TestClient(api.app) as c:
        c.alice = alice
        c.bob = bob
        c.carol = carol
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


def test_history_lists_only_my_sessions_with_a_title(client):
    """登录后的侧边栏数据：只有自己的会话，标题取第一条提问。

    这两条都要钉住：漏了归属过滤就是把别人的对话列给我；标题取错（比如取了
    最后一条）不会报错，只会让侧边栏显示成别的问题。
    """
    first = client.post(
        "/chat", json={"question": "订单 52644 到哪了"}, headers=_auth(client.bob["token"])
    ).json()
    client.post(
        "/chat",
        json={"question": "那退货呢", "session_id": first["session_id"]},
        headers=_auth(client.bob["token"]),
    )

    body = client.get("/sessions", headers=_auth(client.bob["token"])).json()
    assert len(body["sessions"]) == 1
    item = body["sessions"][0]
    assert item["session_id"] == first["session_id"]
    assert item["title"] == "订单 52644 到哪了"
    assert item["messages"] == 4

    # alice 是另一个 principal：列表里不能出现 bob 的会话。
    assert client.get("/sessions", headers=_auth(client.alice["token"])).json()["sessions"] == []


def test_history_detail_is_owner_only(client):
    sid = client.post(
        "/chat", json={"question": "订单 52644 到哪了"}, headers=_auth(client.bob["token"])
    ).json()["session_id"]

    r = client.get(f"/sessions/{sid}", headers=_auth(client.bob["token"]))
    assert r.status_code == 200
    body = r.json()
    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]
    assert body["messages"][0]["content"] == "订单 52644 到哪了"

    # 不存在和不属于自己都是 404，不暴露"存在但没权限"。
    assert client.get(f"/sessions/{sid}", headers=_auth(client.alice["token"])).status_code == 404
    assert (
        client.get("/sessions/not-a-real-session", headers=_auth(client.bob["token"])).status_code
        == 404
    )


def test_history_delete_is_owner_only_and_removes_the_session(client):
    sid = client.post(
        "/chat", json={"question": "hi"}, headers=_auth(client.bob["token"])
    ).json()["session_id"]

    assert (
        client.delete(f"/sessions/{sid}", headers=_auth(client.alice["token"])).status_code == 404
    )
    gone = client.delete(f"/sessions/{sid}", headers=_auth(client.bob["token"]))
    assert gone.status_code == 200
    assert gone.json()["deleted"] == 2
    assert (
        client.get(f"/sessions/{sid}", headers=_auth(client.bob["token"])).status_code == 404
    )
    assert client.get("/sessions", headers=_auth(client.bob["token"])).json()["sessions"] == []


def test_history_requires_a_token(client):
    assert client.get("/sessions").status_code == 401
    assert client.get("/sessions/whatever").status_code == 401
    assert client.delete("/sessions/whatever").status_code == 401


def test_trace_requires_admin(client):
    assert client.get("/trace", headers=_auth(client.bob["token"])).status_code == 403
    # admin 没有记录时是 404，不是 403
    assert client.get("/trace", headers=_auth(client.alice["token"])).status_code == 404


def test_health_reports_auth_mode(client):
    body = client.get("/health").json()
    assert body["auth"] is True


def test_login_issues_a_token_that_can_chat(client):
    auth_store.set_password("bob", "a-long-enough-password")

    r = client.post("/auth/login", json={"username": "bob", "password": "wrong"})
    assert r.status_code == 401

    r = client.post(
        "/auth/login", json={"username": "bob", "password": "a-long-enough-password"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["role"] == "customer"
    assert body["expires_at"]

    r = client.post(
        "/chat",
        json={"question": "hi"},
        headers={"Authorization": f"Bearer {body['token']}"},
    )
    assert r.status_code == 200


def test_customer_id_is_passed_to_the_orchestrator(client, monkeypatch):
    """登录 principal 绑定客户号后，问答链路必须收到它——否则订单层还会猜客户。"""
    auth_store.set_customer("bob", "C20755")
    seen = {}

    def fake_answer(question, history, customer_id=""):  # noqa: ANN001
        seen["customer_id"] = customer_id
        return {
            "request_id": "req-customer",
            "answer": "ok",
            "sources": [],
            "decision": "answer",
            "plan": ["order"],
            "tokens": {},
        }

    monkeypatch.setattr(api, "answer", fake_answer)
    r = client.post(
        "/chat", json={"question": "我的订单"}, headers=_auth(client.bob["token"])
    )
    assert r.status_code == 200
    assert seen["customer_id"] == "C20755"


# ——— 人工工单 ———

def _human_answer(monkeypatch):
    """把编排层换成"什么都没查出来"：这正是唯一该转人工的决策。"""
    monkeypatch.setattr(
        api,
        "answer",
        lambda question, history: {
            "request_id": "req-human",
            "answer": "好的，需要人工客服。",
            "sources": [],
            "decision": "human",
            "decision_reason": "没有任何可转达的信息",
            "plan": [],
            "tokens": {},
        },
    )


def test_human_decision_opens_a_ticket_and_writes_the_number_into_history(client, monkeypatch):
    _human_answer(monkeypatch)

    body = client.post(
        "/chat", json={"question": "这件事你能处理吗"}, headers=_auth(client.bob["token"])
    ).json()
    ticket = body["handoff"]
    sid = body["session_id"]

    assert ticket["status"] == "open"
    assert ticket["ticket_id"] in body["answer"]

    # 工单号跟着答复一起落盘：用户重新打开历史时还能看到它。
    detail = client.get(f"/sessions/{sid}", headers=_auth(client.bob["token"])).json()
    assert detail["messages"][-1]["content"] == body["answer"]

    # 别人的账号读不到这张工单
    assert (
        client.get(f"/sessions/{sid}/ticket", headers=_auth(client.alice["token"])).status_code
        == 404
    )
    own = client.get(f"/sessions/{sid}/ticket", headers=_auth(client.bob["token"])).json()
    assert own["ticket"]["ticket_id"] == ticket["ticket_id"]


def test_stream_final_carries_the_handoff(client, monkeypatch):
    monkeypatch.setattr(
        api,
        "answer_stream",
        lambda question, history: iter(
            [
                {"type": "start", "request_id": "req-human"},
                {"type": "delta", "text": "好的，需要人工客服。"},
                {
                    "type": "final",
                    "state": {
                        "request_id": "req-human",
                        "answer": "好的，需要人工客服。",
                        "sources": [],
                        "decision": "human",
                        "decision_reason": "没有任何可转达的信息",
                        "plan": [],
                        "tokens": {},
                    },
                    "elapsed_ms": 1,
                },
            ]
        ),
    )

    r = client.post(
        "/chat/stream", json={"question": "转人工"}, headers=_auth(client.bob["token"])
    )

    assert r.status_code == 200
    assert '"handoff"' in r.text
    assert '"ticket_id"' in r.text


def test_ticket_queue_requires_an_agent(client, monkeypatch):
    _human_answer(monkeypatch)
    client.post("/chat", json={"question": "转人工"}, headers=_auth(client.bob["token"]))

    assert client.get("/tickets").status_code == 401
    assert client.get("/tickets", headers=_auth(client.bob["token"])).status_code == 403

    queue = client.get("/tickets", headers=_auth(client.carol["token"]))
    assert queue.status_code == 200
    assert len(queue.json()["tickets"]) == 1


def test_agent_claims_replies_and_closes_and_the_customer_sees_it(client, monkeypatch):
    _human_answer(monkeypatch)
    body = client.post(
        "/chat", json={"question": "订单 52644 能退吗"}, headers=_auth(client.bob["token"])
    ).json()
    sid, tid = body["session_id"], body["handoff"]["ticket_id"]

    detail = client.get(f"/tickets/{tid}", headers=_auth(client.carol["token"])).json()
    assert detail["title"].startswith("订单 52644")
    # 客服需要看到前因：用户和机器人的对话在 transcript 里。
    assert detail["transcript"][0]["role"] == "user"

    # 没认领就回复：冲突
    assert (
        client.post(
            f"/tickets/{tid}/reply",
            json={"content": "我来处理"},
            headers=_auth(client.carol["token"]),
        ).status_code
        == 409
    )

    claimed = client.post(f"/tickets/{tid}/claim", headers=_auth(client.carol["token"])).json()
    assert claimed["ticket"]["status"] == "claimed"
    assert claimed["ticket"]["claimed_by_name"] == "carol"

    replied = client.post(
        f"/tickets/{tid}/reply",
        json={"content": "我是人工客服，正在帮你查退款。"},
        headers=_auth(client.carol["token"]),
    )
    assert replied.status_code == 200

    conversation = client.get(f"/sessions/{sid}", headers=_auth(client.bob["token"])).json()
    assert conversation["messages"][-1]["role"] == "agent"
    assert conversation["messages"][-1]["author"] == "carol"
    assert conversation["messages"][-1]["content"] == "我是人工客服，正在帮你查退款。"

    closed = client.post(f"/tickets/{tid}/close", headers=_auth(client.carol["token"])).json()
    assert closed["ticket"]["status"] == "closed"


def test_handoff_endpoint_is_idempotent_and_owner_only(client):
    sid = client.post(
        "/chat", json={"question": "hi"}, headers=_auth(client.bob["token"])
    ).json()["session_id"]

    assert (
        client.post(f"/sessions/{sid}/handoff", headers=_auth(client.alice["token"])).status_code
        == 404
    )

    first = client.post(f"/sessions/{sid}/handoff", headers=_auth(client.bob["token"])).json()
    second = client.post(f"/sessions/{sid}/handoff", headers=_auth(client.bob["token"])).json()
    assert first["ticket"]["ticket_id"] == second["ticket"]["ticket_id"]


def test_tickets_are_closed_in_demo_mode(client, monkeypatch):
    """公网演示不提供人工客服——没有 agent 账号，也没有理由留一个客服台。"""
    monkeypatch.setattr(api.demo, "enabled", lambda: True)

    assert client.get("/tickets", headers=_auth(client.alice["token"])).status_code == 404
    # 不走 /chat：演示模式下它会先要求邀请码，测不到工单端点。
    sid = api.session.ensure_session("", owner_id=client.bob["id"])
    assert (
        client.post(f"/sessions/{sid}/handoff", headers=_auth(client.bob["token"])).status_code
        == 404
    )


def test_tickets_do_not_exist_without_the_identity_layer(client, monkeypatch):
    monkeypatch.setattr(auth_settings, "AUTH_ENABLED", False)

    assert client.get("/tickets").status_code == 404
