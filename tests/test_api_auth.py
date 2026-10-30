"""身份层接入 API 的测试：token、角色、会话归属。

默认的 `AUTH_ENABLED=0` 行为由 `tests/test_api.py` 守着；这里只验打开之后
多出来的那三条：**没 token 401、不是自己的会话 404、trace 要 admin**。
"""
import pytest

pytest.importorskip("fastapi", reason="需要 uv sync --extra api")
pytest.importorskip("httpx", reason="TestClient 需要 httpx（dev 依赖组）")

from fastapi.testclient import TestClient  # noqa: E402
from starlette.requests import Request  # noqa: E402

import api  # noqa: E402
from ecom import chat_quota  # noqa: E402
from ecom.auth import settings as auth_settings  # noqa: E402
from ecom.auth import store as auth_store  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api.session, "SESSION_DB_PATH", str(tmp_path / "sessions.db"))
    monkeypatch.setattr(api.trace, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(auth_store, "AUTH_DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(auth_settings, "AUTH_ENABLED", True)
    monkeypatch.setattr(api, "API_KEY", "")
    # 限流账本是进程内全局变量；换一个干净的，避免测试之间互相计数。
    monkeypatch.setattr(api, "_login_hits", {})
    # 账号配额同样有进程内计数和磁盘账本，测试一律写到 tmp_path。
    monkeypatch.setattr(
        chat_quota, "CHAT_USAGE_DB_PATH", str(tmp_path / "chat_usage.sqlite")
    )
    monkeypatch.setattr(chat_quota, "CHAT_RATE_LIMIT_PER_MINUTE", 30)
    monkeypatch.setattr(chat_quota, "CHAT_DAILY_TOKEN_BUDGET", 200000)
    # 全站预算默认关掉，让每条用例只关心自己那一道闸；要测它的用例自己打开。
    monkeypatch.setattr(chat_quota, "CHAT_GLOBAL_DAILY_TOKEN_BUDGET", 0)
    chat_quota.reset_rate_limits()
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


def _request(headers=None, client=("10.0.0.1", 1234)):
    """只给限流用得到的那几个 scope 字段，不启动 ASGI。"""
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/auth/login",
            "headers": [
                (key.lower().encode(), value.encode())
                for key, value in (headers or {}).items()
            ],
            "client": client,
        }
    )


def test_production_refuses_to_start_without_auth_or_demo(monkeypatch):
    # 默认 dev 下不检查；只有显式 APP_ENV=prod 才生效。
    monkeypatch.setattr(auth_settings, "IS_PRODUCTION", True)
    monkeypatch.setattr(auth_settings, "AUTH_ENABLED", False)
    monkeypatch.setattr(api.demo, "enabled", lambda: False)

    message = auth_settings.production_problem(demo_mode=False)
    assert message and "AUTH_ENABLED" in message

    # 起不来的服务，比一个看起来正常、实际裸奔的服务安全。
    with pytest.raises(RuntimeError, match="AUTH_ENABLED"):
        with TestClient(api.app):
            pass


def test_production_allows_auth_or_explicit_read_only_demo(monkeypatch):
    monkeypatch.setattr(auth_settings, "IS_PRODUCTION", True)

    monkeypatch.setattr(auth_settings, "AUTH_ENABLED", True)
    assert auth_settings.production_problem(demo_mode=False) is None

    monkeypatch.setattr(auth_settings, "AUTH_ENABLED", False)
    assert auth_settings.production_problem(demo_mode=True) is None


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


def test_client_ip_ignores_forwarded_headers_unless_the_proxy_is_trusted(client, monkeypatch):
    """默认不信任 `X-Forwarded-For`：否则换一个假 IP 就绕过登录限流。"""
    monkeypatch.setattr(auth_settings, "TRUST_PROXY_HEADERS", False)
    request = _request({"X-Forwarded-For": "203.0.113.9", "X-Real-IP": "203.0.113.10"})
    assert api._client_ip(request) == "10.0.0.1"

    # 显式信任反代后取最后一跳：最左边是客户端自报的，可以伪造。
    monkeypatch.setattr(auth_settings, "TRUST_PROXY_HEADERS", True)
    request = _request({"X-Forwarded-For": "1.1.1.1, 198.51.100.7"})
    assert api._client_ip(request) == "198.51.100.7"
    assert api._client_ip(_request({"X-Real-IP": "198.51.100.8"})) == "198.51.100.8"


def test_login_limiter_blocks_one_account_across_different_ips(client, monkeypatch):
    """
    只按 IP 限流挡不住"换出口 IP 打同一个账号"。这里钉住账号桶独立生效，
    同时另一个账号从同一来源仍能登录——不能因为某个人被刷就把整台机器锁死。
    """
    monkeypatch.setattr(auth_settings, "TRUST_PROXY_HEADERS", True)
    monkeypatch.setattr(api, "_LOGIN_ACCOUNT_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(api, "_LOGIN_MAX_ATTEMPTS", 100)

    def attempt(name, ip):
        return client.post(
            "/auth/login",
            json={"username": name, "password": "definitely-wrong"},
            headers={"X-Forwarded-For": ip},
        )

    assert attempt("bob", "203.0.113.1").status_code == 401
    assert attempt("bob", "203.0.113.2").status_code == 401
    # 第三个 IP 也没用：账号维度已经满了。用户名大小写不同是同一个桶。
    assert attempt("BOB", "203.0.113.3").status_code == 429
    assert attempt("carol", "203.0.113.3").status_code == 401


def test_logout_revokes_the_login_token_immediately(client):
    auth_store.set_password("carol", "a-long-enough-password")
    issued = client.post(
        "/auth/login", json={"username": "carol", "password": "a-long-enough-password"}
    ).json()
    headers = _auth(issued["token"])

    assert client.get("/sessions", headers=headers).status_code == 200
    r = client.post("/auth/logout", headers=headers)
    assert r.status_code == 200
    assert r.json()["revoked"] is True

    # 不需要等 12 小时自然过期。
    assert client.get("/sessions", headers=headers).status_code == 401
    assert client.post("/auth/logout", headers=headers).status_code == 401


def test_logout_does_not_revoke_the_long_lived_api_token(client):
    """登出只结束当前登录会话；账号的根凭证轮换是另一个显式动作。"""
    r = client.post("/auth/logout", headers=_auth(client.carol["token"]))
    assert r.status_code == 200
    assert r.json()["revoked"] is False
    assert client.get("/sessions", headers=_auth(client.carol["token"])).status_code == 200


def test_logout_is_absent_without_the_identity_layer(client, monkeypatch):
    monkeypatch.setattr(auth_settings, "AUTH_ENABLED", False)
    assert client.post("/auth/logout", headers=_auth("whatever")).status_code == 404


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


def test_agent_cannot_read_another_agents_claimed_ticket_but_admin_can(client, monkeypatch):
    _human_answer(monkeypatch)
    body = client.post(
        "/chat", json={"question": "转人工"}, headers=_auth(client.bob["token"])
    ).json()
    tid = body["handoff"]["ticket_id"]
    dave = auth_store.create_principal("dave", "agent")

    client.post(f"/tickets/{tid}/claim", headers=_auth(client.carol["token"]))

    # 列表按人收口：排队中的大家都能看，别人接手的看不见。
    assert client.get("/tickets?status=open", headers=_auth(dave["token"])).json()[
        "tickets"
    ] == []
    assert client.get("/tickets?status=claimed", headers=_auth(dave["token"])).json()[
        "tickets"
    ] == []
    assert len(
        client.get("/tickets?status=claimed", headers=_auth(client.alice["token"])).json()[
            "tickets"
        ]
    ) == 1

    # 绕过列表直接查详情，也要被挡住。
    assert (
        client.get(f"/tickets/{tid}", headers=_auth(dave["token"])).status_code == 403
    )
    assert client.get(f"/tickets/{tid}", headers=_auth(client.carol["token"])).status_code == 200
    assert client.get(f"/tickets/{tid}", headers=_auth(client.alice["token"])).status_code == 200

    # admin 可以强制关闭别人认领的单，并留下 override 审计。
    closed = client.post(f"/tickets/{tid}/close", headers=_auth(client.alice["token"]))
    assert closed.status_code == 200
    assert closed.json()["ticket"]["status"] == "closed"


def test_admin_can_force_close_an_unclaimed_ticket(client, monkeypatch):
    _human_answer(monkeypatch)
    body = client.post(
        "/chat", json={"question": "转人工"}, headers=_auth(client.bob["token"])
    ).json()
    tid = body["handoff"]["ticket_id"]

    # 普通客服仍然只能关自己认领的。
    assert client.post(f"/tickets/{tid}/close", headers=_auth(client.carol["token"])).status_code == 409
    closed = client.post(f"/tickets/{tid}/close", headers=_auth(client.alice["token"]))
    assert closed.status_code == 200
    assert closed.json()["ticket"]["status"] == "closed"


def test_admin_can_transfer_and_release_a_ticket(client, monkeypatch):
    _human_answer(monkeypatch)
    body = client.post(
        "/chat", json={"question": "转人工"}, headers=_auth(client.bob["token"])
    ).json()
    tid = body["handoff"]["ticket_id"]
    dave = auth_store.create_principal("dave", "agent")
    client.post(f"/tickets/{tid}/claim", headers=_auth(client.carol["token"]))

    # 客服不能自己转派。
    assert (
        client.post(
            f"/tickets/{tid}/transfer",
            json={"assignee": "dave"},
            headers=_auth(client.carol["token"]),
        ).status_code
        == 403
    )

    moved = client.post(
        f"/tickets/{tid}/transfer",
        json={"assignee": "dave"},
        headers=_auth(client.alice["token"]),
    )
    assert moved.status_code == 200
    assert moved.json()["ticket"]["claimed_by_name"] == "dave"
    # 原认领人失去读取和操作权限，新认领人接手。
    assert client.get(f"/tickets/{tid}", headers=_auth(client.carol["token"])).status_code == 403
    assert client.get(f"/tickets/{tid}", headers=_auth(dave["token"])).status_code == 200
    assert (
        client.post(
            f"/tickets/{tid}/reply",
            json={"content": "我来接手"},
            headers=_auth(dave["token"]),
        ).status_code
        == 200
    )

    released = client.post(
        f"/tickets/{tid}/transfer",
        json={"assignee": None},
        headers=_auth(client.alice["token"]),
    )
    assert released.status_code == 200
    assert released.json()["ticket"]["status"] == "open"
    # 回到队列后，所有客服都能在排队里看到它（也才能重新认领）。
    assert client.get(f"/tickets/{tid}", headers=_auth(client.carol["token"])).status_code == 200
    assert (
        client.post(
            f"/tickets/{tid}/transfer",
            json={"assignee": "bob"},
            headers=_auth(client.alice["token"]),
        ).status_code
        == 422
    )


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


# ——— 账号配额 ———

def test_chat_rate_limit_is_per_principal(client, monkeypatch):
    """一个账号刷满不该把另一个账号也挡在门外。"""
    monkeypatch.setattr(chat_quota, "CHAT_RATE_LIMIT_PER_MINUTE", 1)

    assert (
        client.post("/chat", json={"question": "hi"}, headers=_auth(client.bob["token"])).status_code
        == 200
    )
    blocked = client.post(
        "/chat", json={"question": "hi again"}, headers=_auth(client.bob["token"])
    )
    assert blocked.status_code == 429
    assert blocked.json()["error"] == "请求过于频繁，请稍后再试"
    # 流式入口也必须在开始发 SSE 之前拦住，否则状态码已经是 200 了。
    assert (
        client.post(
            "/chat/stream", json={"question": "hi again"}, headers=_auth(client.bob["token"])
        ).status_code
        == 429
    )

    # carol 是另一个 principal，自己的窗口还是空的。
    assert (
        client.post("/chat", json={"question": "hi"}, headers=_auth(client.carol["token"])).status_code
        == 200
    )


def test_chat_daily_budget_is_per_principal_and_persisted(client, monkeypatch):
    """预算按账号记账：bob 用完不影响 carol，且账本在请求返回后就落盘。"""
    monkeypatch.setattr(chat_quota, "CHAT_DAILY_TOKEN_BUDGET", 2)

    # 测试替身每次回答记 {"in": 1, "out": 1}，正好把 2 token 的额度用完。
    assert (
        client.post("/chat", json={"question": "hi"}, headers=_auth(client.bob["token"])).status_code
        == 200
    )
    assert chat_quota.tokens_used_today(client.bob["id"]) == 2
    assert (
        client.post("/chat", json={"question": "again"}, headers=_auth(client.bob["token"])).status_code
        == 429
    )

    assert (
        client.post("/chat", json={"question": "hi"}, headers=_auth(client.carol["token"])).status_code
        == 200
    )
    assert chat_quota.tokens_used_today(client.carol["id"]) == 2


def test_chat_global_budget_stops_every_principal(client, monkeypatch):
    """单人额度之外还有一道全站上限：一个人花完，下一个人也进不来。"""
    monkeypatch.setattr(chat_quota, "CHAT_GLOBAL_DAILY_TOKEN_BUDGET", 2)

    assert (
        client.post("/chat", json={"question": "hi"}, headers=_auth(client.bob["token"])).status_code
        == 200
    )
    assert chat_quota.global_tokens_used_today() == 2
    assert chat_quota.tokens_used_today(client.carol["id"]) == 0

    blocked = client.post(
        "/chat", json={"question": "hi"}, headers=_auth(client.carol["token"])
    )
    assert blocked.status_code == 429
    assert blocked.json()["error"] == "今天的全站 token 额度已经用完"


def test_chat_fails_closed_when_the_quota_ledger_is_unreadable(client, monkeypatch):
    """
    账本坏了要拒绝，而不是当成"今天还没用过"继续花。两个入口都要在
    开始回答之前拦下来。
    """

    def broken(principal_id):  # noqa: ANN001
        raise chat_quota.LedgerUnavailable("boom")

    monkeypatch.setattr(chat_quota, "tokens_used_today", broken)

    for path in ("/chat", "/chat/stream"):
        r = client.post(path, json={"question": "hi"}, headers=_auth(client.bob["token"]))
        assert r.status_code == 503, path
        assert "账本" in r.json()["error"], path


def test_chat_quota_is_a_no_op_without_a_principal(tmp_path, monkeypatch):
    """匿名请求由 demo 的按 IP 闸负责；这里不能凭空给一个空 id 记账。"""
    monkeypatch.setattr(chat_quota, "CHAT_USAGE_DB_PATH", str(tmp_path / "quota.sqlite"))

    assert chat_quota.allow_request("") is True
    assert chat_quota.budget_exhausted("") is False
    assert chat_quota.add_tokens("", {"in": 100, "out": 100}) == 0
    assert chat_quota.tokens_used_today("") == 0
