"""
HTTP 接口（api.py）的测试。**全仓库第一个覆盖 api.py 的文件。**

在此之前 api.py 靠"有人真的 curl 一下"来保证正确——那是最弱的一种保证，
因为接口坏掉的方式恰好是"curl 一下看起来没问题"：少一个字段、错误体换了
形状、多轮接不上，这三种都不会让请求失败。

三条规矩，和全仓库一致：

1. **不联网、不调模型。** `answer` 被整个换掉，所以这里跑的是"接口这一层
   的参数解析、会话读写、鉴权、错误体形状"，不含任何 LLM。真正的端到端
   验证是 README 里那次"真起服务打一次"。
2. **不碰真数据。** 会话库指向 tmp_path，trace 文件也是。
3. 没有 fastapi / httpx 的环境（比如宿主上那个只装了内核的 venv）**跳过**，
   而不是失败——这个仓库的测试要能在"只装了内核"的环境里跑完。
"""
import json

import pytest

pytest.importorskip("fastapi", reason="需要 uv sync --extra api")
pytest.importorskip("httpx", reason="TestClient 需要 httpx（dev 依赖组）")

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    """一个不联网、不碰真数据的客户端。

    `session.SESSION_DB_PATH` 要换掉：`/chat` 会真的去开会话库，
    不换的话测试会往仓库的 data/ 里写东西。
    """
    monkeypatch.setattr(api.session, "SESSION_DB_PATH", str(tmp_path / "sessions.db"))
    monkeypatch.setattr(api.trace, "LOGS_DIR", tmp_path)
    api.session.reset_connection()

    seen = []

    def _fake_answer(question, history):  # noqa: ANN001
        seen.append({"question": question, "history": list(history)})
        return {
            "request_id": f"req-{len(seen)}",
            "answer": f"答复：{question}",
            "sources": ["订单库"],
            "decision": "answer",
            "plan": ["order"],
            "tokens": {"in": 1, "out": 2},
        }

    monkeypatch.setattr(api, "answer", _fake_answer)

    streamed = []

    def _fake_answer_stream(question, history):  # noqa: ANN001
        streamed.append({"question": question, "history": list(history)})
        yield {"type": "start", "request_id": "req-stream"}
        yield {"type": "delta", "text": "答复："}
        yield {"type": "delta", "text": question}
        yield {
            "type": "final",
            "state": {
                "request_id": "req-stream",
                "answer": f"答复：{question}",
                "sources": ["订单库"],
                "decision": "answer",
                "plan": ["order"],
                "tokens": {"in": 1, "out": 2},
            },
            "elapsed_ms": 5,
        }

    monkeypatch.setattr(api, "answer_stream", _fake_answer_stream)
    # **默认把 API_KEY 清掉**：模块导入时会读一次 .env，本机如果真配了
    # API_KEY，下面那些"没配就能放行"的用例会以一种很难懂的方式红掉。
    monkeypatch.setattr(api, "API_KEY", "")

    with TestClient(api.app) as c:
        c.seen = seen
        c.streamed = streamed
        yield c

    api.session.reset_connection()


# ——— 基本形状 ———

def test_health_reports_components_without_touching_the_model(client):
    r = client.get("/health")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["order_db"]          # sqlite / mysql，取决于配置
    assert body["max_rounds"] >= 1
    assert isinstance(body["workers"], list)


def test_chat_returns_every_declared_field(client):
    r = client.post("/chat", json={"question": "订单 52644 到哪了"})

    assert r.status_code == 200
    body = r.json()
    # 一个字段都不能少：少一个在 curl 里看不出来，在客户端里是 KeyError。
    for key in ("request_id", "session_id", "answer", "sources", "decision", "plan", "tokens"):
        assert key in body, f"响应里缺 {key}"
    assert body["answer"] == "答复：订单 52644 到哪了"


def test_a_request_without_a_session_id_gets_one_back(client):
    """不给会话号 = 新开会话，但**服务端必须把生成的号还回来**。

    不还的话，落盘会话就只有服务端知道——多轮在 HTTP 上根本用不起来，
    而症状是"每次提问都像第一次"，看起来像模型的问题。
    """
    body = client.post("/chat", json={"question": "你好"}).json()

    assert body["session_id"]
    assert len(body["session_id"]) >= 30, "会话号是凭证，不能短到可枚举"


# ——— 多轮：这条才是会话持久化的真回归 ———

def test_the_second_request_with_the_same_session_id_sees_the_first(client):
    first = client.post("/chat", json={"question": "第一问"}).json()
    sid = first["session_id"]

    second = client.post("/chat", json={"question": "第二问", "session_id": sid}).json()

    assert second["session_id"] == sid, "同一个会话号不该被换掉"
    history = client.seen[1]["history"]
    assert [m["content"] for m in history] == ["第一问", "答复：第一问"]
    assert [m["role"] for m in history] == ["user", "assistant"]


def test_two_sessions_do_not_share_context(client):
    a = client.post("/chat", json={"question": "A 的问题"}).json()["session_id"]
    b = client.post("/chat", json={"question": "B 的问题"}).json()["session_id"]

    client.post("/chat", json={"question": "A 又问", "session_id": a})

    assert a != b
    assert client.seen[-1]["history"][0]["content"] == "A 的问题"


def test_both_turns_are_written_to_the_store(client):
    sid = client.post("/chat", json={"question": "问"}).json()["session_id"]

    stored = api.session.history(sid)

    assert [m["content"] for m in stored] == ["问", "答复：问"]


# ——— 流式：SSE 的事件形状与会话语义 ———

def _sse_events(body: str) -> list:
    """把 SSE 正文拆回事件。只认 `data: ` 开头的行——注释/空行是协议的一部分。"""
    return [
        json.loads(line[len("data: "):])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def _stream_body(client, question="你好", session_id=""):  # noqa: ANN001
    payload = {"question": question}
    if session_id:
        payload["session_id"] = session_id
    with client.stream("POST", "/chat/stream", json=payload) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        return "".join(r.iter_text())


def test_stream_emits_start_delta_final_in_order(client):
    events = _sse_events(_stream_body(client, "订单 52644 到哪了"))

    assert [e["type"] for e in events] == ["start", "delta", "delta", "final"]
    assert events[0]["request_id"] == "req-stream"
    assert events[0]["session_id"] == events[-1]["session_id"]
    assert events[-1]["answer"] == "答复：订单 52644 到哪了"
    assert events[-1]["tokens"] == {"in": 1, "out": 2}
    assert events[-1]["elapsed_ms"] == 5


def test_stream_writes_the_turn_to_the_store_after_final(client):
    events = _sse_events(_stream_body(client, "问"))

    sid = events[-1]["session_id"]
    assert [m["content"] for m in api.session.history(sid)] == ["问", "答复：问"]


def test_stream_reuses_the_session_id_on_the_second_turn(client):
    first = _sse_events(_stream_body(client, "第一问"))
    sid = first[-1]["session_id"]

    second = _sse_events(_stream_body(client, "第二问", sid))

    assert second[-1]["session_id"] == sid
    assert [m["content"] for m in api.session.history(sid)] == [
        "第一问", "答复：第一问", "第二问", "答复：第二问",
    ]


def test_stream_reports_an_error_without_persisting_a_half_turn(client, monkeypatch):
    """流到一半挂了：发 `error` 事件、**不落盘**、不回异常原文。

    半句话接进下一轮，比没答更坏——下一轮模型会把它当成"用户已经知道的事"。
    """
    def _boom(question, history):  # noqa: ANN001
        yield {"type": "start", "request_id": "req-boom"}
        yield {"type": "delta", "text": "答了一半"}
        raise RuntimeError("机密：/app/data/orders.sqlite 打不开")

    monkeypatch.setattr(api, "answer_stream", _boom)

    events = _sse_events(_stream_body(client, "随便"))

    assert [e["type"] for e in events] == ["start", "delta", "error"]
    assert events[-1]["request_id"] == "req-boom"
    assert "orders.sqlite" not in events[-1]["error"], "内部路径被回给客户端了"
    assert api.session.history(events[0]["session_id"]) == []


# ——— 鉴权：配了才强制 ———

def test_without_a_configured_key_everything_is_open(client):
    """**没配 API_KEY 就该放行**（并且 import 时已经告警过）。

    这是默认策略，不是漏做：这个接口的用途是"clone 下来就能 curl 一下"。
    把它反过来做成 fail closed，会挡住所有第一次接触的人。
    """
    assert client.post("/chat", json={"question": "裸奔"}).status_code == 200
    assert client.get("/trace").status_code in (200, 404)   # 404 是"还没有记录"，不是鉴权


def test_a_configured_key_is_enforced(client, monkeypatch):
    monkeypatch.setattr(api, "API_KEY", "s3cret")

    assert client.post("/chat", json={"question": "没带 key"}).status_code == 401
    assert client.post(
        "/chat", json={"question": "带错 key"}, headers={"X-API-Key": "wrong"}
    ).status_code == 401
    assert client.post(
        "/chat", json={"question": "带对 key"}, headers={"X-API-Key": "s3cret"}
    ).status_code == 200


def test_a_non_ascii_key_is_compared_without_exploding(monkeypatch):
    """比较那一行**不能**因为 key 里有中文就抛 TypeError。

    这条直接调 `require_api_key` 而不走 HTTP，是因为**它走不了 HTTP**：
    HTTP 头的值只能是 ASCII 字节，httpx 在客户端就会拒收一个中文值
    （`UnicodeEncodeError: 'ascii' codec can't encode …`）。所以真实世界里
    "配了中文 API_KEY"的后果是**每一个请求都 401**——客户端根本发不出这个值。
    api.py 在 import 时会为此打一条 WARNING。

    这里要钉的是另一半：服务端拿到这种 key 时**不该 500**。只写
    `secrets.compare_digest(a, b)` 而不先 encode，碰上非 ASCII 的 str 会抛
    TypeError——那会把"配错了 key"表现成"服务坏了"，两种问题得能分开。
    """
    monkeypatch.setattr(api, "API_KEY", "中文口令·2026")

    api.require_api_key(x_api_key="中文口令·2026")   # 对了就放行，不抛
    with pytest.raises(HTTPException) as exc:        # 错了就 401，不是 TypeError
        api.require_api_key(x_api_key="别的口令")
    assert exc.value.status_code == 401


def test_health_and_docs_stay_open_even_with_a_key_configured(client, monkeypatch):
    """**健康检查和文档永远不放鉴权。**

    探针唯一的用途就是"在什么都还没配好的时候也能被调到"；给它加凭证，
    等于让"服务起没起来"这件事本身也依赖一个配置项。
    """
    monkeypatch.setattr(api, "API_KEY", "s3cret")

    assert client.get("/health").status_code == 200
    assert client.get("/openapi.json").status_code == 200


# ——— 错误体：一条形状，且不带原文 ———

def test_a_missing_trace_gives_a_structured_404(client):
    body = client.get("/trace").json()

    assert set(body) == {"error", "request_id"}, "错误体只有这一种形状"
    assert body["error"]


def test_a_bad_request_body_gives_a_structured_422(client):
    r = client.post("/chat", json={"question": ""})   # min_length=1

    assert r.status_code == 422
    body = r.json()
    assert set(body) == {"error", "request_id"}
    assert "question" in body["error"], "哪个字段错了要说出来"


def test_an_unhandled_exception_does_not_leak_its_text(client, monkeypatch):
    """兜底 handler：异常原文只进日志。

    图里会抛的东西五花八门，原文里可能有内部路径甚至配置片段。客户端拿到
    的应该是一句人话 + 一个 request_id，而不是一份源码地图。
    """
    def _boom(question, history):  # noqa: ANN001
        raise RuntimeError("机密：/app/data/orders.sqlite 打不开")

    monkeypatch.setattr(api, "answer", _boom)
    with TestClient(api.app, raise_server_exceptions=False) as c:
        c.seen = []
        r = c.post("/chat", json={"question": "随便"})

    assert r.status_code == 500
    body = r.json()
    assert set(body) == {"error", "request_id"}
    assert "orders.sqlite" not in body["error"], "内部路径被回给客户端了"
    assert body["request_id"], "没有 request_id，维护者就没法对日志"


# ——— trace：字段要原样透传 ———

def test_trace_passes_through_fields_it_does_not_know(client, tmp_path):
    """`TraceItem` 只声明了它保证有的字段，其余的必须原样出来。

    这一条钉的是一个**很容易静默发生**的错：给响应模型列了字段清单，却没开
    `extra="allow"`，于是 trace 里最有用的那部分（每个 worker 的成功与否和
    耗时）在序列化时被悄悄吃掉——接口 200、字段齐全、内容空了。
    """
    line = {
        "request_id": "r-1",
        "question": "订单 52644 到哪了",
        "plan": ["order"],
        "router_source": "rule",
        "results": [{"worker": "order", "ok": True, "elapsed_ms": 12}],
    }
    (tmp_path / "requests.jsonl").write_text(json.dumps(line, ensure_ascii=False), encoding="utf-8")

    body = client.get("/trace").json()

    assert body["count"] == 1
    item = body["items"][0]
    assert item["request_id"] == "r-1"
    assert item["results"] == [{"worker": "order", "ok": True, "elapsed_ms": 12}], "extra 字段被吃掉了"


def test_trace_skips_a_half_written_line(client, tmp_path):
    """正在写的那一行可能是半行——跳过它，不要 500。

    这个文件是追加写的，读的时候正好撞上一次写了一半，是正常情况。
    """
    path = tmp_path / "requests.jsonl"
    path.write_text(
        json.dumps({"request_id": "完整"}, ensure_ascii=False) + "\n" + '{"request_id": "半',
        encoding="utf-8",
    )

    body = client.get("/trace").json()

    assert body["count"] == 1
    assert body["items"][0]["request_id"] == "完整"


# ——— 文档本身也是被交付的东西 ———

def test_the_openapi_document_carries_the_hand_written_metadata(client):
    """`/docs` 上那份文档不是自动生成完就完事的——标题、分组、每个端点的说明
    都是手写的，会被别人当成"这个项目做了文档"的依据。所以它得真的在。"""
    spec = client.get("/openapi.json").json()

    assert spec["info"]["title"]
    assert spec["info"]["description"], "顶层说明没了，/docs 首页会是一片空白"
    assert {t["name"] for t in spec["tags"]} == {"health", "chat", "trace"}
    for path in ("/health", "/chat", "/trace"):
        assert spec["paths"][path]
    assert spec["paths"]["/chat"]["post"]["summary"]
    # 401 的响应模型要出现在文档里，否则看文档的人不知道有无鉴权
    assert "401" in spec["paths"]["/chat"]["post"]["responses"]


# ——— 公网演示：邀请码 / 限流 / 预算 / trace ——

@pytest.fixture
def demo_client(client, monkeypatch, tmp_path):
    """打开演示模式，并把账本、限流都换成测试可控的。"""
    monkeypatch.setattr(api.demo, "DEMO_MODE", True)
    monkeypatch.setattr(api.demo, "DEMO_INVITE_CODE", "let-me-in")
    monkeypatch.setattr(api.demo, "DEMO_USAGE_DB_PATH", tmp_path / "usage.sqlite")
    monkeypatch.setattr(api.demo, "DEMO_DAILY_TOKEN_BUDGET", 0)      # 默认不限额
    monkeypatch.setattr(api.demo, "DEMO_RATE_LIMIT_PER_MINUTE", 0)   # 默认不限流
    api.demo.reset_rate_limits()
    yield client
    api.demo.reset_rate_limits()


INVITE = {"X-Invite-Code": "let-me-in"}


def test_demo_mode_requires_the_invite_code(demo_client):
    assert demo_client.post("/chat", json={"question": "没带码"}).status_code == 403
    assert demo_client.post(
        "/chat", json={"question": "带错码"}, headers={"X-Invite-Code": "wrong"}
    ).status_code == 403
    assert demo_client.post(
        "/chat", json={"question": "带对码"}, headers=INVITE
    ).status_code == 200


def test_demo_mode_requires_the_invite_code_on_the_stream_too(demo_client):
    """流式端点不能是后门：它调的是同一套编排，也会花钱。"""
    assert demo_client.post("/chat/stream", json={"question": "没带码"}).status_code == 403
    with demo_client.stream(
        "POST", "/chat/stream", json={"question": "带对码"}, headers=INVITE
    ) as r:
        assert r.status_code == 200


def test_demo_mode_hides_trace(demo_client, tmp_path):
    """公网演示不开放 trace——里面有用户问过的话和内部耗时。

    返回 404 而不是 403：403 等于告诉对方"这里有东西，只是不给你看"。
    """
    (tmp_path / "requests.jsonl").write_text(
        json.dumps({"request_id": "r-1"}) + "\n", encoding="utf-8"
    )

    assert demo_client.get("/trace").status_code == 404


def test_health_reports_demo_mode_and_hides_disabled_workers(demo_client):
    body = demo_client.get("/health").json()

    assert body["demo"] is True
    assert "analytics" not in body["workers"]


def test_demo_rate_limit_returns_429_before_calling_the_model(demo_client, monkeypatch):
    monkeypatch.setattr(api.demo, "DEMO_RATE_LIMIT_PER_MINUTE", 1)

    assert demo_client.post("/chat", json={"question": "第一次"}, headers=INVITE).status_code == 200
    r = demo_client.post("/chat", json={"question": "第二次"}, headers=INVITE)

    assert r.status_code == 429
    assert r.headers["retry-after"] == "60"
    # 被限流的那次**不该**走到 answer()：否则钱已经花了才说"太频繁"
    assert len(demo_client.seen) == 1


def test_demo_daily_budget_blocks_further_requests(demo_client, monkeypatch):
    monkeypatch.setattr(api.demo, "DEMO_DAILY_TOKEN_BUDGET", 3)

    # 假 answer 每次报 3 个 token（in=1, out=2），刚好用满额度
    assert demo_client.post("/chat", json={"question": "第一次"}, headers=INVITE).status_code == 200
    assert api.demo.tokens_used_today() == 3

    r = demo_client.post("/chat", json={"question": "第二次"}, headers=INVITE)
    assert r.status_code == 429
    assert "额度" in r.json()["error"]
    assert len(demo_client.seen) == 1


def test_the_stream_also_records_its_tokens(demo_client):
    """流式的用量在 `final` 里，不记的话预算对流式用户就是失效的。"""
    with demo_client.stream(
        "POST", "/chat/stream", json={"question": "问"}, headers=INVITE
    ) as r:
        assert r.status_code == 200
        "".join(r.iter_text())

    assert api.demo.tokens_used_today() == 3
