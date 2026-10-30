"""
MCP 工具层的契约测试：**不装 SDK、不连网络、不碰真库**。

覆盖三类东西：
  1. 输入校验——不合法时是 `invalid_input`，不是抛异常
  2. 输出裁剪——PII 字段不许出现在只读查单里
  3. 开关——PII / 写操作默认关闭，开了之后还有第二道闸（confirm / 演示只读）
"""
import pytest

from ecom.mcp import tools


def _data(out):
    assert out["ok"] is True, out
    assert out["meta"]["elapsed_ms"] >= 0
    return out["data"]


def _error(out, code):
    assert out["ok"] is False, out
    assert out["error"]["code"] == code, out
    assert out["error"]["message"]
    return out["error"]


# ——————————————————————————————————————————————————————————————
# get_order
# ——————————————————————————————————————————————————————————————
def test_get_order_returns_order_and_logistics(db):
    data = _data(tools.get_order("52644", db=db))
    assert data["order"] == {
        "order_id": "52644",
        "status": "COMPLETE",
        "amount": 1234.5,
        "region": "华东",
        "created_at": "2024-01-02",
        "paid_at": None,
    }
    assert data["logistics"]["to_city"] == "上海"


def test_get_order_never_returns_customer_identity(db):
    """白名单不是"少写几列"：客户姓名和客户号都不该出现在返回里。"""
    data = _data(tools.get_order("52644", db=db))
    assert "customer_name" not in data["order"]
    assert "customer_id" not in data["order"]


@pytest.mark.parametrize("bad", ["", "1234", "C20755", "52644abc", None])
def test_get_order_rejects_bad_id(bad, db):
    _error(tools.get_order(bad, db=db), "invalid_input")


def test_get_order_reports_not_found(db):
    _error(tools.get_order("99999", db=db), "not_found")


def test_get_order_internal_error_does_not_leak_detail(db, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("dsn=mysql://user:secret@host/db")

    monkeypatch.setattr(tools, "query_order", boom)
    error = _error(tools.get_order("52644", db=db), "internal")
    assert "secret" not in error["message"]


# ——————————————————————————————————————————————————————————————
# get_order_returns
# ——————————————————————————————————————————————————————————————
def test_get_order_returns_lists_existing_requests(db):
    data = _data(tools.get_order_returns("52644", db=db))
    assert data["count"] == 1
    assert data["returns"][0]["status"] == "待审核"
    assert data["returns"][0]["return_id"] == "R0000000001"


def test_get_order_returns_distinguishes_missing_order_from_no_returns(db):
    """订单不存在 ≠ 没有退货申请。两者都返回空列表的话，用户会把号打错当成没退过。"""
    _error(tools.get_order_returns("99999", db=db), "not_found")
    data = _data(tools.get_order_returns("52645", db=db))
    assert data == {"order_id": "52645", "count": 0, "returns": []}


# ——————————————————————————————————————————————————————————————
# search_knowledge
# ——————————————————————————————————————————————————————————————
@pytest.fixture
def fake_retrieval(monkeypatch):
    calls = {}

    def fake_retrieve(query, store):
        calls["query"] = query
        return [
            {"chunk_id": "c1", "source": "退货政策.md", "text": "七天无理由……", "score": 0.9},
            {"chunk_id": "c2", "source": "物流说明.md", "text": "偏远地区……", "score": 0.5},
        ]

    def fake_rerank(query, contexts, top_k=3):
        calls["top_k"] = top_k
        return contexts[:top_k]

    monkeypatch.setattr("ecom.knowledge.retriever.retriever.retrieve", fake_retrieve)
    monkeypatch.setattr("ecom.knowledge.retriever.reranker.rerank", fake_rerank)
    monkeypatch.setattr("ecom.knowledge.vector_store.get_vector_store", lambda: object())
    return calls


def test_search_knowledge_returns_passages_not_answers(fake_retrieval):
    data = _data(tools.search_knowledge("退货要多久", top_k=1))
    assert data["count"] == 1
    assert data["hits"][0]["source"] == "退货政策.md"
    assert "answer" not in data
    assert fake_retrieval == {"query": "退货要多久", "top_k": 1}


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_search_knowledge_rejects_empty_query(bad):
    _error(tools.search_knowledge(bad), "invalid_input")


def test_search_knowledge_rejects_long_query(monkeypatch):
    monkeypatch.setattr(tools.settings, "MCP_MAX_QUERY_CHARS", 5)
    _error(tools.search_knowledge("一二三四五六"), "invalid_input")


@pytest.mark.parametrize("bad", [0, 6, True, "abc"])
def test_search_knowledge_rejects_bad_top_k(bad):
    _error(tools.search_knowledge("退货", top_k=bad), "invalid_input")


def test_search_knowledge_treats_null_top_k_as_default(fake_retrieval):
    data = _data(tools.search_knowledge("退货", top_k=None))
    assert data["count"] == 2


def test_search_knowledge_reports_unavailable_when_retrieval_breaks(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("embedding 服务挂了")

    monkeypatch.setattr("ecom.knowledge.retriever.retriever.retrieve", boom)
    monkeypatch.setattr("ecom.knowledge.vector_store.get_vector_store", lambda: object())
    error = _error(tools.search_knowledge("退货"), "unavailable")
    assert "embedding" not in error["message"]


# ——————————————————————————————————————————————————————————————
# list_customer_orders（PII 闸）
# ——————————————————————————————————————————————————————————————
def test_customer_orders_are_disabled_by_default(db):
    _error(tools.list_customer_orders("C20755", db=db), "not_allowed")


def test_customer_orders_return_only_minimal_fields(db, monkeypatch):
    monkeypatch.setattr(tools.settings, "MCP_ENABLE_PII", True)
    data = _data(tools.list_customer_orders("c20755", db=db))
    assert data["customer_id"] == "C20755"
    assert data["count"] == 2
    assert set(data["orders"][0]) == {"order_id", "status", "amount", "created_at"}


def test_customer_orders_honor_limit(db, monkeypatch):
    monkeypatch.setattr(tools.settings, "MCP_ENABLE_PII", True)
    data = _data(tools.list_customer_orders("C20755", limit=1, db=db))
    assert data["count"] == 1


@pytest.mark.parametrize("bad", ["", "C123", "C123456789", "20755", None])
def test_customer_orders_reject_bad_id(bad, db, monkeypatch):
    monkeypatch.setattr(tools.settings, "MCP_ENABLE_PII", True)
    _error(tools.list_customer_orders(bad, db=db), "invalid_input")


@pytest.mark.parametrize("bad", [0, 21, True, "x"])
def test_customer_orders_reject_bad_limit(bad, db, monkeypatch):
    monkeypatch.setattr(tools.settings, "MCP_ENABLE_PII", True)
    _error(tools.list_customer_orders("C20755", limit=bad, db=db), "invalid_input")


# ——————————————————————————————————————————————————————————————
# create_return（写闸 + confirm 闸）
# ——————————————————————————————————————————————————————————————
@pytest.fixture
def writes_on(monkeypatch):
    monkeypatch.setattr(tools.settings, "MCP_ENABLE_WRITES", True)


def test_create_return_is_disabled_by_default(db):
    _error(tools.create_return("52646", "不喜欢", confirm=True, db=db), "not_allowed")


def test_create_return_is_blocked_in_demo_mode(db, writes_on, monkeypatch):
    """开关开了也不能在只读演示环境里落库——两道服务端闸是独立的。"""
    monkeypatch.setattr("ecom.demo.DEMO_MODE", True)
    _error(tools.create_return("52646", "不喜欢", confirm=True, db=db), "not_allowed")
    assert db.count("return_requests") == 1


def test_create_return_requires_explicit_confirmation(db, writes_on):
    _error(tools.create_return("52646", "不喜欢", db=db), "invalid_input")
    _error(tools.create_return("52646", "不喜欢", confirm="true", db=db), "invalid_input")
    assert db.count("return_requests") == 1


def test_create_return_writes_pending_request(db, writes_on):
    data = _data(tools.create_return("52646", "  尺寸不合  ", confirm=True, db=db))
    assert data["status"] == "待审核"
    assert data["order_id"] == "52646"
    assert data["return_id"]
    rows = db.query("SELECT * FROM return_requests WHERE order_id = ?", ("52646",))
    assert len(rows) == 1
    assert rows[0]["status"] == "待审核"
    assert rows[0]["reason"] == "尺寸不合"


def test_create_return_rejects_non_returnable_order(db, writes_on):
    _error(tools.create_return("52645", "不想要了", confirm=True, db=db), "not_allowed")
    assert db.count("return_requests") == 1


def test_create_return_rejects_duplicate_request(db, writes_on):
    _data(tools.create_return("52646", "第一次", confirm=True, db=db))
    _error(tools.create_return("52646", "第二次", confirm=True, db=db), "not_allowed")
    assert db.count("return_requests") == 2


def test_create_return_rejects_missing_order(db, writes_on):
    _error(tools.create_return("99999", "不想要了", confirm=True, db=db), "not_found")


@pytest.mark.parametrize("reason", ["", "   ", "x" * 256])
def test_create_return_rejects_bad_reason(reason, db, writes_on):
    _error(tools.create_return("52646", reason, confirm=True, db=db), "invalid_input")


def test_create_return_rejects_bad_order_id(db, writes_on):
    _error(tools.create_return("1234", "不想要了", confirm=True, db=db), "invalid_input")
