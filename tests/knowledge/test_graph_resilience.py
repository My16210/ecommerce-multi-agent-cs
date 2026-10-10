"""
图节点的降级行为。

这是 A1 那次改动的验收点：网络抖一下，用户**必须**还能拿到答案。
retrieve_node 和 answer_node 是图上仅有的两个会发网络请求的节点
（web_search 自己已经兜底了），它们失败时图要照常跑到 END，
而不是把异常抛给 app.invoke() 的调用方、把整个会话打断。

这里的三个外部调用全部被 monkeypatch 掉了，测试不联网。
"""
import pytest

from ecom.knowledge.core import graph as graph_module
from ecom.knowledge.core.graph import ANSWER_FALLBACK, build_graph

THREAD = {"configurable": {"thread_id": "test-session"}}


class _FakeStore:
    """build_graph 只把它透传给 retrieve()。这里非空即可

    注意不能传 None——build_graph 里是 `store or get_vector_store()`，
    传 None 会真的去建一个向量库连接。
    """

    def count(self):
        return 1

    def close(self):
        pass


@pytest.fixture
def app(monkeypatch):
    # web_search 也会发网络请求，一并假掉，保证测试完全不联网
    monkeypatch.setattr(graph_module, "web_search", lambda question: [])
    return build_graph(_FakeStore())


def test_retrieval_failure_degrades_to_empty_contexts(monkeypatch, app):
    """embedding 挂了：知识库这一路退化成空，问答照常进行"""

    def boom(question, store, top_k=5):
        raise RuntimeError("embedding 接口 503")

    monkeypatch.setattr(graph_module, "retrieve", boom)
    monkeypatch.setattr(graph_module, "generate_answer", lambda *a, **kw: "照常回答")

    result = app.invoke({"question": "随便问点什么"}, config=THREAD)

    assert result["contexts"] == []
    assert result["answer"] == "照常回答"


def test_generation_failure_returns_fallback_text(monkeypatch, app):
    """千问挂了：给一句明确的兜底文案，图和 invoke() 都正常收场"""

    def boom(*args, **kwargs):
        raise RuntimeError("千问 429")

    monkeypatch.setattr(graph_module, "retrieve", lambda *a, **kw: [])
    monkeypatch.setattr(graph_module, "rerank", lambda *a, **kw: [])
    monkeypatch.setattr(graph_module, "generate_answer", boom)

    result = app.invoke({"question": "随便问点什么"}, config=THREAD)

    assert result["answer"] == ANSWER_FALLBACK


def test_history_still_advances_when_generation_fails(monkeypatch, app):
    """一次故障不能让多轮记忆断档

    失败的那轮也要照常往 chat_history 里追加 user/assistant 两条，
    否则历史就断在这里了。
    """
    monkeypatch.setattr(graph_module, "retrieve", lambda *a, **kw: [])
    monkeypatch.setattr(graph_module, "rerank", lambda *a, **kw: [])
    monkeypatch.setattr(graph_module, "generate_answer", lambda *a, **kw: "正常回答")

    app.invoke({"question": "第一个问题"}, config=THREAD)

    def boom(*args, **kwargs):
        raise RuntimeError("挂了")

    monkeypatch.setattr(graph_module, "generate_answer", boom)
    result = app.invoke({"question": "第二个问题"}, config=THREAD)

    history = result["chat_history"]
    assert [message["role"] for message in history] == [
        "user", "assistant", "user", "assistant",
    ]
    assert history[-1]["content"] == ANSWER_FALLBACK
