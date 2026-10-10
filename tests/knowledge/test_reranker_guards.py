"""
rerank 对外部返回值的防御。

index 是接口给的，属于不可信输入：
  越界正数 -> IndexError，整条链路被打断
  负数     -> **静默**取到尾部 chunk，结果看着正常但是错的（最隐蔽的一种）
  true     -> JSON 的 true 是 bool，而 bool 是 int 的子类，会骗过 isinstance 检查

三种都必须被丢掉，并且丢光之后还能退化成"按向量分数截断"，
而不是留下一个空的上下文列表。
"""
import pytest

from ecom.knowledge.retriever import reranker


@pytest.fixture
def contexts():
    return [
        {"text": "第一段", "source": "a.md", "chunk_id": 0, "score": 0.5},
        {"text": "第二段", "source": "a.md", "chunk_id": 1, "score": 0.4},
        {"text": "第三段", "source": "a.md", "chunk_id": 2, "score": 0.3},
    ]


def _stub_api(monkeypatch, ranked):
    """替掉接口调用，直接喂进想测的 index"""
    monkeypatch.setattr(reranker, "_rerank_by_api", lambda question, texts: ranked)


def test_valid_results_pass_through(monkeypatch, contexts):
    _stub_api(monkeypatch, [{"index": 2, "score": 0.9}, {"index": 0, "score": 0.7}])
    result = reranker.rerank("q", contexts, top_k=2)
    assert [item["chunk_id"] for item in result] == [2, 0]


def test_out_of_range_index_is_dropped(monkeypatch, contexts):
    _stub_api(monkeypatch, [{"index": 0, "score": 0.9}, {"index": 99, "score": 0.8}])
    result = reranker.rerank("q", contexts, top_k=3)
    assert [item["chunk_id"] for item in result] == [0]


def test_negative_index_is_dropped(monkeypatch, contexts):
    """-1 会静默命中最后一段，必须丢掉；丢光后退化为按向量分数截断"""
    _stub_api(monkeypatch, [{"index": -1, "score": 0.9}])
    result = reranker.rerank("q", contexts, top_k=3)
    assert [item["chunk_id"] for item in result] == [0, 1, 2]


def test_bool_index_is_dropped(monkeypatch, contexts):
    """JSON 里的 true 是 bool，bool 又是 int 的子类，会骗过 isinstance 检查"""
    _stub_api(monkeypatch, [{"index": True, "score": 0.9}])
    result = reranker.rerank("q", contexts, top_k=3)
    assert [item["chunk_id"] for item in result] == [0, 1, 2]


def test_non_int_index_is_dropped(monkeypatch, contexts):
    _stub_api(monkeypatch, [{"index": "0", "score": 0.9}])
    result = reranker.rerank("q", contexts, top_k=3)
    assert [item["chunk_id"] for item in result] == [0, 1, 2]


def test_partial_invalid_keeps_the_good_ones(monkeypatch, contexts):
    _stub_api(monkeypatch, [
        {"index": 1, "score": 0.9},
        {"index": 99, "score": 0.8},
        {"index": -1, "score": 0.7},
    ])
    result = reranker.rerank("q", contexts, top_k=3)
    assert [item["chunk_id"] for item in result] == [1]


def test_missing_index_key_is_dropped(monkeypatch, contexts):
    _stub_api(monkeypatch, [{"score": 0.9}])
    result = reranker.rerank("q", contexts, top_k=3)
    assert [item["chunk_id"] for item in result] == [0, 1, 2]
