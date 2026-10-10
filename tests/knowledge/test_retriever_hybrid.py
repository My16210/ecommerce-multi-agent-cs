"""
混合检索的编排与降级。

这个文件守的最重要的一条是**降级的不对称性**：
  - 向量那一路失败必须抛出去
  - BM25 那一路失败只退回纯向量

写反了不会报错，只会让 eval.py 在 embedding 全挂时把每道题都印成"未命中"，
把真正的服务故障伪装成"检索质量差"——那正是这个项目里最贵的一类 bug。
剩下的几条守住"关掉混合检索就等于完全回到旧行为"，否则这个开关就是个谎。

不联网、不碰向量库：get_embedding 和 bm25_search 都被替换掉了。
"""
import pytest

import ecom.knowledge.retriever.retriever as retriever


class _StubStore:
    """只实现 retrieve 用到的 search（向量那一路）"""

    def __init__(self, hits, fail=False):
        self.hits = hits
        self.fail = fail
        self.asked_top_k = None

    def search(self, query_embedding, top_k=5):
        if self.fail:
            raise RuntimeError("embedding 服务挂了")
        self.asked_top_k = top_k
        return [dict(hit) for hit in self.hits[:top_k]]


def _hit(chunk_id, score=0.9, source="a.md"):
    return {"text": f"第{chunk_id}段", "source": source, "chunk_id": chunk_id, "score": score}


@pytest.fixture
def wired(monkeypatch):
    """把 embedding 换成假的，返回一个"接好线"的辅助对象"""
    monkeypatch.setattr(retriever, "get_query_embedding", lambda text: [0.0] * 4)

    def keyword_hits(hits):
        monkeypatch.setattr(retriever, "bm25_search", lambda store, q, top_k: hits)

    def keyword_boom(exc):
        def boom(store, q, top_k):
            raise exc
        monkeypatch.setattr(retriever, "bm25_search", boom)

    return keyword_hits, keyword_boom


# --------------------------------------------------------------------------
# 开关关掉时：必须与旧行为逐字节一致
# --------------------------------------------------------------------------


def test_hybrid_off_returns_exact_old_shape(monkeypatch, wired):
    """HYBRID_SEARCH=false 时，返回的字段集合要和混合检索上线前一模一样

    这个开关是 A/B 对比和回退基线的唯一手段，它要是"关不干净"（比如仍然
    多带出 vector_rank），对比出来的结论就是错的。
    """
    monkeypatch.setattr(retriever, "HYBRID_SEARCH", False)
    store = _StubStore([_hit(0, 0.88), _hit(1, 0.77)])

    results = retriever.retrieve("随便问问", store, top_k=2)

    assert all(set(hit) == {"text", "source", "chunk_id", "score"} for hit in results)
    assert [hit["score"] for hit in results] == [0.88, 0.77]


def test_hybrid_off_does_not_widen_recall(monkeypatch, wired):
    """关掉时不该多召回——直接就是 store.search(top_k)"""
    monkeypatch.setattr(retriever, "HYBRID_SEARCH", False)
    store = _StubStore([_hit(i) for i in range(30)])

    results = retriever.retrieve("随便问问", store, top_k=5)

    assert store.asked_top_k == 5
    assert len(results) == 5


# --------------------------------------------------------------------------
# 降级：两路的不对称
# --------------------------------------------------------------------------


def test_vector_failure_propagates(monkeypatch, wired):
    """向量这一路失败必须抛出去——eval.py 靠它暴露 embedding 故障

    这里若改成"失败就返回空列表"，评测会把每道题都印成"未命中"，
    看起来像检索质量差，其实是服务挂了。这是全文件最重要的一条。
    """
    monkeypatch.setattr(retriever, "HYBRID_SEARCH", True)
    store = _StubStore([_hit(0)], fail=True)

    with pytest.raises(RuntimeError, match="embedding"):
        retriever.retrieve("随便问问", store, top_k=3)


def test_bm25_failure_degrades_to_vector_only(monkeypatch, wired):
    """BM25 只是增强，它挂了不能拖垮整条检索"""
    _, keyword_boom = wired
    monkeypatch.setattr(retriever, "HYBRID_SEARCH", True)
    keyword_boom(RuntimeError("jieba 没装"))
    store = _StubStore([_hit(i, 0.9 - i * 0.1) for i in range(8)])

    results = retriever.retrieve("随便问问", store, top_k=3)

    assert [hit["chunk_id"] for hit in results] == [0, 1, 2]
    # 降级路径不额外挂明细字段——它就是纯粹的向量结果
    assert all("vector_rank" not in hit for hit in results)


def test_empty_bm25_result_is_safe(monkeypatch, wired):
    """BM25 一条都没召回（词一个都没对上）时退回纯向量，不抛"""
    keyword_hits, _ = wired
    monkeypatch.setattr(retriever, "HYBRID_SEARCH", True)
    keyword_hits([])
    store = _StubStore([_hit(i) for i in range(4)])

    results = retriever.retrieve("随便问问", store, top_k=2)

    assert [hit["chunk_id"] for hit in results] == [0, 1]


# --------------------------------------------------------------------------
# 融合路径
# --------------------------------------------------------------------------


def test_fused_hit_carries_vector_and_bm25_fields(monkeypatch, wired):
    keyword_hits, _ = wired
    monkeypatch.setattr(retriever, "HYBRID_SEARCH", True)
    store = _StubStore([_hit(0, 0.90), _hit(1, 0.80)])
    keyword_hits([_hit(1, 12.0), _hit(10, 7.0)])

    results = retriever.retrieve("随便问问", store, top_k=5)

    both = next(hit for hit in results if hit["chunk_id"] == 1)
    assert both["vector_rank"] == 2
    assert both["bm25_rank"] == 1
    assert both["vector_score"] == pytest.approx(0.80)
    assert both["bm25_score"] == pytest.approx(12.0)

    only_bm25 = next(hit for hit in results if hit["chunk_id"] == 10)
    assert "vector_rank" not in only_bm25


def test_fused_result_is_truncated_to_top_k(monkeypatch, wired):
    """两路各召回一批，融合后的条数可能远超 top_k，必须截回来

    retrieve() 返回正好 top_k 条这个语义不能破：eval 的 Recall@5 和命中位次
    都建立在"返回的就是最终 top-5"上面。
    """
    keyword_hits, _ = wired
    monkeypatch.setattr(retriever, "HYBRID_SEARCH", True)
    store = _StubStore([_hit(i) for i in range(3)])
    keyword_hits([_hit(10 + i) for i in range(3)])

    results = retriever.retrieve("随便问问", store, top_k=2)

    assert len(results) == 2


def test_recall_width_is_wider_than_top_k(monkeypatch, wired):
    """每路先按 RECALL_K 召回再融合，不能只取 top_k 条去融合

    只取 5 条融合的话，BM25 命中但排在向量第 8 位的那条根本没机会进候选，
    混合检索就白做了。
    """
    keyword_hits, _ = wired
    monkeypatch.setattr(retriever, "HYBRID_SEARCH", True)
    store = _StubStore([_hit(i) for i in range(50)])
    keyword_hits([_hit(40)])

    retriever.retrieve("随便问问", store, top_k=5)

    assert store.asked_top_k == retriever.VECTOR_RECALL_K
    assert store.asked_top_k > 5
