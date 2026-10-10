"""
手写 BM25 的正确性。

为什么这个模块值得有测试：**BM25 算错了不会报错，只会悄悄给出一个更差的排序。**
没有测试的话，"召回率好像变低了"会变成一句无法追查的感觉——而这类"坏结果不喊疼"
的缺陷，正是这个项目反复吃过的亏（语料那次也是）。

另外这里钉住了两个容易被"顺手优化掉"的地方：
  - idf 用的是 `ln(1 + ...)` 那一支（不加 1 会对高频词给出负分）
  - 索引缓存必须跟着 `store.count()` 失效（Streamlit 是长驻进程）

全部离线：jieba 是本机装的，不联网；假 store 就是个内存列表。
"""
import pytest

from ecom.knowledge.retriever.bm25 import (
    BM25Index,
    get_index,
    idf,
    reset_index,
    term_score,
    tokenize,
)


@pytest.fixture(autouse=True)
def _clean_index_cache():
    """索引缓存是模块级的，不隔离的话用例之间会互相看到对方的库"""
    reset_index()
    yield
    reset_index()


def _chunk(source, chunk_id, text):
    return {"text": text, "source": source, "chunk_id": chunk_id}


class _StubStore:
    """只实现 bm25.py 真正调用的两个方法（count / all_chunks）"""

    def __init__(self, chunks=None):
        self.chunks = list(chunks or [])

    def count(self):
        return len(self.chunks)

    def all_chunks(self):
        return list(self.chunks)


# --------------------------------------------------------------------------
# tokenize
# --------------------------------------------------------------------------


def test_tokenize_lowercases_latin_and_keeps_chinese():
    """拉丁词要转小写，否则"提问写 SQLiteSaver、正文写 SqliteSaver"就匹配不上"""
    tokens = tokenize("LangGraph 的 Checkpointer 怎么用")
    assert "langgraph" in tokens
    assert "checkpointer" in tokens
    assert "的" in tokens


def test_tokenize_drops_punctuation_and_whitespace():
    """标点和空格对打分毫无意义，却会被算进文档长度，把长度归一化带偏"""
    tokens = tokenize("甲，乙。 丙\n丁")
    assert all(any(ch.isalnum() for ch in token) for token in tokens)
    assert "，" not in tokens
    assert "。" not in tokens


# --------------------------------------------------------------------------
# idf / term_score
# --------------------------------------------------------------------------


def test_idf_stays_positive_for_very_common_terms():
    """加 1 的那一支在词出现在超过一半文档时仍然是正的

    换成 ln((N-n+0.5)/(n+0.5))，这里会得到负数——中文的"的/是/在"几乎必然触发，
    那些停用词就会反过来把文档的分数往下拉。
    """
    assert idf(100, 90) > 0
    assert idf(100, 99) > 0
    # 再罕见的词得分更高，单调性也要在
    assert idf(100, 1) > idf(100, 50) > idf(100, 99)


def test_term_frequency_saturates():
    """词频不是线性的：出现 3 次比 1 次相关，但绝不会是 3 倍"""
    once = term_score(1, 100, 100, 1.0, 1.5, 0.75)
    thrice = term_score(3, 100, 100, 1.0, 1.5, 0.75)
    assert thrice > once
    assert thrice < 3 * once


def test_length_normalization_penalizes_longer_document():
    """b 管长度归一化；b=0 时长度就完全不影响打分了"""
    short = term_score(2, 50, 100, 1.0, 1.5, 0.75)
    long_ = term_score(2, 200, 100, 1.0, 1.5, 0.75)
    assert long_ < short

    assert term_score(2, 50, 100, 1.0, 1.5, 0.0) == term_score(2, 200, 100, 1.0, 1.5, 0.0)


def test_term_score_survives_an_empty_corpus():
    """空库时 avgdl 是 0，不能除出 ZeroDivisionError"""
    assert term_score(1, 0, 0.0, 1.0, 1.5, 0.75) == 0.0


# --------------------------------------------------------------------------
# BM25Index.search
# --------------------------------------------------------------------------


def test_rare_term_beats_common_term():
    """罕见词命中的那篇要排在前面——这就是 idf 的全部意义"""
    index = BM25Index([
        _chunk("a.md", 0, "常见词 内容一"),
        _chunk("a.md", 1, "常见词 罕见词 内容二"),
        _chunk("a.md", 2, "常见词 内容三"),
    ])
    results = index.search("常见词 罕见词", top_k=3)
    assert results[0]["chunk_id"] == 1


def test_search_returns_hits_in_the_contract_shape():
    """返回的 dict 要和 store.search 的 Hit 同形——融合层按这四个键合并"""
    index = BM25Index([_chunk("a.md", 0, "苹果 香蕉")])
    hit = index.search("苹果", top_k=1)[0]
    assert set(hit) == {"text", "source", "chunk_id", "score"}


def test_search_respects_top_k_and_is_sorted():
    index = BM25Index([
        _chunk("a.md", 0, "苹果"),
        _chunk("a.md", 1, "苹果 苹果 苹果"),
        _chunk("a.md", 2, "苹果 香蕉 橙子"),
    ])
    results = index.search("苹果", top_k=2)
    assert len(results) == 2
    assert results[0]["score"] >= results[1]["score"]


def test_search_ignores_chunks_with_no_matching_term():
    """一个词都没匹配上就是没命中

    不过滤的话，score=0 的一堆 chunk 会被当成"召回了"，看着正常其实全是噪声。
    """
    index = BM25Index([_chunk("a.md", 0, "苹果"), _chunk("a.md", 1, "香蕉")])
    assert index.search("苹果", top_k=10) == index.search("苹果", top_k=10)
    assert [hit["chunk_id"] for hit in index.search("苹果", top_k=10)] == [0]


def test_empty_query_and_empty_index_return_empty():
    assert BM25Index([]).search("随便什么", top_k=5) == []
    index = BM25Index([_chunk("a.md", 0, "苹果")])
    assert index.search("", top_k=5) == []
    assert index.search("毫无关系的词", top_k=5) == []
    assert index.search("苹果", top_k=0) == []


def test_index_orders_chunks_so_ties_are_deterministic():
    """all_chunks 不保证顺序（契约条款 5），索引必须自己排

    否则同分时谁在前取决于库的存储顺序，eval 的命中位次会随机抖动。
    """
    index = BM25Index([
        _chunk("b.md", 9, "苹果"),
        _chunk("a.md", 3, "苹果"),
    ])
    results = index.search("苹果", top_k=2)
    assert [(hit["source"], hit["chunk_id"]) for hit in results] == [("a.md", 3), ("b.md", 9)]


# --------------------------------------------------------------------------
# 缓存
# --------------------------------------------------------------------------


def test_index_is_cached_until_store_count_changes():
    """长驻进程里索引陈旧的唯一防线

    Streamlit 是长驻进程，ingest.py 在另一个进程跑，缓存不会自己感知语料变化。
    按 count 失效能覆盖"新增了 reviewed 文档"这个主路径。
    """
    store = _StubStore([_chunk("a.md", 0, "苹果")])
    first = get_index(store)
    assert get_index(store) is first          # 同一个 key，命中缓存

    store.chunks.append(_chunk("a.md", 1, "香蕉"))
    second = get_index(store)
    assert second is not first                # count 变了 → 重建
    assert len(second) == 2


def test_reset_index_drops_the_cache():
    store = _StubStore([_chunk("a.md", 0, "苹果")])
    first = get_index(store)
    reset_index()
    assert get_index(store) is not first
