"""
问题级向量缓存。

缓存只在这个项目里"允许存在"得有一种理由：它**不改变任何结果**。
所以这里钉的不是"快了多少"，而是三条契约：

  1. 逐字相同的问题才命中（不做近似）——命中就意味着向量必然一致；
  2. 返回的是副本——调用方改它不能污染缓存里那份；
  3. **失败不进缓存**——否则一次网络抖动会被记成"这个问题永远查不到"，
     而且和 retrieve() 不兜底、评测靠它暴露故障那条约定直接冲突。

不联网：底层的 get_embedding 被换成计数器。
"""
import pytest

from ecom.knowledge.embedding import embedder


@pytest.fixture
def counted(monkeypatch):
    """换掉真实 embedding，记录调用次数"""
    calls = []

    def fake(texts):
        calls.append(list(texts))
        return [[float(len(texts))] * 4]

    monkeypatch.setattr(embedder, "get_embedding", fake)
    embedder._query_embedding_cached.cache_clear()
    yield calls
    embedder._query_embedding_cached.cache_clear()


def test_same_question_hits_the_cache(counted):
    first = embedder.get_query_embedding("退货政策是什么")
    second = embedder.get_query_embedding("退货政策是什么")

    assert first == second
    assert len(counted) == 1, "同一个问题问了两次，底层却被调用了两次"


def test_different_questions_do_not_share(counted):
    embedder.get_query_embedding("退货政策是什么")
    embedder.get_query_embedding("怎么申请发票")

    assert len(counted) == 2
    assert counted[0] != counted[1], "两次问的文本不同，不该合成同一次调用"


def test_near_miss_does_not_hit(counted):
    """只差一个字就不算命中——这条就是"不做语义缓存"的机器化表述"""
    embedder.get_query_embedding("怎么退货")
    embedder.get_query_embedding("怎么换货")

    assert len(counted) == 2


def test_returned_vector_is_a_copy(counted):
    """调用方拿到的是副本：它要是原地改了，缓存里那份不能跟着变"""
    vector = embedder.get_query_embedding("退货政策是什么")
    vector[0] = 999.0

    again = embedder.get_query_embedding("退货政策是什么")
    assert again[0] != 999.0


def test_failure_is_not_cached(monkeypatch):
    """失败必须照常抛出，且不被缓存——下一次还得真的重试"""
    attempts = []

    def flaky(texts):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("网络抖了一下")
        return [[0.0] * 4]

    monkeypatch.setattr(embedder, "get_embedding", flaky)
    embedder._query_embedding_cached.cache_clear()

    with pytest.raises(RuntimeError):
        embedder.get_query_embedding("同一个问题")

    # 同一个问题再问一次：必须重新真正调用，而不是复用上次的失败
    assert embedder.get_query_embedding("同一个问题") == [0.0] * 4
    assert len(attempts) == 2

    embedder._query_embedding_cached.cache_clear()
