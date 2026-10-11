"""
向量库契约测试。

契约定义在 ecom/knowledge/vector_store/base.py，两个后端必须表现得一模一样。
这里钉住的三件事，都是项目文档反复强调、而且**写错了很难发现**的：

  1. score 的方向 —— Chroma 要翻转，Milvus 不要（两个后端原生语义相反）
  2. add_chunks 按 source 幂等 —— 这是"重复跑 ingest.py 安全"的全部依据
  3. 空库 / top_k 越界不抛异常 —— 契约明写的，但 Chroma 的 n_results
     超限本身是会报错的，很容易漏掉夹取那一步

全部离线，用假客户端，详见 tests/fakes.py。
"""
import pytest

from ecom.knowledge.vector_store.base import VectorStore
from tests.knowledge.fakes import install_fake_chroma, install_fake_pymilvus

DIM = 1024


def _embeddings(count):
    """假向量。没有真模型在算，内容无所谓，长度对齐即可"""
    return [[0.0] * DIM for _ in range(count)]


def _chunks(specs):
    """specs 形如 [("售后政策.md", "第一段"), ...]"""
    return [
        {"text": text, "source": source, "chunk_id": index}
        for index, (source, text) in enumerate(specs)
    ]


# --------------------------------------------------------------------------
# 契约条款 2、3、4：两个后端表现必须一致（用 store_factory 参数化）
# --------------------------------------------------------------------------


def test_add_then_count(store_factory):
    store = store_factory()
    store.add_chunks(_chunks([("a.md", "x"), ("a.md", "y")]), _embeddings(2))
    assert store.count() == 2


def test_add_chunks_is_idempotent_per_source(store_factory):
    """同一个 source 灌两次，条数不翻倍"""
    store = store_factory()
    chunks = _chunks([("a.md", "x"), ("a.md", "y")])
    store.add_chunks(chunks, _embeddings(2))
    store.add_chunks(chunks, _embeddings(2))
    assert store.count() == 2


def test_shrinking_a_source_leaves_no_stale_chunks(store_factory):
    """文档变短时，旧 chunk 不能残留在库里"""
    store = store_factory()
    store.add_chunks(_chunks([("a.md", "x"), ("a.md", "y"), ("a.md", "z")]), _embeddings(3))
    store.add_chunks(_chunks([("a.md", "x")]), _embeddings(1))
    assert store.count() == 1


def test_rewriting_a_source_does_not_touch_other_sources(store_factory):
    """按 source 删除不能误伤别的文档"""
    store = store_factory()
    store.add_chunks(_chunks([("a.md", "x")]), _embeddings(1))
    store.add_chunks(_chunks([("b.md", "y")]), _embeddings(1))
    store.add_chunks(_chunks([("a.md", "x2")]), _embeddings(1))
    assert store.count() == 2


def test_length_mismatch_raises_before_any_delete(store_factory):
    """契约条款 1：长度不一致抛 ValueError

    而且要**在删除之前**抛——先删了再报错就等于无声地丢数据。
    """
    store = store_factory()
    store.add_chunks(_chunks([("a.md", "x")]), _embeddings(1))

    with pytest.raises(ValueError):
        store.add_chunks(_chunks([("a.md", "x"), ("a.md", "y")]), _embeddings(1))

    assert store.count() == 1  # 原来那条还在


def test_empty_store_search_returns_empty_list(store_factory):
    """契约条款 3：空库检索返回 []，不抛异常"""
    store = store_factory()
    assert store.search([0.0] * DIM, top_k=5) == []


def test_oversized_top_k_does_not_raise(store_factory):
    """契约条款 3：top_k 比库里条数大也不许抛异常"""
    store = store_factory()
    store.add_chunks(_chunks([("a.md", "x")]), _embeddings(1))
    assert len(store.search([0.0] * DIM, top_k=99)) == 1


def test_search_hit_has_exactly_the_contract_keys(store_factory):
    store = store_factory()
    store.add_chunks(_chunks([("a.md", "hello")]), _embeddings(1))

    hit = store.search([0.0] * DIM, top_k=1)[0]

    assert set(hit) == {"text", "source", "chunk_id", "score"}
    assert hit["text"] == "hello"
    assert hit["source"] == "a.md"
    assert hit["chunk_id"] == 0


def test_implements_vector_store_protocol(store_factory):
    """runtime_checkable 只检查方法齐不齐，不做签名校验"""
    assert isinstance(store_factory(), VectorStore)


# --------------------------------------------------------------------------
# 契约条款 5：all_chunks 全量枚举（语料还原、孤儿检测的入口）
# --------------------------------------------------------------------------


def test_all_chunks_returns_everything(store_factory):
    store = store_factory()
    store.add_chunks(
        _chunks([("a.md", "第一段"), ("a.md", "第二段"), ("b.md", "另一篇")]),
        _embeddings(3),
    )

    got = store.all_chunks()

    assert len(got) == 3
    assert {chunk["source"] for chunk in got} == {"a.md", "b.md"}
    # 不含向量也不含 score——还原语料只用得上正文和出处
    assert set(got[0]) == {"text", "source", "chunk_id"}
    assert {chunk["text"] for chunk in got} == {"第一段", "第二段", "另一篇"}


def test_all_chunks_on_empty_store_returns_empty_list(store_factory):
    assert store_factory().all_chunks() == []


# --------------------------------------------------------------------------
# 契约条款 6：delete_source —— 清理"库里有、盘上没有"的孤儿
#
# 这一条补得晚：add_chunks 的幂等只管得住"文件还在盘上"的情况，
# 文件被删掉之后，库里的 chunk 谁也动不了，检索却照样召回。
# --------------------------------------------------------------------------


def test_delete_source_removes_only_that_source(store_factory):
    store = store_factory()
    store.add_chunks(_chunks([("a.md", "x"), ("b.md", "y")]), _embeddings(2))

    store.delete_source("a.md")

    assert store.count() == 1
    assert {chunk["source"] for chunk in store.all_chunks()} == {"b.md"}


def test_delete_missing_source_is_a_noop(store_factory):
    """删一个不存在的 source 不能报错

    孤儿清理是照着"库里有"的名单去删的，中间天然会出现删不中的情况，
    这里一抛异常，整轮入库就断在最后一步了。
    """
    store = store_factory()
    store.add_chunks(_chunks([("a.md", "x")]), _embeddings(1))

    store.delete_source("从来没入过库.md")

    assert store.count() == 1


def test_deleting_every_source_empties_the_store(store_factory):
    """语料整体换掉时，库里不能剩下任何旧 chunk"""
    store = store_factory()
    store.add_chunks(_chunks([("a.md", "x"), ("b.md", "y")]), _embeddings(2))

    store.delete_source("a.md")
    store.delete_source("b.md")

    assert store.count() == 0
    assert store.all_chunks() == []
    assert store.search([0.0] * DIM, top_k=5) == []


def test_all_chunks_reflects_deletes(store_factory):
    """按 source 幂等删除后，all_chunks 不能还看得见旧 chunk"""
    store = store_factory()
    store.add_chunks(_chunks([("a.md", "x"), ("a.md", "y")]), _embeddings(2))
    store.add_chunks(_chunks([("a.md", "x")]), _embeddings(1))

    assert [chunk["text"] for chunk in store.all_chunks()] == ["x"]


def test_milvus_all_chunks_warns_when_hitting_query_limit(monkeypatch, caplog):
    """够到单次 query 上限时必须吭声

    Milvus 超出 limit 是静默截断的，不报错。还原语料时静默少一截，
    产出的就是一份看着完整、实际缺了尾巴的文档——比直接报错更危险。
    """
    module = install_fake_pymilvus(monkeypatch)
    monkeypatch.setattr(module, "QUERY_LIMIT", 2)  # 不用真造 16384 条

    store = module.MilvusStore()
    store.add_chunks(
        _chunks([("a.md", "1"), ("a.md", "2"), ("a.md", "3")]), _embeddings(3)
    )

    with caplog.at_level("WARNING"):
        got = store.all_chunks()

    assert len(got) == 2  # 真 Milvus 就是这样截断的
    assert "上限" in caplog.text


# --------------------------------------------------------------------------
# score 方向：两个后端原生语义相反，是整个抽象层最容易写错的一处
# --------------------------------------------------------------------------


def test_chroma_converts_cosine_distance_to_similarity(monkeypatch, tmp_path):
    """Chroma 的 cosine 返回**距离**（越小越相关），契约要的是相似度，必须翻转"""
    module = install_fake_chroma(monkeypatch, tmp_path)
    store = module.ChromaStore()
    store.add_chunks(_chunks([("a.md", "近"), ("a.md", "远")]), _embeddings(2))

    store.collection.next_distances = [0.1, 0.4]  # 0.1 更近 → 应该更相关
    hits = store.search([0.0] * DIM, top_k=2)

    assert hits[0]["score"] == pytest.approx(0.9)
    assert hits[1]["score"] == pytest.approx(0.6)
    assert hits[0]["score"] > hits[1]["score"]  # 距离小的那个分数更大


def test_milvus_keeps_cosine_similarity_as_is(monkeypatch):
    """Milvus 的 COSINE 返回的**已经是相似度**（越大越相关），绝不能再套 1-x

    这条一旦写反，排序会整体倒过来；而且症状很隐蔽（结果看着都像是对的），
    所以在这里用确定值钉死。ecom/knowledge/utils/eval.py 的 Recall 是最灵敏的探针，
    但那是要联网的集成测试，跑不了每次提交。
    """
    module = install_fake_pymilvus(monkeypatch)
    store = module.MilvusStore()
    store.add_chunks(_chunks([("a.md", "近"), ("a.md", "远")]), _embeddings(2))

    store.client.next_search_distances = [0.8, 0.3]
    hits = store.search([0.0] * DIM, top_k=2)

    assert hits[0]["score"] == pytest.approx(0.8)  # 不是 0.2
    assert hits[1]["score"] == pytest.approx(0.3)  # 不是 0.7
