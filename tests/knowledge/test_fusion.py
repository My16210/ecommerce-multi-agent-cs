"""
RRF 融合。

融合是两路结果第一次碰头的地方，也是"排序被悄悄搞反"最难发现的地方：
输出永远是一份看着合理的列表，只是顺序错了。所以这里钉的是**性质**而不是实现——
尤其是"只看名次、不看分数大小"这一条，它正是当初选择 RRF 而不是归一化加权的全部理由。

纯函数，不碰 store、不碰向量库、不联网。
"""
import pytest

from ecom.knowledge.retriever.fusion import reciprocal_rank_fusion


def _hit(source, chunk_id, score):
    return {"text": f"{source}#{chunk_id}", "source": source, "chunk_id": chunk_id, "score": score}


def _keys(results):
    return [(item["source"], item["chunk_id"]) for item in results]


def test_document_matched_by_both_routes_outranks_single_route():
    """两路都捞到的，应该排在只被一路捞到的前面——RRF 最核心的性质"""
    vector = [_hit("a.md", 0, 0.9), _hit("a.md", 1, 0.8)]
    bm25 = [_hit("a.md", 1, 12.0), _hit("b.md", 5, 7.0)]

    results = reciprocal_rank_fusion({"vector": vector, "bm25": bm25})

    assert _keys(results)[0] == ("a.md", 1)


def test_rrf_ignores_score_magnitude():
    """把一路的原始分整体放大 1000 倍，融合顺序必须纹丝不动

    这条就是"为什么不用归一化加权"的可执行版本：RRF 只读名次，
    所以两路分数根本不在一个量纲上也无所谓。
    """
    vector = [_hit("a.md", 0, 0.90), _hit("a.md", 1, 0.80)]
    bm25 = [_hit("a.md", 1, 12.0), _hit("b.md", 5, 7.0)]

    normal = reciprocal_rank_fusion({"vector": vector, "bm25": bm25})
    inflated = reciprocal_rank_fusion({
        "vector": [_hit("a.md", 0, 900.0), _hit("a.md", 1, 800.0)],
        "bm25": [_hit("a.md", 1, 12000.0), _hit("b.md", 5, 7000.0)],
    })

    assert _keys(normal) == _keys(inflated)
    assert [item["score"] for item in normal] == pytest.approx(
        [item["score"] for item in inflated]
    )


def test_rank_one_contribution_equals_one_over_k_plus_one():
    """钉住公式本身：单路第一名拿到的就是 1/(K+1)"""
    results = reciprocal_rank_fusion({"vector": [_hit("a.md", 0, 0.5)]}, k=60)

    assert results[0]["score"] == pytest.approx(1 / 61)


def test_chunk_present_in_only_one_route_still_scores():
    """只在 BM25 里出现的 chunk 不能丢——它在向量路上没有名次，但有分"""
    vector = [_hit("a.md", 0, 0.9)]
    bm25 = [_hit("b.md", 5, 7.0)]

    results = reciprocal_rank_fusion({"vector": vector, "bm25": bm25})
    only_bm25 = next(item for item in results if item["source"] == "b.md")

    assert only_bm25["score"] > 0
    assert only_bm25["bm25_rank"] == 1
    assert "vector_rank" not in only_bm25


def test_fused_hit_keeps_both_routes_details():
    """两路的原始分和名次都要带出来，UI 要靠它显示"这条是哪一路捞上来的\""""
    vector = [_hit("a.md", 0, 0.9), _hit("a.md", 1, 0.8)]
    bm25 = [_hit("a.md", 1, 12.0)]

    both = next(
        item for item in reciprocal_rank_fusion({"vector": vector, "bm25": bm25})
        if item["chunk_id"] == 1
    )

    assert both["vector_score"] == pytest.approx(0.8)
    assert both["vector_rank"] == 2
    assert both["bm25_score"] == pytest.approx(12.0)
    assert both["bm25_rank"] == 1


def test_ties_are_broken_deterministically():
    """两路名次互换时会撞出完全相同的融合分，破平必须稳定

    破平靠 dict 遍历顺序的话，同一个查询每次跑出来的顺序都可能不同——
    eval 的命中位次会跟着抖，那种"偶尔变差"的 bug 极难追。
    """
    vector = [_hit("b.md", 9, 0.9)]
    bm25 = [_hit("a.md", 3, 5.0)]

    first = reciprocal_rank_fusion({"vector": vector, "bm25": bm25})
    second = reciprocal_rank_fusion({"bm25": bm25, "vector": vector})

    assert first[0]["score"] == pytest.approx(second[0]["score"])
    assert _keys(first) == _keys(second) == [("a.md", 3), ("b.md", 9)]


def test_duplicate_chunk_within_one_route_is_counted_once():
    """同一路里出现重复的 chunk，融合分不能翻倍

    正常的 search() 不会返回重复，但融合是两路合流的边界，
    一旦有脏数据，症状（某条莫名排第一）非常难回溯。
    """
    duplicated = [_hit("a.md", 0, 0.9), _hit("a.md", 0, 0.9)]
    results = reciprocal_rank_fusion({"vector": duplicated})

    assert len(results) == 1
    assert results[0]["score"] == pytest.approx(1 / 61)


def test_empty_input_returns_empty():
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"vector": [], "bm25": []}) == []


# --------------------------------------------------------------------------
# 两路权重
# --------------------------------------------------------------------------


def test_no_weights_equals_all_ones():
    """不给权重 == 全 1.0，逐位一致

    这条是权重功能"纯增量"的保证：加了参数之后，默认行为必须和加之前
    一模一样，否则所有已记录的历史评测数字都作废了。
    """
    vector = [_hit("a.md", 0, 0.9), _hit("a.md", 1, 0.8)]
    bm25 = [_hit("a.md", 1, 12.0), _hit("b.md", 5, 7.0)]

    implicit = reciprocal_rank_fusion({"vector": vector, "bm25": bm25})
    explicit = reciprocal_rank_fusion(
        {"vector": vector, "bm25": bm25}, weights={"vector": 1.0, "bm25": 1.0}
    )

    assert _keys(implicit) == _keys(explicit)
    assert [item["score"] for item in implicit] == pytest.approx(
        [item["score"] for item in explicit]
    )


def test_missing_route_in_weights_defaults_to_one():
    """weights 里没提到的路按 1.0 算，不能变成 0（那就等于把这一路整个丢了）"""
    vector = [_hit("a.md", 0, 0.9)]
    bm25 = [_hit("b.md", 5, 7.0)]

    weighted = reciprocal_rank_fusion({"vector": vector, "bm25": bm25}, weights={"bm25": 2.0})
    only_vector = next(item for item in weighted if item["source"] == "a.md")

    assert only_vector["score"] == pytest.approx(1 / 61)


def test_scaling_all_weights_equally_changes_nothing():
    """整体缩放不改变顺序——融合分只在同一题内比大小，绝对尺度没有意义

    这也说明**只有比例有意义**，所以 tune_rrf.py 固定向量权重为 1.0、
    只在比例上搜，是等价的做法，不是偷懒。
    """
    vector = [_hit("a.md", 0, 0.9), _hit("a.md", 1, 0.8)]
    bm25 = [_hit("a.md", 1, 12.0), _hit("b.md", 5, 7.0)]

    base = reciprocal_rank_fusion({"vector": vector, "bm25": bm25})
    scaled = reciprocal_rank_fusion(
        {"vector": vector, "bm25": bm25}, weights={"vector": 7.0, "bm25": 7.0}
    )

    assert _keys(base) == _keys(scaled)


def test_boosting_bm25_promotes_its_top_hit():
    """把 BM25 权重抬上去，它那一路的第一名要能压过向量路的第一名

    这是加权唯一的作用：**挪动两路之间的话语权**。它不制造新信息，
    只是把边界推向某一边——所以术语型涨、改写型就得跌，总量不会因此变多。
    """
    vector = [_hit("a.md", 0, 0.99)]     # 向量路第一
    bm25 = [_hit("z.md", 0, 20.0)]       # BM25 路第一，另一条

    # 等权：两条各拿 1/61，撞车，破平按来源名升序 → a.md 在前
    equal = reciprocal_rank_fusion({"vector": vector, "bm25": bm25})
    assert _keys(equal)[0] == ("a.md", 0)

    # BM25 权重加倍：z.md 变成 2/61 > 1/61，压过去
    heavy = reciprocal_rank_fusion({"vector": vector, "bm25": bm25},
                                   weights={"vector": 1.0, "bm25": 2.0})
    assert _keys(heavy)[0] == ("z.md", 0)


def test_weight_one_contribution_equals_weight_over_k_plus_one():
    """钉住加权后的公式：单路第一名的得分是 w/(K+1)，不是 1/(K+1)

    "任一路第 1 名恒为 w/(K+1)、与 K 无关"正是"调 K 动不了两路话语权"的根据。
    """
    results = reciprocal_rank_fusion(
        {"bm25": [_hit("a.md", 0, 5.0)]}, k=60, weights={"bm25": 3.0}
    )

    assert results[0]["score"] == pytest.approx(3 / 61)
