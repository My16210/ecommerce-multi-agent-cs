"""
多路检索结果的融合（RRF）

为什么是 RRF，而不是"归一化后加权求和"
--------------------------------------
两路的分数**根本不是一个量纲**：向量路是余弦相似度，通常挤在 0.3~0.7 这一小段；
BM25 路是没有上界的原始分，中文短块常见 0~20+。想让它们相加，就得先把两路都拉到
同一个尺度上，于是很自然地写出 min-max 归一化：

    score_norm = (score - min) / (max - min)

问题出在 min 和 max 是**本次查询的候选集**里算出来的。某一题只召回一条时，
那一条会被归一化抬到 1.0；另一题召回二十条时分布完全不同。也就是说，
两路之间的相对权重**每一题都在漂移**，还得再拍一个 α 去平衡它们。
而且余弦相似度本身不是概率，归一化之后也不能解释成"相关程度"。

RRF（Reciprocal Rank Fusion，Cormack et al. 2009）绕开了整件事：**只用名次**，
完全不管分数是多少。

    RRF(d) = Σ_r 1 / (K + rank_r(d))        rank 从 1 开始

名次天然就是同一个尺度，两路可以直接相加，也没有任何参数需要调。
这就是它成为混合检索默认做法的原因。

两路权重（`weights`）—— 以及为什么调 K 不是同一个东西
------------------------------------------------------
等权 RRF 有个被忽略的性质：**任一路的第 1 名，得分恒为 1/(K+1)**。它和 K 的取值
无关，也和这一路召回多宽无关。也就是说，两路之间那个"相对话语权"是个常数，
**调 K 或调召回宽度都动不了它**——K 只是把靠前几条之间的差距一起压平。

要真正改变两路的话语权，只能显式加权：

    RRF(d) = Σ_r w_r / (K + rank_r(d))

`weights=None` 时所有 w_r=1，输出与加权重之前**逐位一致**（有测试钉住），
所以这个参数是纯增量：不传就是旧行为。

**默认不启用权重**（settings 里两路都是 1.0）。理由见 README「为什么默认不加权」：
在留出集上，调出来的权重没有稳定收益，倒更像是拟合了调参那一半的噪声。
"""
from ecom.knowledge.settings import RRF_K
from ecom.knowledge.utils.logger import setup_logger

logger = setup_logger(__name__)


def reciprocal_rank_fusion(rankings: dict, k: int = RRF_K, weights: dict = None) -> list:
    """
    把多路召回按名次融合成一路。

    rankings 形如 {"vector": [...], "bm25": [...]}，每一路都要**已经按相关度降序排好**
    （也就是各路自己的 Top-N）。用 dict 而不是 list，是为了让产出的明细字段
    （`vector_rank` / `bm25_rank`）能直接叫出正确名字，而不是靠"第 0 路是向量"这种约定。

    weights 形如 {"vector": 1.0, "bm25": 2.0}；缺省的路按 1.0 算，None 即全 1.0。

    返回融合后的完整列表，按融合分降序。每个元素是：

        {"text", "source", "chunk_id",
         "score",                        # 融合分（RRF），越大越相关
         "<路名>_score", "<路名>_rank"}   # 每一路的原始分与名次，该路没召回时缺省

    注意 `score` 在这里**不再是余弦相似度**，而是一个 RRF 分（单路第一名约 w/(K+1)）。
    它保持"越大越相关"的单调性，所以排序语义没变；但绝对值没有物理意义，
    要看真实的相似度请读 `vector_score`。
    """
    weights = weights or {}
    fused = {}

    for route, hits in (rankings or {}).items():
        weight = weights.get(route, 1.0)
        seen = set()
        for rank, hit in enumerate(hits or [], start=1):
            key = (hit.get("source", ""), hit.get("chunk_id", -1))

            # 同一路里出现重复的 chunk 会把 w/(K+rank) 加两次，融合分凭空翻倍。
            # 正常的 search() 不会返回重复，但这里是两路合并的边界，
            # 一旦有脏数据，症状（某条莫名排第一）非常难回溯，所以直接挡掉。
            if key in seen:
                continue
            seen.add(key)

            entry = fused.get(key)
            if entry is None:
                entry = {
                    "text": hit.get("text", ""),
                    "source": key[0],
                    "chunk_id": key[1],
                    "score": 0.0,
                }
                fused[key] = entry

            entry["score"] += weight / (k + rank)
            entry[f"{route}_score"] = hit.get("score")
            entry[f"{route}_rank"] = rank

    # 破平按 (source, chunk_id)：RRF 分撞车完全可能（两路名次互换成对出现），
    # 而 dict 的遍历顺序不该泄漏到结果里——否则 eval 的命中位次会随机抖动。
    return sorted(
        fused.values(), key=lambda entry: (-entry["score"], entry["source"], entry["chunk_id"])
    )


if __name__ == "__main__":
    vector_hits = [
        {"text": "甲", "source": "a.md", "chunk_id": 0, "score": 0.82},
        {"text": "乙", "source": "a.md", "chunk_id": 1, "score": 0.71},
    ]
    bm25_hits = [
        {"text": "乙", "source": "a.md", "chunk_id": 1, "score": 12.4},
        {"text": "丙", "source": "b.md", "chunk_id": 3, "score": 7.1},
    ]
    for item in reciprocal_rank_fusion({"vector": vector_hits, "bm25": bm25_hits}):
        print(f"[{item['score']:.4f}] {item['source']}#{item['chunk_id']}  {item}")
