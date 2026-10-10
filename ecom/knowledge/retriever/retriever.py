"""
检索模块
功能：用户提问后，检索相关chunk

第一阶段：纯向量检索
第二阶段：加BM25混合检索  ← 现在这一步

混合检索长这样：

    向量检索 Top-20  ─┐
                      ├─→ RRF 融合 ─→ Top-K ─→ （交给 rerank）
    BM25 检索 Top-20 ─┘

为什么要在"检索"这一层融合，而不是换个向量库
--------------------------------------------
两个后端的原生能力不对等：Milvus 2.5+ 有内置 BM25，chromadb 1.5 本地版完全没有。
所以 BM25 是自己实现的（ecom/knowledge/retriever/bm25.py），只依赖契约里的 `all_chunks()`，
底下是 Chroma 还是 Milvus 它不知道也不关心。详见 FLOWCHART.md 那一节。

返回值的字段
------------
比契约里的 `Hit` 多几个：`score` 变成**融合分**（RRF），另外带上每一路的原始分和名次
（`vector_score`/`bm25_score`/`vector_rank`/`bm25_rank`），没被某一路召回的 chunk
就没有那一路的字段。这几个字段**不能**加进 base.py 的 `Hit`——契约只管存储层，
而契约测试钉死了 `search()` 的返回恰好是四个键。
"""
from ecom.knowledge.embedding.embedder import get_query_embedding
from ecom.knowledge.retriever.bm25 import bm25_search
from ecom.knowledge.retriever.fusion import reciprocal_rank_fusion
from ecom.knowledge.settings import (
    BM25_RECALL_K,
    HYBRID_SEARCH,
    RRF_K,
    RRF_WEIGHT_BM25,
    RRF_WEIGHT_VECTOR,
    TOP_K,
    VECTOR_RECALL_K,
)
from ecom.knowledge.utils.logger import setup_logger
from ecom.knowledge.utils.timing import stage
from ecom.knowledge.vector_store import get_vector_store
from ecom.knowledge.vector_store.base import Hit, VectorStore

logger = setup_logger(__name__)


class RetrievalHit(Hit, total=False):
    """
    retrieve() 的返回单元 = 契约的 Hit + 融合明细。

    继承 Hit 而不是另写一遍，是为了让"这是 Hit 的超集"这件事写在类型里：
    下游（rerank / answer）照着 Hit 读就够，多出来的字段只有 UI 和调试会看。
    """

    vector_score: float      # 这一条在向量路上的余弦相似度
    bm25_score: float        # 这一条在 BM25 路上的原始分
    vector_rank: int         # 在向量路里的名次（从 1 起）
    bm25_rank: int           # 在 BM25 路里的名次（从 1 起）


def route_rankings(question: str, store: VectorStore) -> tuple:
    """
    跑两路召回，返回 (向量榜, BM25 榜)，**都还没融合**。

    单独抽出来，是因为"两路是怎么来的"必须只有一份定义：retrieve() 用它，
    tune_rrf.py 也要用它——调权重需要在**同一批原始名次**上反复融合。
    调参脚本要是自己再拼一遍 embedding + search，万一哪天 retrieve() 改了召回宽度
    或换了分词，脚本还在按旧定义算，调出来的权重就落不到生产上。

    embedding 失败直接抛（见 retrieve 的降级语义说明）。
    问题向量走 get_query_embedding，同一问题重复问时这一段的耗时和钱都是 0。
    """
    with stage("embed"):
        query_vector = get_query_embedding(question)

    with stage("vector_search"):
        vector_hits = store.search(query_vector, top_k=VECTOR_RECALL_K)

    try:
        with stage("bm25"):
            keyword_hits = bm25_search(store, question, top_k=BM25_RECALL_K)
    except Exception as exc:
        # jieba 没装、索引构建异常、库里的数据有问题……都走这里
        logger.warning("BM25 检索不可用，本轮退化为纯向量检索：%s", exc)
        keyword_hits = []

    return vector_hits, keyword_hits


def retrieve(question: str, store: VectorStore, top_k: int = TOP_K) -> list:
    """
    根据用户问题检索相关chunk

    返回：[{"text","source","chunk_id","score", ...}, ...]，按 score 降序。

    降级语义（很重要，别改）：
      - **向量这一路失败必须抛出去**。graph.py 的 retrieve_node 有兜底，
        而 eval.py 恰恰靠 retrieve() 不兜底来暴露 embedding 故障——
        如果这里包一层 try，评测在 embedding 全挂时会把每道题都印成"未命中"，
        把真正的故障盖住。
      - **BM25 这一路失败只降级**（退回纯向量）。它是个增强，不该拖垮检索。
    """
    if not HYBRID_SEARCH:
        # 纯向量这条路不经过 route_rankings：它要的是 top_k 条，
        # 而不是"召回 20 条再融合截断"，两者的召回宽度不一样。
        with stage("embed"):
            query_vector = get_query_embedding(question)
        with stage("vector_search"):
            hits = store.search(query_vector, top_k=top_k)
        logger.info("问题 %r 检索到 %d 个 chunk（纯向量）", question, len(hits))
        return hits

    vector_hits, keyword_hits = route_rankings(question, store)

    if not keyword_hits:
        # 这一路的召回为空（词一个都没对上，或上面降级了）。
        # 退回的这几条是纯 Hit，没有 vector_rank 之类的明细字段——读取方用 .get()。
        hits = vector_hits[:top_k]
        logger.info("问题 %r 检索到 %d 个 chunk（纯向量）", question, len(hits))
        return hits

    with stage("fusion"):
        hits = reciprocal_rank_fusion(
            {"vector": vector_hits, "bm25": keyword_hits},
            k=RRF_K,
            # 默认两路都是 1.0 = 不加权（与加权重之前逐位一致）。
            # 改这两个值能让 BM25 那一路更/更不显眼，见 fusion.py 那段说明。
            weights={"vector": RRF_WEIGHT_VECTOR, "bm25": RRF_WEIGHT_BM25},
        )[:top_k]
    logger.info(
        "问题 %r 混合检索：向量 %d + BM25 %d → 融合后取 %d",
        question,
        len(vector_hits),
        len(keyword_hits),
        len(hits),
    )
    return hits


if __name__ == "__main__":
    from ecom.knowledge.settings import HYBRID_SEARCH as _HYBRID

    demo_store = get_vector_store()
    try:
        results = retrieve("退货政策是什么", demo_store)
        print(f"混合检索={'开' if _HYBRID else '关'}，检索到{len(results)}个相关chunk")
        for item in results:
            routes = []
            if item.get("vector_rank"):
                routes.append(f"向量#{item['vector_rank']}({item.get('vector_score'):.3f})")
            if item.get("bm25_rank"):
                routes.append(f"BM25#{item['bm25_rank']}({item.get('bm25_score'):.2f})")
            origin = " + ".join(routes) if routes else "—"
            print(f"  [{item['score']:.4f}] {item['source']}#{item['chunk_id']} [{origin}]")
            print(f"      {item['text'][:40]}...")
    finally:
        demo_store.close()
