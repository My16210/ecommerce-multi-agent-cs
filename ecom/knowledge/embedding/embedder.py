"""
Embedding模块
功能：用BGE-M3把文本转成向量（硅基流动API）
"""
from functools import lru_cache

from openai import OpenAI

from ecom.knowledge.settings import (
    BGE_M3_MODEL,
    EMBED_BATCH_SIZE,
    QUERY_EMBED_CACHE_SIZE,
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
)
from ecom.knowledge.utils.logger import setup_logger
from ecom.knowledge.utils.timing import metric

logger = setup_logger(__name__)

_client = None


def _get_client() -> OpenAI:
    """模块级单例，避免每次调用都重建 client"""
    global _client
    if _client is None:
        if not SILICONFLOW_API_KEY:
            raise RuntimeError("缺少 SILICONFLOW_API_KEY，请在项目根目录的 .env 中配置")
        _client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)
    return _client


def get_embedding(texts: list) -> list:
    """
    把文本列表转成向量，返回向量列表（与输入顺序一一对应）

    注意：BGE-M3 输出固定 1024 维，不要传 dimensions 参数，
    硅基流动对 bge 系列传了会直接返回 400。
    """
    if not texts:
        return []

    client = _get_client()
    vectors = []

    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start:start + EMBED_BATCH_SIZE]
        response = client.embeddings.create(model=BGE_M3_MODEL, input=batch)
        # 用量能拿到就记进计时器（兼容接口带 usage），拿不到就算了——成本统计
        # 缺一项不影响主流程，但**不能**因此假设它是 0，画像里会标出来。
        usage = getattr(response, "usage", None)
        metric("embed_tokens", getattr(usage, "total_tokens", 0) if usage else 0)
        # 按 index 还原顺序，不依赖接口返回的顺序
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors.extend(item.embedding for item in ordered)
        logger.info("向量化进度 %d/%d", min(start + EMBED_BATCH_SIZE, len(texts)), len(texts))

    return vectors


# ---------------------------------------------------------------------------
# 问题级向量缓存（只有精确重复才算命中）
# ---------------------------------------------------------------------------
#
# 为什么只做"精确匹配"这一档，不做语义缓存
# ----------------------------------------
# 语义缓存（相似度超过阈值就复用旧问题的向量/答案）看着很美，代价却是**静默**
# 的：它用一个近似问题的结果回答另一个问题，而且错了没有任何迹象——问题问的
# 是"怎么退货"，命中的可能是"怎么换货"。这与本项目一路守着的"宁可暴露也不
# 掩盖"（retrieve() 不兜底、judge 用 ANSWER_FALLBACK 判空）恰好相反。
# 所以这里只缓存**逐字相同**的问题：命中意味着答案必然一致，没有近似。
#
# 为什么放在 embedder 这一层
# -------------------------
# 向量化是整条链路里最贵、也最可能被重复触发的 I/O（画像里占检索段的 96.7%）。
# 缓存键就是文本本身，所以入库那一批文本若完全重复也一并受益，且不会串味——
# 拿到的向量与现算的逐位相同。


@lru_cache(maxsize=QUERY_EMBED_CACHE_SIZE)
def _query_embedding_cached(text: str) -> tuple:
    """
    缓存的底层实现：返回 tuple（不可变，避免调用方改到缓存里的那份）。

    用 functools.lru_cache 而不是自建 dict，因为它**不缓存异常**——
    向量化失败时照常往上抛，检索该暴露的故障不会被缓存掩盖（与 retrieve()
    不兜底同一条理由）。lru_cache 内部有锁，Streamlit 多线程下也安全。
    """
    return tuple(get_embedding([text])[0])


def get_query_embedding(text: str) -> list:
    """问题级向量化：同一问题重复提问不再走一次 API 往返。"""
    return list(_query_embedding_cached(text))


if __name__ == "__main__":
    # 测试
    vectors = get_embedding(["退货政策", "安装指南"])
    print(f"向量维度: {len(vectors[0])}")
