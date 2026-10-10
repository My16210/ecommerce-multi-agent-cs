"""
向量库抽象层
功能：定义统一的存储/检索契约，Chroma 和 Milvus 两个后端都实现它

为什么需要这一层
----------------
两个后端的原始返回语义是**相反**的：

    Chroma  cosine 返回「距离」   —— 越小越相关
    Milvus  COSINE 返回「相似度」 —— 越大越相关，范围 [-1, 1]

如果让调用方直接面对这个差异，检索、rerank、测评三处都得知道"当前用的是哪个库"，
换个后端就要改一遍排序逻辑。所以契约在这里把 score 钉死成
**相似度，越大越相关**，转换放在各后端内部做：

    ChromaStore.search()  ->  score = 1 - distance   （翻转）
    MilvusStore.search()  ->  score = distance       （直接用，不要再取 1-x）

调用方（retriever / reranker / eval）只认这一个语义。
"""
from typing import Protocol, TypedDict, runtime_checkable


class Chunk(TypedDict):
    """入库单元。source 和 chunk_id 是引用追溯的源头，不能丢"""

    text: str
    source: str
    chunk_id: int


class Hit(TypedDict):
    """检索结果。score 一律是相似度，越大越相关"""

    text: str
    source: str
    chunk_id: int
    score: float


@runtime_checkable
class VectorStore(Protocol):
    """
    向量库契约。两个实现都必须满足下面四条：

    1. `add_chunks` 要求 `len(chunks) == len(embeddings)`，否则抛 ValueError
    2. `add_chunks` **按 source 幂等**：先删掉该 source 的全部旧 chunk 再插入。
       这样重复入库不产生重复 chunk，文档变短时也不会残留旧 chunk
    3. 空库、或 `top_k` 超过库里总条数，`search` 都**不得抛异常**，返回 `[]`
    4. `count()` 允许是近似值（Milvus 的 row_count 有这个性质），只用于判空和夹 top_k
    5. `all_chunks()` 返回全库 chunk，但**不保证顺序**，调用方按 (source, chunk_id) 自己排
    """

    def add_chunks(self, chunks: list, embeddings: list) -> None:
        """把 chunk 和对应向量写入。chunks 形如 [{"text","source","chunk_id"}, ...]"""
        ...

    def search(self, query_embedding: list, top_k: int = 5) -> list:
        """返回 [{"text","source","chunk_id","score"}, ...]，按 score 降序"""
        ...

    def count(self) -> int:
        """当前库里的条数"""
        ...

    def all_chunks(self) -> list:
        """
        取出库里全部 chunk，供语料还原、以及"库里有盘上没有"的孤儿数据检测用。

        返回 [{"text","source","chunk_id"}, ...]，**不含向量也不含 score**——
        这两个用途都只看正文和出处。

        顺序不保证：Chroma 按内部 id 序、Milvus 按 segment 存储序，
        两边都不等于 chunk_id。调用方自己按 (source, chunk_id) 排。

        没有分页，chunk 上万时注意内存。
        """
        ...

    def close(self) -> None:
        """释放连接。Chroma 是空实现，Milvus 要关掉 gRPC channel"""
        ...


if __name__ == "__main__":
    from ecom.knowledge.vector_store.chroma_store import ChromaStore

    # runtime_checkable 只能检查方法是否存在，不做签名校验
    print("ChromaStore 满足 VectorStore 契约:", isinstance(ChromaStore(), VectorStore))
