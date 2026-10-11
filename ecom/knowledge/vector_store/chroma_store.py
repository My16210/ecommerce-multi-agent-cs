"""
Chroma向量存储模块
功能：把chunk和向量存进Chroma，支持检索

实现 ecom/knowledge/vector_store/base.py 里的 VectorStore 契约。
score 的语义在这里处理：Chroma 的 cosine 返回的是**距离**（越小越相关），
统一转成相似度（越大越相关）再返回给调用方，转换只在这一处发生。
"""
from pathlib import Path

import chromadb

from ecom.knowledge.settings import COLLECTION_NAME, VECTOR_DB_PATH
from ecom.knowledge.utils.logger import setup_logger

logger = setup_logger(__name__)


class ChromaStore:
    def __init__(self, collection_name: str = COLLECTION_NAME):
        Path(VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list, embeddings: list) -> None:
        """
        把chunk和向量存进去
        chunks 形如 [{"text": ..., "source": ..., "chunk_id": ...}, ...]

        幂等：同一个 source 先删后加，重复入库不会产生重复 chunk，
        文档变短时也不会残留旧 chunk。
        """
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(f"chunk 数({len(chunks)})与向量数({len(embeddings)})不一致")

        # 先把要写的数据全部备好，再动删除。"准备期"出错时还没碰过库里的旧数据。
        # 删除和写入之间仍有窗口（Chroma 不提供跨 delete+add 的事务），
        # 但窗口越短越好——这段是本可以在删除前就失败的活。
        ids = [f"{chunk['source']}#{chunk['chunk_id']}" for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [
            {"source": chunk["source"], "chunk_id": chunk["chunk_id"]}
            for chunk in chunks
        ]

        for source in {chunk["source"] for chunk in chunks}:
            self.collection.delete(where={"source": {"$eq": source}})

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info("写入 %d 个 chunk，集合当前共 %d 条", len(chunks), self.collection.count())

    def delete_source(self, source: str) -> None:
        """
        删掉某个 source 的全部 chunk（用途见契约：清理已从盘上删掉的文档）。

        删一个不存在的 source 不会报错——Chroma 的 delete 本来就是空操作。
        """
        self.collection.delete(where={"source": {"$eq": source}})

    def search(self, query_embedding: list, top_k: int = 5) -> list:
        total = self.collection.count()
        if total == 0:
            logger.warning("向量库为空，请先运行 ingest.py 入库")
            return []

        # n_results 超过集合大小时 chromadb 会直接抛异常，必须先夹住
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, total),
            include=["documents", "metadatas", "distances"],
        )

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            hits.append({
                "text": text,
                "source": metadata.get("source", ""),
                "chunk_id": metadata.get("chunk_id", -1),
                "score": 1 - distance,  # cosine 距离 → 相似度
            })
        return hits

    def all_chunks(self) -> list:
        """
        取出全部 chunk。不带 embeddings/distances——还原语料和孤儿检测都只用正文和出处。

        Chroma 的 get() 不传 limit 就是全量，没有 Milvus 那种服务端上限。
        """
        result = self.collection.get(include=["documents", "metadatas"])

        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        return [
            {
                "text": text,
                "source": (metadata or {}).get("source", ""),
                "chunk_id": (metadata or {}).get("chunk_id", -1),
            }
            for text, metadata in zip(documents, metadatas)
        ]

    def count(self) -> int:
        return self.collection.count()

    def close(self) -> None:
        """Chroma 是嵌入式客户端，关闭它释放本地文件句柄"""
        self.client.close()


if __name__ == "__main__":
    store = ChromaStore()
    print(f"ChromaStore已初始化，当前共 {store.count()} 条")
