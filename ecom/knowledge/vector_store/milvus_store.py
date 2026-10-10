"""
Milvus 向量存储模块
功能：把 chunk 和向量存进 Milvus，支持检索

和 ChromaStore 实现同一套契约（见 ecom/knowledge/vector_store/base.py）。
最关键的一条差异：Milvus 的 COSINE 返回的是**相似度**（越大越相关），
不像 Chroma 返回距离，所以这里**不能**写 `1 - distance`。

需要 Milvus 服务在跑：docker compose --profile milvus up -d
依赖是可选组，记得 uv sync --extra milvus
"""
from pymilvus import DataType, MilvusClient

from ecom.knowledge.settings import (
    COLLECTION_NAME,
    EMBED_DIM,
    MILVUS_TOKEN,
    MILVUS_URI,
)
from ecom.knowledge.utils.logger import setup_logger

logger = setup_logger(__name__)

METRIC_TYPE = "COSINE"
VECTOR_FIELD = "vector"

# query 的单次返回上限。Milvus 服务端有个 maxQueryResultWindow（默认 16384），
# 不显式给 limit 时超出的部分会被**静默丢掉**——对"还原整份语料"这种场景，
# 静默截断等于产出一份缺了尾巴的文档，比直接报错还危险。所以这里给足，
# 且真的够到上限时补一条 warning。
QUERY_LIMIT = 16384


def _quote(value: str) -> str:
    """
    把字符串安全地嵌进 Milvus 的 filter 表达式。

    Milvus 的表达式语法里字符串必须带双引号，且引号内的 " 和 \\ 要转义。
    这里**不使用** filter_params 模板化参数：有资料称 delete() 不支持它，
    来源冲突且无法实测，与其依赖不确定的行为，不如自己转义。

    换行/回车/制表符直接替换成空格而不做转义——表达式解析器对这些转义序列的
    支持不明确，而 source 来自文件名，Windows 下本来就不可能出现这些字符。
    """
    cleaned = (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", " ")
        .replace("\n", " ")
        .replace("\t", " ")
    )
    return f'"{cleaned}"'


class MilvusStore:
    def __init__(self, collection_name: str = COLLECTION_NAME):
        self.collection_name = collection_name

        # 本地 Standalone 无鉴权，token 为空时不要传空字符串
        kwargs = {"uri": MILVUS_URI}
        if MILVUS_TOKEN:
            kwargs["token"] = MILVUS_TOKEN

        try:
            self.client = MilvusClient(**kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"连不上 Milvus（{MILVUS_URI}）。\n"
                f"起服务：docker compose --profile milvus up -d"
                f"（本机已有 Milvus 就别起，把 MILVUS_URI 指过去即可）\n"
                f"注意容器里不能写 localhost——那是容器自己，"
                f"要用 http://host.docker.internal:19530\n"
                f"原始错误：{exc}"
            ) from exc

        if self.client.has_collection(collection_name):
            # 已存在就直接复用；但检索/删除都要求集合处于 loaded 状态
            self.client.load_collection(collection_name)
            logger.info("复用已有集合 %s（当前 %d 条）", collection_name, self.count())
        else:
            self._create_collection(collection_name)
            logger.info("已创建集合 %s（dim=%d, metric=%s）", collection_name, EMBED_DIM, METRIC_TYPE)

    def _create_collection(self, collection_name: str) -> None:
        """
        显式定义 schema，不用 create_collection 的简化路径。

        简化路径只会建 id + vector 两个字段，text/source 会被塞进动态字段（$meta），
        那样按 source 做标量过滤（幂等删除依赖它）就不可靠了。
        """
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=512)
        schema.add_field(field_name=VECTOR_FIELD, datatype=DataType.FLOAT_VECTOR, dim=EMBED_DIM)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64)

        # FLOAT_VECTOR 字段必须有索引才能检索；把 index_params 一起传给 create_collection，
        # 建完会自动 load，省掉一次手动 load_collection
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name=VECTOR_FIELD,
            index_type="AUTOINDEX",
            metric_type=METRIC_TYPE,
        )

        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )

    def add_chunks(self, chunks: list, embeddings: list) -> None:
        """
        把chunk和向量存进去
        chunks 形如 [{"text": ..., "source": ..., "chunk_id": ...}, ...]

        幂等：按 source 先删后加（和 ChromaStore 一致），
        重复入库不产生重复 chunk，文档变短时也不会残留旧 chunk。
        """
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(f"chunk 数({len(chunks)})与向量数({len(embeddings)})不一致")

        # 先把待写数据全部备好，**再**删除。原来这段在删除之后，
        # 意味着"构建 rows 时出错"的场景下旧数据已经被删掉了——
        # 明明是本可以在删除前就失败的活，却排在了删除后面。
        rows = [
            {
                # 主键沿用 Chroma 的复合写法，两个后端的唯一标识语义保持一致
                "id": f"{chunk['source']}#{chunk['chunk_id']}",
                VECTOR_FIELD: embedding,
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_id": int(chunk["chunk_id"]),
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]

        for source in {chunk["source"] for chunk in chunks}:
            self.client.delete(
                collection_name=self.collection_name,
                filter=f"source == {_quote(source)}",
            )

        self.client.insert(collection_name=self.collection_name, data=rows)

        # 不 flush 的话新数据还在 growing segment 里，get_collection_stats 的 row_count 会读到 0
        self.client.flush(self.collection_name)
        logger.info("写入 %d 个 chunk，集合当前共 %d 条", len(rows), self.count())

    def search(self, query_embedding: list, top_k: int = 5) -> list:
        """
        检索最相关的chunk
        返回：[{"text": ..., "source": ..., "chunk_id": ..., "score": ...}, ...]
        """
        total = self.count()
        if total == 0:
            logger.warning("向量库为空，请先运行 ingest.py 入库")
            return []

        result = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=min(top_k, total),
            output_fields=["text", "source", "chunk_id"],
            search_params={"metric_type": METRIC_TYPE},
        )

        hits = []
        for hit in result[0]:
            entity = hit.get("entity") or {}
            hits.append({
                "text": entity.get("text", ""),
                "source": entity.get("source", ""),
                "chunk_id": entity.get("chunk_id", -1),
                # COSINE 返回的已经是相似度（越大越相关），直接取用。
                # 这里是和 ChromaStore 唯一语义相反的地方，别再套 1 - x。
                "score": float(hit["distance"]),
            })
        return hits

    def all_chunks(self) -> list:
        """
        取出全部 chunk，供语料还原、孤儿数据检测用（见 ChromaStore.all_chunks）。

        filter="" 表示不过滤，和 count() 里的写法一致。必须显式给 limit：
        不给的话服务端按 maxQueryResultWindow 截断且不报错，会悄悄少掉一截。
        """
        rows = self.client.query(
            collection_name=self.collection_name,
            filter="",
            output_fields=["text", "source", "chunk_id"],
            limit=QUERY_LIMIT,
        )
        if not rows:
            return []

        chunks = [
            {
                "text": row.get("text", ""),
                "source": row.get("source", ""),
                "chunk_id": row.get("chunk_id", -1),
            }
            for row in rows
        ]
        if len(chunks) >= QUERY_LIMIT:
            logger.warning(
                "all_chunks 取到 %d 条，已达到单次 query 上限，结果可能不完整",
                len(chunks),
            )
        return chunks

    def count(self) -> int:
        """
        用 query(count(*)) 取真值，**不要**用 get_collection_stats 的 row_count。

        row_count 把逻辑删除的行也算在内，要等 Milvus 自己做 compaction 才减掉。
        实测：删掉 253 条又插回 253 条之后，query(count(*)) 报 253，
        row_count 却报 506。count() 要参与"重复入库数量不变"的幂等验证，
        报虚高的数字会把这个保证直接掩盖掉。
        """
        result = self.client.query(
            collection_name=self.collection_name,
            filter="",
            output_fields=["count(*)"],
        )
        if not result:
            return 0
        return int(result[0].get("count(*)", 0))

    def close(self) -> None:
        """释放 gRPC 连接"""
        self.client.close()


if __name__ == "__main__":
    store = MilvusStore()
    print(f"MilvusStore 已初始化，集合 {store.collection_name} 当前共 {store.count()} 条")
    store.close()
