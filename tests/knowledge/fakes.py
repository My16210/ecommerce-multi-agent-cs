"""
测试用的假客户端。

放这里而不是 conftest.py，是因为 conftest 里的东西默认不能被 import。

为什么不留方便、直接开真的库：
- **Chroma**：一建 PersistentClient 就会拉起 posthog telemetry，测试里不该有网络行为；
  而且会往盘上写真实数据，测试之间互相污染。
- **Milvus**：需要在跑服务，而且 pymilvus 是可选的 `--extra milvus`，
  不加 extra 的环境里 `import pymilvus` 直接炸。塞个假模块进 sys.modules
  就能让契约测试在任何环境下都跑得起来。

假客户端只实现被测代码真正调用的那几个方法——这不是"完整模拟"，
是"够用的替身"。多写的方法反而会掩盖被测代码的真实调用面。
"""
import importlib
import sys
import types

# --------------------------------------------------------------------------
# Chroma
# --------------------------------------------------------------------------


class FakeChromaCollection:
    """内存版 collection，只实现 ChromaStore 用到的四个方法

    next_distances 是留给测试注入的：ChromaStore 会把 query 返回的距离
    做 `1 - distance`，只有注入确定值，才能钉死"翻转"这个动作确实发生了。
    """

    def __init__(self):
        self.rows = {}              # id -> {"document", "metadata", "embedding"}
        self.next_distances = None

    def count(self):
        return len(self.rows)

    def delete(self, where):
        # ChromaStore 只用 {"source": {"$eq": ...}} 这一种写法
        source = where["source"]["$eq"]
        self.rows = {
            key: row for key, row in self.rows.items()
            if row["metadata"]["source"] != source
        }

    def add(self, ids, documents, metadatas, embeddings):
        for id_, document, metadata, embedding in zip(ids, documents, metadatas, embeddings):
            self.rows[id_] = {
                "document": document,
                "metadata": metadata,
                "embedding": embedding,
            }

    def get(self, include):
        """真 Chroma 的 get() 不传 limit 就是全量

        只回 include 里点名的字段，这样 all_chunks 若顺手要了 embeddings
        会在这里直接暴露出来（还原语料根本不需要向量，要了就是白拉一趟）。
        """
        rows = list(self.rows.values())
        result = {}
        if "documents" in include:
            result["documents"] = [row["document"] for row in rows]
        if "metadatas" in include:
            result["metadatas"] = [row["metadata"] for row in rows]
        return result

    def query(self, query_embeddings, n_results, include):
        # 真 Chroma 在 n_results 超过集合大小时**直接抛异常**，不是少返回几条。
        # 这里必须照样抛：Python 切片越界不会报错，若假客户端"宽容"地切片，
        # ChromaStore 里那句 min(top_k, total) 的夹取就永远测不出来，
        # 那条测试会退化成永远通过的空测试。
        if n_results > len(self.rows):
            raise ValueError(
                f"Number of requested results {n_results} is greater than "
                f"number of elements in index {len(self.rows)}"
            )

        rows = list(self.rows.values())[:n_results]
        if self.next_distances is None:
            distances = [0.1 * (index + 1) for index in range(len(rows))]
        else:
            distances = list(self.next_distances)[:len(rows)]
        return {
            "documents": [[row["document"] for row in rows]],
            "metadatas": [[row["metadata"] for row in rows]],
            "distances": [distances],
        }


class FakeChromaClient:
    def __init__(self, collection=None):
        self.collection = collection or FakeChromaCollection()
        self.closed = False

    def get_or_create_collection(self, name, metadata=None):
        return self.collection

    def close(self):
        self.closed = True


def install_fake_chroma(monkeypatch, tmp_path):
    """让 ChromaStore() 落在内存假 collection 上，返回 chroma_store 模块

    拿到的 store 的 `.collection` 就是这个假 collection，测试可以直接
    往上面注入 next_distances，或读 rows 做断言。
    """
    import ecom.knowledge.vector_store.chroma_store as module

    client = FakeChromaClient()
    monkeypatch.setattr(module.chromadb, "PersistentClient", lambda **kwargs: client)
    # 否则 ChromaStore.__init__ 会在真实的 data/vector_store 下建目录
    monkeypatch.setattr(module, "VECTOR_DB_PATH", str(tmp_path))
    return module


# --------------------------------------------------------------------------
# Milvus
# --------------------------------------------------------------------------


class _FakeDataType:
    """只列 MilvusStore._create_collection 用到的三个枚举值"""

    VARCHAR = "VARCHAR"
    FLOAT_VECTOR = "FLOAT_VECTOR"
    INT64 = "INT64"


def _unquote(literal):
    """milvus_store._quote 的逆运算，让假客户端能读懂 filter 表达式"""
    text = literal.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    return text.replace('\\"', '"').replace("\\\\", "\\")


class FakeMilvusClient:
    """内存版 MilvusClient，只实现 MilvusStore 用到的那几个方法"""

    def __init__(self, **kwargs):
        self.rows = {}                    # id -> row dict
        self.deleted_sources = []         # 记一下删过谁，便于断言
        self.next_search_distances = None
        self.closed = False

    def has_collection(self, name):
        # 恒为 True，让 MilvusStore 走"复用已有集合"分支，绕开 _create_collection
        return True

    def load_collection(self, name):
        pass

    def flush(self, name):
        pass

    def close(self):
        self.closed = True

    def insert(self, collection_name, data):
        for row in data:
            self.rows[row["id"]] = row

    def delete(self, collection_name, filter):
        # 表达式形如：source == "某文件名.md"
        source = _unquote(filter.split("==", 1)[1])
        self.deleted_sources.append(source)
        self.rows = {
            key: row for key, row in self.rows.items() if row["source"] != source
        }

    def query(self, collection_name, filter, output_fields, limit=None):
        if output_fields == ["count(*)"]:
            return [{"count(*)": len(self.rows)}]
        if "text" in output_fields:
            rows = list(self.rows.values())
            # 真 Milvus 超出 limit 是**静默截断**，不报错。假客户端照做，
            # 否则"够到上限时补一条 warning"那段逻辑永远测不出来。
            if limit is not None:
                rows = rows[:limit]
            return [
                {"text": row["text"], "source": row["source"], "chunk_id": row["chunk_id"]}
                for row in rows
            ]
        raise AssertionError(f"假客户端没预料到的 output_fields：{output_fields}")

    def search(self, collection_name, data, limit, output_fields, search_params):
        rows = list(self.rows.values())[:limit]
        if self.next_search_distances is None:
            distances = [0.5] * len(rows)
        else:
            distances = list(self.next_search_distances)[:len(rows)]
        return [[
            {
                "entity": {
                    "text": row["text"],
                    "source": row["source"],
                    "chunk_id": row["chunk_id"],
                },
                "distance": distance,
            }
            for row, distance in zip(rows, distances)
        ]]


def install_fake_pymilvus(monkeypatch):
    """把假 pymilvus 塞进 sys.modules，再导入 milvus_store，返回该模块

    必须**先**清掉 sys.modules 里缓存的 milvus_store：否则 importlib
    会直接返回上一次那个绑定了旧假模块的版本，测试之间会串。
    """
    fake = types.ModuleType("pymilvus")
    fake.DataType = _FakeDataType
    fake.MilvusClient = FakeMilvusClient

    monkeypatch.setitem(sys.modules, "pymilvus", fake)
    monkeypatch.delitem(sys.modules, "ecom.knowledge.vector_store.milvus_store", raising=False)
    return importlib.import_module("ecom.knowledge.vector_store.milvus_store")
