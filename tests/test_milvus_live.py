"""连**真的 Milvus** 走一遍 MilvusStore（默认不跑，要显式打开）。

    ECOM_LIVE_MILVUS=1 MILVUS_URI=http://localhost:19530 \
        uv run pytest tests/test_milvus_live.py

【为什么它默认不跑】
它连的是**外部服务**，而不是仓库里的一份派生数据。默认跑的话，`pytest` 就会
变成"必须先起一个 Milvus"，和全仓库"clone 下来就能跑测试"那条前提直接冲突。
所以它是个**开关式的验证**，不是常规回归测试——常规那套（Chroma 那条路）
在 `tests/knowledge/` 里，零外部依赖。

【它证明了什么】
`MilvusStore` 的**代码路径真的通**：建集合（显式 schema + AUTOINDEX）、
insert + flush、`get_collection_stats` 数数、COSINE 检索、按 source 删除。
`test_the_chroma_and_milvus_backends_are_wired_the_same_way` 那类测试只能证明
**接口对得上**，证明不了"对面那个服务真的收下了"。

【为什么不给宿主那个 Milvus 留垃圾】
每次用 `ecom_verify_<8位随机>` 这样一个**唯一**的集合名，跑完在 finally 里
drop 掉。固定名字的话，两次并发跑会互相删数据；不 drop 的话，这个验证跑几遍
就在用户的 Milvus 里留几坨。

【已知的一处版本错位，如实记在这里】
本机常驻的是 **Milvus v2.5.5**，而 `pyproject.toml` 把 pymilvus pin 在
`>=2.6,<3`、`docker-compose.yml` 里那个服务是 `milvusdb/milvus:v2.6.24`。
实测客户端 2.6 连 2.5.5 服务端这几个接口都能用（下面就是跑通的证据），
但**这条对应关系没有保证**：真要长期用，服务端该换到 2.6.x 对齐 pin。
"""
import os
import uuid

import pytest

pytest.importorskip("pymilvus")

# 默认跳过。理由见文件头——它要连外部服务，不该成为 `pytest` 的隐含前提。
pytestmark = pytest.mark.skipif(
    os.getenv("ECOM_LIVE_MILVUS") != "1",
    reason="要连真 Milvus 的验证，默认不跑。打开：ECOM_LIVE_MILVUS=1（见文件头）",
)


def test_the_milvus_backend_round_trips():
    from ecom.knowledge.settings import EMBED_DIM, MILVUS_URI
    from ecom.knowledge.vector_store.milvus_store import MilvusStore

    collection = "ecom_verify_" + uuid.uuid4().hex[:8]
    store = MilvusStore(collection_name=collection)

    base = [0.0] * EMBED_DIM
    first = base.copy()
    first[0] = 1.0
    second = base.copy()
    second[1] = 1.0

    try:
        store.add_chunks(
            [
                {"text": "退货政策：七天无理由退货", "source": "verify.md", "chunk_id": 0},
                {"text": "物流时效：下单后三到五天送达", "source": "verify.md", "chunk_id": 1},
            ],
            [first, second],
        )

        # flush 过再数：不 flush 的话新数据还在 growing segment 里，
        # row_count 会读到 0——这个坑 milvus_store.py 里那段注释专门写了。
        assert store.count() == 2

        hits = store.search(first, top_k=2)
        assert len(hits) == 2
        assert hits[0]["text"] == "退货政策：七天无理由退货"
        # COSINE + 完全相同的向量 = 1.0。这条同时钉住了"metric 没有被人
        # 改成 L2 或者反过来套一次 1 - x"——那是 ChromaStore 和 MilvusStore
        # 唯一语义相反的地方。
        assert hits[0]["score"] == pytest.approx(1.0, abs=1e-4)

        # 幂等入库的前提：按 source 能删干净
        store.delete_source("verify.md")
        assert store.count() == 0
        assert store.all_chunks() == []
    finally:
        # 不给别人的 Milvus 留垃圾。**finally**：断言失败时也要清掉，
        # 否则一次红了的验证会在用户的实例里留下一个没人知道的集合。
        store.client.drop_collection(collection)
        store.close()

    assert MILVUS_URI, "MILVUS_URI 没配置——上面全都连上了说明这条不该成立"


def test_the_hosted_instance_has_no_leftovers_from_this_file():
    """上一次跑没跑干净，这里能看出来。

    单独一条而不是塞进上面：上面那条**失败时**根本走不到这个断言，
    而"失败留下的垃圾"恰恰是最需要被发现的情形。
    """
    import pymilvus

    from ecom.knowledge.settings import MILVUS_URI

    client = pymilvus.MilvusClient(uri=MILVUS_URI)
    leftovers = [n for n in client.list_collections() if n.startswith("ecom_verify_")]

    assert leftovers == [], f"上个用例没清理干净：{leftovers}"
