"""
向量库工厂
功能：按配置返回对应的后端实现

调用方一律用 get_vector_store()，不直接 new 具体后端，
这样"用哪个库"就只是一个配置项，检索/rerank/测评的代码都不用动。

MilvusStore 的 import 放在函数内部（延迟导入）：默认的 Chroma 环境没装 pymilvus，
如果在模块顶层 import，连 `import ecom.knowledge.vector_store` 都会失败。
"""
from typing import TYPE_CHECKING

from ecom.knowledge.settings import VECTOR_BACKEND
from ecom.knowledge.vector_store.chroma_store import ChromaStore

# base 只在类型标注里用，放 TYPE_CHECKING 下可以避免 `python -m ecom.knowledge.vector_store.base`
# 时因为包提前导入它而发出的 RuntimeWarning
if TYPE_CHECKING:
    from ecom.knowledge.vector_store.base import VectorStore


def get_vector_store(name: str = None) -> "VectorStore":
    """name 为空时用配置里的 VECTOR_BACKEND（默认 chroma）"""
    backend = (name or VECTOR_BACKEND).strip().lower()

    if backend == "chroma":
        return ChromaStore()

    if backend == "milvus":
        try:
            from ecom.knowledge.vector_store.milvus_store import MilvusStore
        except ImportError as exc:
            raise RuntimeError(
                "要使用 Milvus 后端请先装可选依赖：uv sync --extra milvus"
            ) from exc
        return MilvusStore()

    raise ValueError(f"未知的向量库后端 {backend!r}，可选：chroma / milvus")


if __name__ == "__main__":
    store = get_vector_store()
    print(f"当前后端：{type(store).__name__}，共 {store.count()} 条")
