"""
离线入库脚本
功能：把 data/documents 下的文档切分、向量化、存进向量库

只在文档变更时需要跑。重复执行不会产生重复 chunk（按文件先删后加）。

这是**向量库的唯一写入口**——问答链路绝不直接写库，
否则模型生成的答案会回流成"权威资料"，造成知识库自污染。

用法：uv run python ingest.py
后端由 settings.VECTOR_BACKEND 决定（默认 chroma），也可用环境变量覆盖：
      VECTOR_BACKEND=milvus uv run python ingest.py
"""
from pathlib import Path

from ecom.knowledge.document_loader.splitter import load_and_split
from ecom.knowledge.embedding.embedder import get_embedding
from ecom.knowledge.settings import DOCUMENTS_PATH
from ecom.knowledge.utils.logger import setup_logger
from ecom.knowledge.vector_store import get_vector_store

logger = setup_logger(__name__)

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}


def main():
    documents_dir = Path(DOCUMENTS_PATH)
    files = sorted(
        path
        for path in documents_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )

    if not files:
        print(f"没有在 {documents_dir} 找到可入库的文档")
        return

    print("=" * 50)
    print(f"开始入库，共 {len(files)} 个文件")
    print("=" * 50)

    store = get_vector_store()
    print(f"向量库后端：{type(store).__name__}")
    total_chunks = 0
    done_files = 0

    try:
        for path in files:
            try:
                chunks = load_and_split(str(path))
                if not chunks:
                    print(f"  {path.name}: 切不出内容，跳过")
                    continue

                # 先把整个文件的向量一次算完，再交给 add_chunks。
                # add_chunks 内部是「按 source 先删后加」，所以向量化失败时
                # 根本还没走到删除那一步——库里那份旧数据不会被动。
                embeddings = get_embedding([chunk["text"] for chunk in chunks])
                store.add_chunks(chunks, embeddings)
            except Exception as exc:
                # 单个文件失败只跳过它，不中断整轮入库。
                # 否则前面的文件已入库、后面的一个没处理，
                # 表面上只是"少了几条"，很难看出是半途而废。
                print(f"  {path.name}: 入库失败，跳过（{exc}）")
                logger.warning("入库 %s 失败：%s", path, exc)
                continue

            total_chunks += len(chunks)
            done_files += 1
            print(f"  {path.name}: {len(chunks)} 个 chunk")

        print("-" * 50)
        # 报成功数而不是 len(files)：上面有几种 continue，报总数会虚高
        print(f"入库完成：{done_files}/{len(files)} 个文件，{total_chunks} 个 chunk")
        print(f"向量库当前共 {store.count()} 条（重复运行此数字应保持不变）")
    finally:
        # 契约里声明了 close()，入口就该履行。Milvus 要关 gRPC channel，
        # Chroma 是空实现，多调一次无害。
        store.close()


if __name__ == "__main__":
    main()
