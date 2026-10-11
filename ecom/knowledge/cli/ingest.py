"""
离线入库脚本
功能：把 data/knowledge/documents 下的文档切分、向量化、存进向量库

只在文档变更时需要跑。重复执行不会产生重复 chunk（按文件先删后加），
文档被**删掉**时也会一并清理（库里有、盘上没有的孤儿，见下面 prune 那段）。

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


def find_documents(documents_dir: Path) -> list[Path]:
    """语料目录下所有可入库的文件（含子目录）"""
    return sorted(
        path
        for path in documents_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def duplicate_basenames(files: list[Path]) -> dict[str, list[Path]]:
    """找出**文件名相同**的文件，返回 {文件名: [路径, ...]}。

    chunk 的 source 取的是 `os.path.basename(file_path)`（见 splitter.py），
    所以两个不同子目录下的同名文件会**共用同一个 source**。而 add_chunks 是
    "按 source 先删后加"——后入库的那份会把先入库的整份删掉。

    症状很难查：不报错、不警告，只是库里的条数比盘上少了一整份文档，
    检索永远召回不到先入库的那份，"语料明明在啊"。

    语料以前是平铺一层，撞名不可能发生；加了子目录之后（电商客服示例/、
    法规原文/）它就从"不可能"变成"迟早会"。所以入库前先拦下来。
    """
    by_name: dict[str, set[Path]] = {}
    for path in files:
        by_name.setdefault(path.name, set()).add(path)
    # 用 set 而不是 list：同一个路径被列了两次不是撞名（那是调用方给的重复项），
    # 只有**两个不同路径**共用一个文件名才是。
    return {name: sorted(paths) for name, paths in by_name.items() if len(paths) > 1}


def prune_orphans(store, files: list[Path]) -> int:
    """清掉"库里有、盘上没有"的 chunk 来源，返回清掉的 chunk 数。

    add_chunks 的幂等只在文件还在盘上时成立：它按 source 先删后加，前提是
    这次调用提到了这个 source。文件被删掉之后就再也没人提到它了——那些 chunk
    会一直留在库里，而且检索照样召回得到。于是"换了语料、评测数字却对不上"
    就成了一个查不出原因的谜。
    """
    on_disk = {path.name for path in files}
    existing = store.all_chunks()
    orphans = sorted({chunk["source"] for chunk in existing} - on_disk)
    if not orphans:
        return 0

    orphan_set = set(orphans)
    removed = sum(1 for chunk in existing if chunk["source"] in orphan_set)
    for source in orphans:
        store.delete_source(source)

    print("-" * 50)
    print(f"清理孤儿：{len(orphans)} 份文档已不在盘上，共移除 {removed} 个 chunk")
    for source in orphans:
        print(f"  - {source}")
    return removed


def main():
    documents_dir = Path(DOCUMENTS_PATH)
    files = find_documents(documents_dir)

    if not files:
        print(f"没有在 {documents_dir} 找到可入库的文档")
        return

    clashes = duplicate_basenames(files)
    if clashes:
        # 明知会互相覆盖还继续入库，就是把一份语料悄悄弄丢。宁可什么都不做。
        print("发现同名文件，入库会互相覆盖，已中止：")
        for name, paths in sorted(clashes.items()):
            print(f"  ✗ {name}")
            for path in paths:
                print(f"      {path}")
        print("chunk 的 source 只取文件名，改掉重名的那些再跑。")
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

        # 顺序很重要：先入库、再清理。反过来的话，某个文件这轮入库失败被跳过，
        # 它的旧 chunk 已被当成"盘上还有"而留下——那不是孤儿，是对的。
        # 而"盘上没有"的文件无论如何都该清掉，和本轮成不成功无关。
        prune_orphans(store, files)

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
