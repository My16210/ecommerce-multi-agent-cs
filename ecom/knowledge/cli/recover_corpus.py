"""
从向量库还原原始语料
功能：把库里已经切碎的 chunk 拼回整篇文档，写回 data/documents/

为什么需要这个脚本
------------------
本项目出过一次事故：`data/documents/` 下的三份教程语料在磁盘上丢了，只剩
切好的 253 个 chunk 还留在向量库里。检索还能用，但**只要谁删掉
`data/vector_store/`，语料就永久没了**——项目再也重建不起来。

这是"向量库被当成了数据源"的典型后果。向量库应该是**派生物**：所有内容都由
`data/documents/` 经 ingest.py 单向生成，删掉随时能重建。这个脚本就是把已经
倒过来的依赖关系扳回去，让磁盘重新成为唯一真源。

还原的难点：丢失的分隔符
------------------------
相邻 chunk 之间有重叠（chunk_overlap=50），按"最长后缀 = 最长前缀"就能把
重叠部分去掉接上。**但重叠恰好为 0 的接缝是麻烦所在**：切分器的
`strip_whitespace`（LangChain 默认开启）把每个 chunk 两端的空白删掉了，
接缝处原本的段落分隔符就跟着没了。实测三份文档共 148 处这样的接缝。

这不是小瑕疵：丢掉的若是代码块里的换行，`x = 1` 和 `y = 2` 会粘成一行——
对一份编程教程来说就是实打实的损坏。

怎么把关掉的分隔符找回来
------------------------
**不猜规则，用判据。** 切分器是确定性的：只有把原文拼对了，重切才能切出
一模一样的块。所以对每处零重叠接缝，把候选分隔符代进去，重切一遍，数有多少
块和库里的原始块逐字相同——留下得分最高的那个（见 choose_separators）。

这个判据不需要知道原文，却是可以验证的，而且顺带产出一个诚实的保真度指标，
直接打在报告里，不用读者自己判断"还原得准不准"。

用法
----
    uv run python recover_corpus.py              # 已存在的文件不覆盖
    uv run python recover_corpus.py --force      # 覆盖
    uv run python recover_corpus.py --dry-run    # 只看报告，不写盘
"""
import argparse
import difflib
from collections import defaultdict
from pathlib import Path

from ecom.knowledge.document_loader.splitter import split_text
from ecom.knowledge.settings import DOCUMENTS_PATH
from ecom.knowledge.utils.logger import setup_logger
from ecom.knowledge.vector_store import get_vector_store

logger = setup_logger(__name__)

# 丢失的分隔符只可能是这几种：切分器只在 "\n\n" / "\n" / " " / "" 处切开，
# 而 strip_whitespace 只会删空白，所以 "。" "，" 这些非空白分隔符不会丢。
# 顺序有讲究：并列时优先取靠前的，段落分隔比硬拼更可能符合原意。
SEPARATOR_CANDIDATES = ("\n\n", "\n", "", " ")

# 保真度低于这个比例就提示：很可能是切分参数在入库后被改过，
# 导致"重切能否复现"这个判据本身失效了。
FIDELITY_WARN = 0.8

# 坐标下降的轮数上限。实测两三论就收敛，留点余量。
MAX_ROUNDS = 6


def group_by_source(chunks: list) -> dict:
    """按 source 归拢。排序交给 merge_chunks 做，免得两处各排一次"""
    grouped = defaultdict(list)
    for chunk in chunks:
        grouped[chunk["source"]].append(chunk)
    return grouped


def overlap_size(left: str, right: str) -> int:
    """
    left 的后缀恰好等于 right 的前缀，返回最长的那个长度；没有重叠返回 0。

    取最长是安全的：接缝前后的文本必须真的一模一样才算数，自然语言里不可能
    出现长得离谱的巧合。反过来取最短就危险了——真实的 50 字重叠会被漏掉，
    结果是把同一段内容复制两遍。

    这也解释了为什么不必担心"过删"：命中长度 L 时，left[-L:] 和 right[:L]
    逐字相同，删掉 right 那一份等于只保留一次，没有丢任何信息。
    """
    upper = min(len(left), len(right))
    for size in range(upper, 0, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def merge_chunks(chunks: list, separators: dict) -> tuple:
    """
    把同一个 source 的 chunk 拼成整篇文档。

    separators 只对**零重叠**的接缝生效（有重叠的接缝不需要填任何东西，
    重叠部分本身就是衔接）。

    返回 (正文, 接缝重叠长度列表, 零重叠接缝的下标列表)。
    """
    ordered = sorted(chunks, key=lambda chunk: chunk["chunk_id"])
    text = ordered[0]["text"]
    overlaps = []
    zeros = []

    for index, chunk in enumerate(ordered[1:], start=1):
        size = overlap_size(text, chunk["text"])
        overlaps.append(size)
        if size:
            text += chunk["text"][size:]
        else:
            zeros.append(index)
            text += separators.get(index, "") + chunk["text"]

    return text, overlaps, zeros


def chunk_match_score(text: str, ordered: list) -> tuple:
    """
    把 text 用项目自己的切分器重切一遍，看能不能还原出库里那些原始块。

    返回 (逐字相同的块数, 重切出的块数)。

    用 difflib 对齐而不是按下标硬比：重切出来的块数可能和原始不同，
    一旦多一块少一块，后面所有下标都会错位，硬比会把正确的部分也判成不匹配。
    """
    again = split_text(text)
    matcher = difflib.SequenceMatcher(
        None, [chunk["text"] for chunk in ordered], again, autojunk=False
    )
    matched = sum(
        i2 - i1 for tag, i1, i2, _j1, _j2 in matcher.get_opcodes() if tag == "equal"
    )
    return matched, len(again)


def choose_separators(ordered: list, zeros: list) -> tuple:
    """
    为每处零重叠接缝挑一个分隔符。

    判据是"能不能把原文重切回来"：候选分隔符代进去重切一遍，数有多少块跟库里
    的原始块逐字相同。这个判据不需要知道原文——正因为切分器是确定性的，只有
    拼对了原文才可能切出一模一样的结果。

    做法是坐标下降：从全填段落分隔符起步，逐个接缝试遍候选、留下能让命中变多的
    那个，反复几轮直到不再变好。接缝一百多处、每轮要重切几十次，实测总共十几秒，
    对一次性脚本完全够用。

    返回 (最好的分隔符 dict, 命中块数, 重切块数)。

    注意这里优化的是"重切复现率"这个代理指标。它不完美，但方向是对的：分隔符
    填错会让累积文本的长度漂移，512 的切分边界跟着漂，重切自然对不上。
    实测能拿回九成以上的块。
    """
    separators = {index: SEPARATOR_CANDIDATES[0] for index in zeros}

    def score(candidate: dict) -> tuple:
        text, _overlaps, _zeros = merge_chunks(ordered, candidate)
        return chunk_match_score(text, ordered)

    best_matched, best_chunks = score(separators)

    for _round in range(MAX_ROUNDS):
        improved = False
        for index in zeros:
            keep = separators[index]
            local_best, local_pick = best_matched, keep
            local_chunks = best_chunks

            for candidate in SEPARATOR_CANDIDATES:
                separators[index] = candidate
                matched, chunk_count = score(separators)
                if matched > local_best:
                    local_best, local_pick, local_chunks = matched, candidate, chunk_count

            separators[index] = local_pick
            if local_best > best_matched:
                best_matched, best_chunks, improved = local_best, local_chunks, True

        if not improved:
            break

    return separators, best_matched, best_chunks


def diagnose(chunks: list) -> list:
    """
    拼之前先检查这批 chunk 本身完不完整。

    缺口和重复都不该静默通过——缺一块会拼出一份"中间少了一段但看着很连贯"
    的文档，比直接报错危险得多。返回问题描述列表，没有问题是空的。
    """
    problems = []
    ids = [chunk["chunk_id"] for chunk in chunks]

    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        problems.append(f"chunk_id 重复：{duplicates}")

    ordered = sorted(set(ids))
    gaps = [(a, b) for a, b in zip(ordered, ordered[1:]) if b != a + 1]
    if gaps:
        detail = "、".join(f"{a}→{b}（缺 {b - a - 1} 块）" for a, b in gaps)
        problems.append(f"chunk_id 不连续：{detail}")

    if ordered[0] != 0:
        problems.append(f"不是从 0 开始，前面缺了 {ordered[0]} 块")

    empty = [chunk["chunk_id"] for chunk in chunks if not chunk["text"].strip()]
    if empty:
        problems.append(f"空白 chunk：{empty}")

    return problems


def resolve_target(source: str) -> Path:
    """
    把 source 映射成 data/documents 下的落盘路径。

    source 是入库时记下来的相对路径，可能带子目录（data/documents/reviewed/
    下的文档，source 就是 "reviewed/xxx.md"）。但它是**库里的数据**，不能无条件
    当成可信路径用：绝对路径和 .. 都能指到 documents 目录外面去。
    """
    documents_root = Path(DOCUMENTS_PATH).resolve()
    target = (documents_root / source).resolve()

    if not target.is_relative_to(documents_root):
        raise ValueError(f"source 指向了 {documents_root} 之外，拒绝写入：{source!r}")
    if target == documents_root:
        raise ValueError(f"source 不是一个文件路径：{source!r}")
    return target


def describe_overlaps(overlaps: list) -> str:
    """重叠长度分布，用来看拼接质量"""
    if not overlaps:
        return "只有一块，没有接缝"

    zero = sum(1 for size in overlaps if size == 0)
    parts = [f"零重叠 {zero}"]
    for low, high in ((1, 9), (10, 29), (30, 49), (50, 10**9)):
        count = sum(1 for size in overlaps if low <= size <= high)
        if count:
            label = f"{low} 字以上" if high > 10**8 else f"{low}-{high} 字"
            parts.append(f"{label} {count}")
    return "，".join(parts) + f"（共 {len(overlaps)} 处接缝）"


def main():
    parser = argparse.ArgumentParser(description="从向量库还原 data/documents 下的语料")
    parser.add_argument("--force", action="store_true", help="已存在的文件也覆盖")
    parser.add_argument("--dry-run", action="store_true", help="只打印报告，不写盘")
    args = parser.parse_args()

    store = get_vector_store()
    print("=" * 62)
    print(f"后端：{type(store).__name__}")

    try:
        chunks = store.all_chunks()
        total = store.count()
    finally:
        store.close()

    if not chunks:
        print("向量库是空的，没有东西可以还原。")
        print("（新环境直接把原始文档放进 data/documents/ 再跑 ingest.py 就行）")
        return 1

    print(f"库里 {total} 条，读出 {len(chunks)} 条")
    if total != len(chunks):
        print("  ※ 读出的条数和 count() 对不上，还原结果可能不完整")
    print("=" * 62)

    grouped = group_by_source(chunks)
    written = skipped = 0
    total_matched = total_expected = 0

    for source in sorted(grouped):
        items = grouped[source]
        print(f"\n{source}")

        for problem in diagnose(items):
            print(f"  ※ {problem}")
            logger.warning("还原 %s：%s", source, problem)

        ordered = sorted(items, key=lambda chunk: chunk["chunk_id"])
        _text, overlaps, zeros = merge_chunks(items, {})

        # 先用"不填分隔符"拼一遍拿到零重叠接缝的位置。这个位置和填充策略无关：
        # 候选都是空白，而被拼的 chunk 两端已被 strip，空白不可能伪造出一个匹配。
        if zeros:
            separators, matched, rechunked = choose_separators(ordered, zeros)
        else:
            separators, matched, rechunked = {}, len(ordered), len(ordered)

        text, _overlaps, _zeros = merge_chunks(items, separators)
        char_total = sum(len(chunk["text"]) for chunk in items)

        print(f"  {len(items)} 块 / {char_total} 字 → 拼接后 {len(text)} 字")
        print(f"  接缝重叠：{describe_overlaps(overlaps)}")
        print(
            f"  保真度：重切后 {matched}/{len(ordered)} 块与库中原始块逐字一致"
            f"（{matched / len(ordered):.0%}），重切出 {rechunked} 块"
        )
        if matched < len(ordered) * FIDELITY_WARN:
            print("  ※ 保真度偏低。正文是完整的，但分段可能和原来不同——")
            print("    如果 CHUNK_SIZE / CHUNK_OVERLAP / SEPARATORS 在入库后被改过，")
            print("    这个判据本身就失效了，别把这个数字当成还原质量。")
            logger.warning("还原 %s 保真度仅 %.0f%%", source, matched / len(ordered) * 100)

        total_matched += matched
        total_expected += len(ordered)

        try:
            target = resolve_target(source)
        except ValueError as exc:
            print(f"  × 跳过：{exc}")
            logger.warning("跳过 %s：%s", source, exc)
            continue

        if target.exists() and not args.force:
            print(f"  已存在，跳过（要覆盖加 --force）：{target}")
            skipped += 1
            continue

        if args.dry_run:
            print(f"  [dry-run] 会写入 {target}")
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        # newline="" 关掉换行转换：不同平台的默认值不一样，写出来的字节会跟着变
        with open(target, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        print(f"  已写入 {target}")
        written += 1

    print("\n" + "=" * 62)
    if args.dry_run:
        print("dry-run 结束，没有写任何文件")
    else:
        print(f"还原完成：写入 {written} 份，跳过 {skipped} 份")
        if written:
            print("\n下一步：uv run python ingest.py")
            print("重切后的条数会和原来的不一样，这是预期的，不用去追平——")
            print("还原的正文是完整的，差别在接缝处被 strip 掉的空白。")
    print(f"整体保真度：{total_matched}/{total_expected} 块逐字一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
