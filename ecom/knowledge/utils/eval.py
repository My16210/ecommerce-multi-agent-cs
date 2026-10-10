"""
测评脚本
功能：测检索召回率，有数据才能改进

两套评测集，**粒度不同，回答的问题也不同**：

| | 期望答案有几个候选 | 能回答什么 |
|---|---|---|
| 内置 9 条（source 级） | 3 份文档 | 只能当冒烟测试：Top-5 里"沾到"正确的文档太容易 |
| 合成评测集（chunk 级） | 253 个 chunk | 才分得开好坏 |

source 级的 Recall@5 必然接近满分，**不是语料太小，是粒度太粗**——期望答案
只有 3 个候选时，随便召回 5 条都能命中。换成 chunk 级，候选变成 253 个，
同一套系统立刻量得出差别，而且不需要新语料。合成评测集由 build_eval_set.py
生成，落在 data/eval_set.jsonl。

**评测集只用 SILICONFLOW key（embedding）**。生成评测集才要 QWEN key，
跑评测不要——答疑链路还没通的时候也能先验证检索质量。

用法：
    uv run python -m ecom.knowledge.utils.eval                  # 内置 9 条
    uv run python -m ecom.knowledge.utils.eval --set            # 合成评测集
    uv run python -m ecom.knowledge.utils.eval --set --top-k 10
"""
import argparse
import json

from ecom.knowledge.retriever.retriever import retrieve
from ecom.knowledge.settings import EVAL_SET_PATH, HYBRID_SEARCH, TOP_K
from ecom.knowledge.vector_store import get_vector_store

# 小评测集：问题 → 期望召回的文件名
# 语料是 data/documents 下那三份教程笔记，换语料时这里要跟着改
#
# 这里刻意混了两类问题：
#   - **自然语言改写型**（前 6 条）：问法和原文措辞不一样，考的是语义理解，向量路的强项
#   - **术语精确型**（后 3 条）：问题里带着语料里罕见的专有名词（SqliteSaver / RRF /
#     RecursiveCharacterTextSplitter），考的是精确词面匹配，BM25 的强项
#
# 只测前一类的话，这个评测集**在原理上就测不出词汇检索**——两路都会拿满分，
# 看不出混合检索到底有没有用。补后一类不是为了把数字做好看，是因为一个只有
# 改写型问题的评测集，本来就不足以衡量一个检索系统。
#
# 这 9 条**留着**，而且不删：它们是手写的、每条都有人看过，是合成评测集出事时的
# 对照组。合成集再大也是模型出的，模型出的题集体跑偏（比如全押在某一类上）
# 时，这 9 条是唯一能证明"检索本身没坏"的东西。
TEST_CASES = [
    {"question": "LangGraph 里的节点、边、状态分别是什么？", "expected_source": "LangGraph基础.md"},
    {"question": "LangGraph 和 LangChain 有什么区别？", "expected_source": "LangGraph基础.md"},
    {"question": "怎么用 Checkpointer 保持多轮对话的上下文？", "expected_source": "LangGraph基础.md"},
    {"question": "大模型有哪些局限，RAG 是怎么解决的？", "expected_source": "RAG检索增强生成.md"},
    {"question": "什么是检索增强生成？", "expected_source": "RAG检索增强生成.md"},
    {"question": "RAGFlow 是做什么的？", "expected_source": "RAGFlow介绍.md"},
    # ↓ 术语型：这三个词在整个语料里只出现在对应的那一份文档里
    {"question": "SqliteSaver 是干什么用的？", "expected_source": "LangGraph基础.md"},
    {
        "question": "RRF 是怎么把两路结果融合到一起的？",
        "expected_source": "RAG检索增强生成.md",
    },
    {
        "question": "RecursiveCharacterTextSplitter 的分隔符怎么配？",
        "expected_source": "RAG检索增强生成.md",
    },
]

# 超过这个条数就只印没命中的题。一份两百多条的评测集全打出来没人看，
# 而"哪些题没命中"才是唯一能拿去改代码的信息。
VERBOSE_LIMIT = 20


def load_eval_set(path: str = None) -> list:
    """读合成评测集（JSONL）

    文件不存在是**正常状态**（还没跑过 build_eval_set.py），所以返回空表
    让调用方给一句人话提示，而不是抛 FileNotFoundError 让人去猜。
    """
    path = path or EVAL_SET_PATH
    try:
        with open(path, encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]
    except FileNotFoundError:
        return []


def match_positions(case: dict, results: list) -> list:
    """期望命中的那些结果排在第几位（从 1 数）

    两种粒度在这里分岔：
      - **source 级**（用例没有 expected_chunk_id）：来源文件名对了就算命中，
        返回值可能有多条（召回的 5 条里有 3 条来自期望文档）。
      - **chunk 级**（有 expected_chunk_id）：要来源和段号都对上才算，
        正常最多命中一条。

    返回值是个列表而不是单个名次，因为 source 级的"精度"指标要靠它数条数。
    """
    expected = case["expected_source"]
    chunk_id = case.get("expected_chunk_id")

    if chunk_id is None:
        return [i for i, hit in enumerate(results, 1) if hit["source"] == expected]
    return [
        i for i, hit in enumerate(results, 1)
        if hit["source"] == expected and hit["chunk_id"] == chunk_id
    ]


def _describe(results) -> str:
    return ", ".join(f"{hit['source'].removesuffix('.md')}#{hit['chunk_id']}" for hit in results)


def _seam_note(case, results) -> str:
    """没命中时，看看召回的是不是**同文档的相邻段**

    评测集是按 chunk 出的题，但相邻两段本来就可能都在讲同一件事（切分还带 50 字重叠）。
    出题时指定的是第 31 段，检索返回第 32 段——这不是检索错了，是**标注粒度**的
    固有噪声。把它标出来，免得把这类"假失手"当成真实退化去调参数。
    """
    chunk_id = case.get("expected_chunk_id")
    if chunk_id is None:
        return ""
    nearby = [
        hit["chunk_id"] for hit in results
        if hit["source"] == case["expected_source"] and abs(hit["chunk_id"] - chunk_id) <= 1
    ]
    if not nearby:
        return ""
    return f"    注：召回了同文档相邻段 {nearby}——可能是切分接缝，不是真失手"


def evaluate(top_k: int = TOP_K, cases: list = None, verbose: bool = None,
             title: str = None) -> dict:
    """
    跑检索，看 Top-K 里有没有命中期望的那条，算召回率

    三个指标一起看，因为它们在细语料上各有盲区：
      - **Recall@K**：Top-K 里有没有期望的那条。
      - **MRR**：命中的那条排第几。能区分"排第 1"和"排第 5"。
      - **精度**：召回的每条里有多少条相关。chunk 级只有一个标准答案时它退化成
        Recall/K，看不出新东西，所以只在 source 级报。

    chunk 级还要**按题型分组**看。改写型和术语型考的是两种不同的能力，混在一起的
    总分会把它们互相抵消掉——这正是 A/B 对比时要看的东西。

    返回：{"level", "recall", "mrr", "precision", "hits", "total", "groups"}
    """
    cases = list(TEST_CASES if cases is None else cases)
    level = "chunk" if any(c.get("expected_chunk_id") is not None for c in cases) else "source"
    if verbose is None:
        verbose = len(cases) <= VERBOSE_LIMIT

    store = get_vector_store()
    try:
        if store.count() == 0:
            print("向量库是空的，请先运行：uv run python ingest.py")
            return {
                "level": level, "recall": 0.0, "mrr": 0.0, "precision": 0.0,
                "hits": 0, "total": len(cases), "groups": {},
            }

        # 把当前的检索配置印出来：A/B 对比时这两行要一起贴进文档，
        # 否则事后没人分得清那份数字是哪套配置跑出来的
        # title 只有 gold_set.py 会传：那份集子也是 chunk 级，但不是"合成的"，
        # 沿用上面那句措辞会把人工标注说成模型自产，正好踩在这个项目最忌讳的点上。
        name = title or ('合成（chunk 级）' if level == 'chunk' else '内置（source 级）')
        print(f"混合检索：{'开' if HYBRID_SEARCH else '关'}（.env 的 HYBRID_SEARCH）")
        print(f"评测集：{name} {len(cases)} 条，Top-{top_k} 召回：\n")

        hits = 0
        ranks = []
        relevant_hits = 0
        returned = 0
        groups = {}

        for index, case in enumerate(cases, 1):
            # retrieve() 抛异常时**不兜底**，让它直接冒出来。
            # 测评的意义就是发现问题：embedding 全挂时如果退化成"未命中"，
            # 只会让人以为是检索质量差，把真正的故障盖住。
            results = retrieve(case["question"], store, top_k=top_k)
            positions = match_positions(case, results)

            hit = bool(positions)
            rank = positions[0] if hit else None
            hits += hit
            ranks.append(rank)
            relevant_hits += len(positions)
            returned += len(results)

            group = groups.setdefault(
                case.get("type", "全部"), {"hits": 0, "total": 0, "rr": 0.0}
            )
            group["total"] += 1
            group["hits"] += hit
            group["rr"] += 1 / rank if rank else 0.0

            if verbose or not hit:
                mark = f"命中(第{rank}位)" if hit else "未命中"
                target = f"{case['expected_source']}#{case.get('expected_chunk_id', '')}"
                print(f"[{index}] {mark}  期望：{target}")
                print(f"    问题：{case['question']}")
                print(f"    召回：{_describe(results)}")
                note = _seam_note(case, results)
                if note:
                    print(note)
                print()

        total = len(cases)
        recall = hits / total
        mrr = sum(1 / rank for rank in ranks if rank) / total
        precision = relevant_hits / returned if returned else 0.0

        print("-" * 56)
        print(f"Recall@{top_k} = {hits}/{total} = {recall:.1%}")
        print(f"MRR        = {mrr:.3f}（满分 1.000，命中越靠前越高）")

        if level == "source":
            # 语料只有三份文档，Top-5 里"沾到"正确的那份太容易，Recall 必然接近满分、
            # 区分不出好坏。精度是同一批结果里的另一半信息。
            print(
                f"精度       = {relevant_hits}/{returned} = {precision:.1%}"
                f"（召回的每条里，有多少条真的来自期望文档）"
            )

        if len(groups) > 1:
            print("\n按题型分开看（总分会把两类能力互相抵消）：")
            for name, group in sorted(groups.items()):
                label = {"paraphrase": "改写型", "term": "术语型"}.get(name, name)
                print(
                    f"  {label}  {group['total']:>3} 条   "
                    f"Recall@{top_k} = {group['hits'] / group['total']:.1%}   "
                    f"MRR = {group['rr'] / group['total']:.3f}"
                )

        if recall < 0.8:
            print(
                "\n指标偏低，可以试：调大 TOP_K、调大 VECTOR_RECALL_K / BM25_RECALL_K、"
                "调整切分参数，或检查 .env 里的 HYBRID_SEARCH 是否为 true、"
                "再调 BM25_K1 / BM25_B / RRF_K"
            )

        return {
            "level": level,
            "recall": recall,
            "mrr": mrr,
            "precision": precision,
            "hits": hits,
            "total": total,
            "groups": groups,
        }
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser(description="检索召回率测评")
    parser.add_argument("--set", dest="use_set", action="store_true",
                        help="用合成评测集（data/eval_set.jsonl）而不是内置的 9 条")
    parser.add_argument("--set-path", default=None, help="指定评测集文件")
    parser.add_argument("--top-k", type=int, default=TOP_K, help=f"默认 {TOP_K}")
    args = parser.parse_args()

    if args.use_set or args.set_path:
        cases = load_eval_set(args.set_path)
        if not cases:
            print("没找到合成评测集。先运行：uv run python build_eval_set.py")
            return
        evaluate(top_k=args.top_k, cases=cases)
    else:
        evaluate(top_k=args.top_k)


if __name__ == "__main__":
    main()
