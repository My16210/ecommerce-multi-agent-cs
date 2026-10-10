"""
知识库 worker：售后政策 / 制度 / 商品 FAQ 问答。

**合并之后，这里就是 `ecom.knowledge` 的直接调用。**

在这之前它是一段接缝：把 sibling 仓库 rag_knowledge_base 挂上 `sys.path`、
再 import、import 完还要回头检查"我拿到的到底是不是它的模块"。现在没有
"别人的包"了，那一段（`_load_rag` 的路径操作 + `_assert_imports_came_from`
那道运行时闸门）整段作废、删掉。这是合并真正换来的东西——
不是少写几行，是少一类**不报错的错误**。

【进程内调用，不走子进程】

知识库是热路径（几乎每次提问都会走到），子进程要多付一次 Python 启动 +
chromadb 客户端初始化的钱（实测秒级）。而它是个**纯库**：retrieve /
generate_answer 都是无副作用的函数，给一个进程边界带不来任何隔离收益。

这一点和 analytics worker 是**故意不一样**的，而且理由在合并后反而更清楚了：
那个要执行模型生成的代码，需要一个**能被真杀死**的边界；这个不需要。
两种接法不是不一致，是各自的账算下来不一样。

【只拿四个入口，不跑它的图】

拿 retrieve / rerank / generate_answer / get_vector_store。调用的是**函数**，
不是 `build_graph()`——不是"不 import graph.py"（`ANSWER_FALLBACK` 这个常量
确实还从那里拿，见下面的 import），而是**不跑那张图**。

区别很实际：跑了图就会走到 `answer_node` 里的 `record_question`，把这次问的题
写进人工待审核队列。线上客服每问一次就往审核队列里塞一条，那条队列就废了。
"""
from ecom.knowledge.core.graph import ANSWER_FALLBACK
from ecom.knowledge.generator.answer import generate_answer
from ecom.knowledge.retriever.reranker import rerank
from ecom.knowledge.retriever.retriever import retrieve
from ecom.knowledge.vector_store import get_vector_store
from ecom.orchestrator.utils.logger import setup_logger

logger = setup_logger(__name__)

# 有效答案的最低检索相似度。低于它就把 insufficient 置 True，让 supervisor
# 考虑换个 worker 试试。
#
# **这个阈值是启发式，不是判据。** 依据来自 rag 项目自己的评测：真·不可答的问题
# 实测余弦在 0.53~0.64，可答的在 0.66 以上；但两段有重叠，已知存在误报
# （文档"在讲同一件事"却没给答案时，余弦照样能到 0.66）。所以这里取 0.5 这个
# 偏保守的值——**宁可漏报**：漏报只是少触发一次重路由，误报会让本该由知识库
# 回答的问题被踢去别的 worker，后者更糟。
KB_MIN_SCORE = 0.5


def _top_vector_score(hits: list) -> float:
    """
    取检索结果的最高**余弦相似度**。

    **别读成 `score`。** rag 项目的 `score` 是 RRF 融合分，量纲是 1/(K+rank)，
    实测恒在 0.03 上下——拿它比 0.5 的阈值永远为假，自检就成了摆设。
    余弦在 `vector_score` 字段里。
    """
    best = 0.0
    for hit in hits or []:
        value = hit.get("vector_score")
        if value is None:
            continue
        try:
            best = max(best, float(value))
        except (TypeError, ValueError):
            continue
    return best


def run_knowledge(question: str, chat_history: list = None) -> dict:
    # 1) 混合检索（内部含向量+BM25→RRF；向量那一路失败会抛出来，交给 base 记成 ok=False）
    hits = retrieve(question, get_vector_store())
    # 2) rerank（三级降级，最差也会按 score 截断，不会返回空）
    top = rerank(question, hits)
    top_score = _top_vector_score(hits)

    # 3) 生成。调的是知识库自己那个 generate_answer，不是复制一份逻辑——
    #    复制的那份只要有一处和生产不一样，评出来的结果就开始骗人。
    answer = generate_answer(question, top, [], list(chat_history or []))
    if answer == ANSWER_FALLBACK:
        # 兜底文案 = 生成服务故障，是"坏了"不是"不知道"。抛出去让 base 记 ok=False。
        raise RuntimeError("知识库答案生成服务不可用（返回了兜底文案）")

    sources = []
    for hit in top:
        source = hit.get("source")
        if source and source not in sources:
            sources.append(source)

    insufficient = top_score < KB_MIN_SCORE
    if insufficient:
        logger.info("知识库最高相似度 %.3f 低于 %.2f，标记为未能回答", top_score, KB_MIN_SCORE)

    return {
        "answer": answer,
        "sources": sources,
        "insufficient": insufficient,
        "data": {
            "recalled": len(hits),
            "kept": len(top),
            "top_score": round(top_score, 4),
            # 让上层能一眼看出"这句话是照着哪几段说的"
            "contexts": [h.get("chunk_id") for h in top],
        },
    }
