"""
知识库 worker：售后政策 / 制度 / 商品 FAQ 问答。

**调用的是 sibling 仓库 rag_knowledge_base，不复制它的代码。**
它已经有完整的混合检索（向量+BM25→RRF）、三级降级 rerank、强制引用，
在这里重写一份只会得到一份更差的、而且会和原版慢慢分叉。

【进程内 import，不做子进程——这是唯一一处为速度让路的地方】
知识库是热路径（几乎每次提问都会走到），子进程要多付一次 Python 启动 +
chromadb 客户端初始化的钱（实测秒级）。而它是个**纯库**：retrieve / generate_answer
都是无副作用的函数，给一个进程边界带不来任何隔离收益。所以放进进程。
代价见 pyproject：本项目的依赖里得带上它那几个运行时依赖。

【不 import 它的 graph.py】
只 import retrieve / rerank / generate_answer / get_vector_store 这四个入口。
不 import graph 有两个好处：一是绕开 ddgs 那一路依赖（我们用不到联网搜索），
二是天然避开 `answer_node` 里的 `record_question` 副作用——评测和线上都试过，
再走一遍那条路会把答过的题再写进待审核队列，污染人工审核的数据。
"""
import sys
from pathlib import Path

from cs.settings import RAG_KB_ROOT
from cs.utils.logger import setup_logger

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

_loaded: dict = {}


def _load_rag():
    """
    把 sibling 仓库挂上 sys.path 再 import。

    只在第一次调用时执行。路径不存在时抛的错要写清楚——这是面试官 clone 下来
    最可能撞上的第一个问题（两个仓库不在同一层目录）。
    """
    if _loaded:
        return _loaded

    root = Path(RAG_KB_ROOT)
    if not (root / "src" / "retriever" / "retriever.py").exists():
        raise RuntimeError(
            f"找不到知识库仓库：{root}。请把它和本项目放在同一层目录，"
            "或在 .env 里设 RAG_KB_ROOT=/绝对路径/rag_knowledge_base"
        )

    if str(root) not in sys.path:
        # 插到最前面：本项目自己的包只叫 cs/，既不占 src/ 也不占 config/
        # （包名要足够独特的全部理由，见 cs/settings.py 开头）。
        sys.path.insert(0, str(root))

    from src.core.graph import ANSWER_FALLBACK
    from src.generator.answer import generate_answer
    from src.retriever.reranker import rerank
    from src.retriever.retriever import retrieve
    from src.vector_store import get_vector_store

    _assert_imports_came_from(root)

    _loaded.update(
        retrieve=retrieve,
        rerank=rerank,
        generate_answer=generate_answer,
        get_vector_store=get_vector_store,
        answer_fallback=ANSWER_FALLBACK,
    )
    logger.info("已接入知识库仓库：%s", root)
    return _loaded


def _assert_imports_came_from(root: Path) -> None:
    """
    确认刚 import 进来的这两个模块**真的来自知识库仓库**。

    为什么值得专门查一次：同名顶层包撞车时 Python **不报错**。它会安安静静地把
    另一个仓库的同名模块给你，然后你要么拿到一个缺失属性的 ImportError（还算
    幸运），要么拿到一个同名但语义不同的东西继续跑下去（灾难）。这条断言把
    "撞名"从"某个字段莫名其妙不对"提前成"启动时就告诉你是哪两个包撞了"。

    真出过事：本项目原来的配置包叫 `config/`，正好和 rag 的 `config/` 同名，
    于是 rag 的 16 个模块 import 到了我们的配置——`ImportError: cannot import
    name 'LOGS_PATH' from 'config.settings'`。所以这里连 `config` 一起查。

    查 `sys.modules` 而不是再 import 一次：这两个模块是**调用方刚 import 进来的**
    （就在上面几行），一定已经在 sys.modules 里。再 import 一遍反而绕进
    `import a.b.c` 的包层级机制，把一件"看看表"的事做成"再走一遍导入流程"。
    """
    for name in ("src.core.graph", "config.settings"):
        module = sys.modules.get(name)
        if module is None:
            continue
        origin = Path(getattr(module, "__file__", "") or "")
        if not origin.is_relative_to(root):
            raise RuntimeError(
                f"模块 {name} 实际来自 {origin}，不在知识库仓库 {root} 里。"
                "这说明本进程里已经有一个同名顶层包被别的代码 import 过——"
                "两个仓库的顶层包名撞了。请把本项目的同名包改名"
                "（同一进程里，顶层包名就是全局命名空间）。"
            )


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
    rag = _load_rag()

    # 1) 混合检索（内部含向量+BM25→RRF；向量那一路失败会抛出来，交给 base 记成 ok=False）
    hits = rag["retrieve"](question, rag["get_vector_store"]())
    # 2) rerank（三级降级，最差也会按 score 截断，不会返回空）
    top = rag["rerank"](question, hits)
    top_score = _top_vector_score(hits)

    # 3) 生成。走的是图里那个同一个 generate_answer，不是复制一份逻辑——
    #    复制的那份只要有一处和生产不一样，评出来的结果就开始骗人。
    answer = rag["generate_answer"](question, top, [], list(chat_history or []))
    if answer == rag["answer_fallback"]:
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
