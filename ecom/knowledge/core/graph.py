"""
LangGraph工作流
功能：把「知识库检索」和「联网搜索」并行跑，两路都回来后再汇总生成答案

              START
             /      \
            ▼        ▼
    retrieve_node  web_search_node      ← 同一超步并行执行
            \\        /
             ▼      ▼
            answer_node                 ← 等两路都完成才执行
                 │
                 ▼
                END

检索和 rerank 合在 retrieve_node 内部：两者是串行数据依赖，
拆成两个图节点会让它和 web_search 的并行关系变得别扭。
"""
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from ecom.knowledge.core.state import RAGState
from ecom.knowledge.generator.answer import generate_answer
from ecom.knowledge.retriever.reranker import rerank
from ecom.knowledge.retriever.retriever import retrieve
from ecom.knowledge.utils.logger import setup_logger
from ecom.knowledge.vector_store import get_vector_store
from ecom.knowledge.vector_store.base import VectorStore
from ecom.knowledge.web_search.web_search import web_search

logger = setup_logger(__name__)

# 生成失败时的兜底文案。用常量而不是散落的字面量，
# 因为 main.py 的提问判定要靠它识别"这一轮是故障，不是一次正常提问"，
# 免得把服务抖动当成知识库缺口存进待审核队列。
ANSWER_FALLBACK = "抱歉，答案生成服务暂时不可用，请稍后重试。"


def _collect_sources(state: RAGState) -> list:
    """汇总两路来源并去重，保持出现顺序"""
    sources = []

    for chunk in state.get("contexts") or []:
        source = chunk.get("source")
        if source and source not in sources:
            sources.append(source)

    for chunk in state.get("web_contexts") or []:
        url = chunk.get("url")
        if url and url not in sources:
            sources.append(url)

    return sources


def build_graph(store: VectorStore = None, checkpointer=None):
    """
    构建RAG工作流

    store：向量库实例，默认按配置新建一个（见 get_vector_store）
    checkpointer：状态存储器，默认 InMemorySaver（进程退出即丢）；
                  需要跨进程保留多轮会话，请传 SqliteSaver
    """
    store = store or get_vector_store()
    checkpointer = checkpointer or InMemorySaver()

    def retrieve_node(state: RAGState) -> dict:
        """知识库这一路：向量检索 + rerank"""
        question = state["question"]
        try:
            hits = retrieve(question, store)
            return {"contexts": rerank(question, hits)}
        except Exception as exc:
            # 知识库只是两路之一。这一步失败（embedding 接口抖动等）就退回空
            # contexts，让 answer_node 仍能靠联网结果和历史作答，
            # 而不是把 app.invoke() 整个抛出去、把会话打断。
            #
            # 兜底放在图这个边界、而不是放到 retrieve() 里面：那样 eval.py 在
            # embedding 全挂时会把每道题都印成"未命中"，反而把故障盖住。
            # retrieve() 保持"失败就抛"的严格语义，直调用方自己决定怎么办。
            logger.warning("知识库检索失败，本轮退化为无知识库结果：%s", exc)
            return {"contexts": []}

    def web_search_node(state: RAGState) -> dict:
        """联网搜索这一路"""
        return {"web_contexts": web_search(state["question"])}

    def answer_node(state: RAGState) -> dict:
        """两路都完成后：汇总生成答案，并把本轮对话追加进历史"""
        question = state["question"]
        try:
            answer = generate_answer(
                question,
                state.get("contexts") or [],
                state.get("web_contexts") or [],
                chat_history=state.get("chat_history"),
            )
        except Exception as exc:
            # 生成是最后一步，没有下一级可降级，只能给一句明确的兜底文案。
            # 关键是**照样**往下走：图必须跑到 END，返回一个有意义的状态，
            # 而不是把异常抛给 app.invoke() 的调用方。
            logger.warning("答案生成失败，返回兜底文案：%s", exc)
            answer = ANSWER_FALLBACK

        # 默认覆盖语义下，历史必须在这里自己追加后整体写回。
        # 生成失败时也照常追加——否则一次故障会让多轮记忆断档。
        history = list(state.get("chat_history") or [])
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "sources": _collect_sources(state),
            "chat_history": history,
        }

    graph = StateGraph(RAGState)
    graph.add_node(retrieve_node)
    graph.add_node(web_search_node)
    graph.add_node(answer_node)

    # 两路从 START 同时出发（并行），再各自汇入 answer_node（汇合）
    graph.add_edge(START, "retrieve_node")
    graph.add_edge(START, "web_search_node")
    graph.add_edge("retrieve_node", "answer_node")
    graph.add_edge("web_search_node", "answer_node")
    graph.add_edge("answer_node", END)

    return graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    app = build_graph()
    app.get_graph().print_ascii()
