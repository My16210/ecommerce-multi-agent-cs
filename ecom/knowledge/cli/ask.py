"""
入口文件
功能：启动RAG问答系统，用户输入问题，返回答案

工作流本身在 ecom/knowledge/core/graph.py（检索 ‖ 联网搜索 → 汇总生成），
这里只负责会话管理：用 SqliteSaver 存状态，同一个会话ID可以跨进程接着聊。

另外负责一件事：答案打印完之后，让大模型判断这次提问值不值得沉淀进知识库，
值得的进待审核队列（用 review.py 审）。这一步**不在 LangGraph 图里**——
放进图节点会被 invoke 同步等待，用户要多等一次 LLM 往返才看到答案。
"""
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from ecom.knowledge.core.graph import build_graph
from ecom.knowledge.review.judge import record_question
from ecom.knowledge.settings import CHAT_DB_PATH, ENABLE_QUESTION_REVIEW
from ecom.knowledge.utils import streaming
from ecom.knowledge.utils.logger import setup_logger
from ecom.knowledge.vector_store import get_vector_store

logger = setup_logger(__name__)


def _printer(collector: list):
    """
    造一个"边收边打"的汇。

    写成工厂函数而不是在循环里直接 def 一个闭包，是因为后者会捕获循环变量——
    ruff 的 B023 正好盯着这个。这里的调用其实每次都在同一轮内同步发生，
    不会出错，但把汇的构造拿出来，语义就不再依赖"什么时候被调用"了。
    """
    def emit(text: str) -> None:
        collector.append(text)
        print(text, end="", flush=True)

    return emit


def _capture_question(question: str, result: dict) -> None:
    """
    判定提问价值，值得沉淀的话把结果印出来。

    判定本身（含所有守卫）在 judge.record_question 里，和 app.py 共用同一份；
    这里只负责命令行该长什么样。判定结果不会直接进向量库——唯一写入口是 ingest.py。
    """
    info = record_question(question, result)
    if not info:
        return

    verdict = info["verdict"]
    if verdict["worthy"]:
        action = "新增" if info["is_new"] else "更新"
        print(f"\n[待审核] 这次提问被判为值得沉淀（{verdict['category']}，"
              f"分数 {verdict['score']:.2f}），已{action}到队列 #{info['row_id']}")
        print(f"  理由：{verdict['reason']}")
        print(f"  审核：uv run python review.py show {info['row_id']}")


def main():
    print("=" * 50)
    print("企业知识库问答系统")
    print("=" * 50)

    # 初始化向量库（后端由 settings.VECTOR_BACKEND 决定，默认 chroma）
    store = get_vector_store()
    print(f"向量库后端：{type(store).__name__}")

    # store 一拿到就进 try：下面那个 count()==0 的提前 return 也要走到 finally，
    # 否则那条路径上连接不会释放。
    try:
        if store.count() == 0:
            print("向量库是空的，请先运行：uv run python ingest.py")
            return

        Path(CHAT_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False：LangGraph 可能在后台线程里操作这个连接
        conn = sqlite3.connect(CHAT_DB_PATH, check_same_thread=False)
        try:
            saver = SqliteSaver(conn)
            # 建表。全新机器上 data/chat_history.db 不存在（它被 gitignore 了），
            # 不调用 setup() 第一次 invoke 就会撞上 "no such table: checkpoints"。
            # 内部是 CREATE TABLE IF NOT EXISTS，重复调用无害。
            saver.setup()
            app = build_graph(store, checkpointer=saver)

            session = input("会话ID（回车使用 default）：").strip() or "default"
            config = {"configurable": {"thread_id": session}}
            print(f"当前会话：{session}（用同一个会话ID再次启动可接着上次聊）")

            while True:
                question = input("\n请输入你的问题（输入q退出）：").strip()
                if question.lower() == "q":
                    break
                if not question:
                    continue

                # 命令行也逐字出：汇直接往 stdout 写。
                # 注意汇只影响"什么时候看到"，不看"看到什么"——下面的兜底
                # 与 app.py 里 placeholder 的处理是同一套道理。
                collected: list = []
                print("\n答案：", end="", flush=True)

                # 只传 question —— 历史由 checkpointer 从会话状态里恢复。
                # 如果把 chat_history 也传进来，会覆盖掉累积的历史，多轮就失效了。
                with streaming.sink(_printer(collected)):
                    result = app.invoke({"question": question}, config=config)

                # 一个 token 都没出来（生成失败、兜底成 ANSWER_FALLBACK）时补印一次，
                # 否则这一轮就只剩"答案："三个字。正常情况不会走到这里。
                if not collected:
                    print(result["answer"], end="")
                print()
                if result.get("sources"):
                    print("\n引用来源：")
                    for source in result["sources"]:
                        print(f"  - {source}")

                # 答案已经给到用户了，才轮到判定——顺序不能反
                if ENABLE_QUESTION_REVIEW:
                    _capture_question(question, result)
        finally:
            conn.close()
    finally:
        # 契约里声明了 close()，入口就该履行。Milvus 要关 gRPC channel。
        store.close()


if __name__ == "__main__":
    main()
