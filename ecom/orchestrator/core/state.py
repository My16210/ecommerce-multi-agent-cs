"""
共享状态。

【这是多 Agent 协作最容易被面试追问的一处，所以先把话说死】
Agent 之间**不传自然语言**。Worker 之间不互相对话，它们只跟状态对象打交道，
而且写进状态的是一份**结构化的结果契约**（见下面的 WorkerResult）。理由：

1. 自然语言消息要再解析一遍，等于把"理解"这件事做了两次，每次都可能出错，
   而且出错时很难定位是"上游表达不清"还是"下游理解错"。
2. 结构化结果能被程序直接消费——汇总节点可以按字段拼接、trace 能按字段聚合，
   不用再写一层"从散文里抠信息"的代码。
3. 上下文隔离的落点就在这里：worker 之间**不共享会话**，各自拿到 `question`
   和它自己需要的那点上下文。"最近几轮对话"是随调用一起传进来的**节点输入**
   （`chat_history`），不是谁都能改的共享状态——见下面 `CSState.chat_history` 的说明。

【为什么 results 要用 reducer】
worker 节点是**并行**执行的（同一超步），多个节点同时往 `results` 写。
LangGraph 对没有 reducer 的键会在并行写入时报错——这不是麻烦，是保护：
它逼你想清楚"两个节点同时写同一个键"到底该覆盖还是该合并。这里显然是合并。
"""
import operator
from typing import Annotated, TypedDict


class WorkerResult(TypedDict, total=False):
    """worker 的统一返回契约。所有 worker 必须返回这个形状。

    worker         worker 名（knowledge / order / analytics）
    ok             这次调用本身有没有成功（不含"答不出来"）
    answer         给人看的一段结论；失败时为 None
    data           结构化结果，各 worker 自定义（订单 dict、图表路径、召回条数…）
    sources        引用来源（文档名 / URL / 订单号），汇总时去重
    insufficient   **调用成功、但确实没答上来**。和 ok=False 是两件事：
                   ok=False 是"坏了"，insufficient=True 是"看过了，没有"。
                   aggregate 只对后者考虑重新路由——把"坏了"也拿去重路由，
                   就变成"服务一抖动就换个 worker 再抖一次"。
    error          失败原因（ok=False 时才有意义）
    elapsed_ms     这次调用耗时
    tokens         {"in": .., "out": ..}
    """

    worker: str
    ok: bool
    answer: str
    data: dict
    sources: list
    insufficient: bool
    error: str
    elapsed_ms: int
    tokens: dict


class CSState(TypedDict, total=False):
    # ——— 输入 ———
    question: str
    # [{"role": "user"/"assistant"/"memory", "content": ...}]
    # **这是节点输入，不是图状态**：由调用方塞进来，图自己不累加、不修改它。
    # 读它的有三个地方：router（指代消解，见 core/router.py 开头）、order worker
    # （抽实体）、knowledge worker（生成答案）。
    # **汇总节点（core/aggregate.py）刻意不读它**——那一步只做"各说各的、不替谁
    # 判对错"，给它历史等于给它素材去改写和脑补：多一次幻觉机会，零信息增益。
    # （"只在汇总时用一次"是仓库里说反了的旧说法，一并纠正。）
    chat_history: list

    # ——— 路由 ———
    plan: list  # 本轮要调用哪些 worker（worker 名的子集）
    plan_reason: str  # 路由这么决定的理由（进日志，也给面试官看）
    plan_confidence: float  # 0~1
    router_source: str  # "llm" / "rule" / "fallback"——路由结论是谁给的
    router_tokens: dict  # 路由本身花的 token（重试会累加，所以要单独记账）
    tried: list  # 已经调用过的 worker（跨轮累积，防止原地打转）
    round: int  # 第几轮（从 1 起，由 router 节点自增）

    # ——— 结果 ——
    # Annotated + operator.add：并行 worker 各自 append，最终合起来。
    # **注意它是跨轮累加、清不掉的**——往一个 add-型通道写 [] 等于没写。这是 reducer
    # 的语义，也是"并行写入不丢数据"的代价。所以每条结果都带 round 标记，
    # 汇总时用 `base.round_results` 把本轮的挑出来。见 workers/base.stamp_round。
    results: Annotated[list, operator.add]

    # ——— 汇总 ———
    decision: str  # "answer" / "re_route" / "human"
    decision_reason: str  # 为什么这么决定（进日志，也是面试时最该讲的一处）
    answer: str  # 最终给人的回答
    sources: list  # 去重后的引用来源
    # 汇总用了 LLM 还是直接拼接。它决定"这次答案有没有被改写过的可能"，
    # 是个该被记录的事实，不能只留在日志里。
    synth_source: str  # "llm" / "concat"
    synth_tokens: dict
    # 重路由时把已经攒下的话带进下一轮，防止下一轮空手而归时把它弄丢
    pending_note: str

    # ——— 可观测 ———
    request_id: str


def make_initial_state(question: str, chat_history: list = None, request_id: str = "") -> CSState:
    """两个入口（CLI / FastAPI）共用，避免抄多份之后漂移。"""
    return {
        "question": question,
        "chat_history": chat_history or [],
        "plan": [],
        "plan_reason": "",
        "plan_confidence": 0.0,
        "router_source": "",
        "router_tokens": {"in": 0, "out": 0},
        "tried": [],
        "round": 0,
        "results": [],
        "decision": "",
        "decision_reason": "",
        "answer": "",
        "sources": [],
        "synth_source": "",
        "synth_tokens": {"in": 0, "out": 0},
        "pending_note": "",
        "request_id": request_id,
    }
