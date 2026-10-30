"""
Supervisor 图：路由 → 并行 worker → 汇总 → （必要时回环）→ 出答案。

【图形状】一个 router，三个 worker 节点，一个 aggregate：

        ┌──────────────────────────────┐
        │                              │ （decision == "re_route"）
        ▼                              │
    START → router ─┬→ worker_knowledge ─┐
                    ├→ worker_order     ─┼→ aggregate ─┬→ END（answer / human）
                    └→ worker_analytics ─┘             │
                                                       └──┘

【为什么"三个 worker 节点都连上，各自判断自己在不在计划里"】
另一种写法是从 router 出来一条条件边，只连到"本轮要用的那几个"节点上。
这里没这么做，两个原因：

1. **条件边要动态决定连到谁**，LangGraph 里得靠 `Send`（map-reduce 那套）。
   计划在 router 里已经算出来了，fan-out 固定成三条、由节点自己判空，
   比在边上做动态分派更好读——**图的结构是静态的，面试时能把这张图直接画出来**。
2. **并行是白拿的**：三个节点在同一超步（superstep）里跑，不在计划里的那次
   几乎是零开销（一次 `if`），在计划里的就真的并发跑。不用写任何并发代码。

代价：图上永远有 3 条边，哪怕这次只用一个 worker；trace 里也会看到两个
"没参与"的节点。这是为了让**图结构可读**付的代价，我觉得值。

【回环为什么是有界的】
aggregate 只在 `round < MAX_ROUNDS` 时才会给出 re_route，round 由 router 自增。
所以从图上看是一个环，实际最多绕 MAX_ROUNDS 次。**一个没有出口的环是这套架构
最容易出的 bug**（"没答上来 → 重新路由 → 还是它"），所以有界这件事必须由代码
保证，不能靠提示词里写一句"不要重复路由"。
"""
import time

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from ecom.orchestrator.core import aggregate as aggregate_mod
from ecom.orchestrator.core.router import route
from ecom.orchestrator.core.state import CSState, make_initial_state
from ecom.orchestrator.settings import MAX_QUESTION_CHARS, MAX_ROUNDS
from ecom.orchestrator.utils import trace
from ecom.orchestrator.utils.intent import wants_human
from ecom.orchestrator.utils.logger import setup_logger
from ecom.orchestrator.utils.tokens import EMPTY, add
from ecom.orchestrator.workers.base import run, stamp_round, worker_names

logger = setup_logger(__name__)

NODE_ROUTER = "router"
NODE_AGGREGATE = "aggregate"

# 轨迹里给人看的短标签。**只影响展示**，不改事件里的原始值——消费方要判断
# 决策时仍然拿 `decision == "answer"`，不能依赖中文。
_ROUTER_SOURCE_LABELS = {"llm": "模型路由", "rule": "规则路由", "fallback": "兜底路由"}
_DECISION_LABELS = {"answer": "直接回答", "re_route": "重新路由", "human": "转人工"}


def _node_name(worker: str) -> str:
    return f"worker_{worker}"


def _router_node(state: dict, llm=None) -> dict:
    # 把上一轮也给它：没有它，"那能退吗"这类追问在这一层就没有主语（见 router 模块开头）。
    result = route(
        state.get("question") or "",
        tried=state.get("tried") or [],
        llm=llm,
        chat_history=state.get("chat_history") or [],
    )
    plan = result["plan"]
    round_no = int(state.get("round") or 0) + 1

    # tried 在这里一次性追加，**不由 worker 节点写**：三个 worker 并行，
    # 让它们各写一次 tried 就是三个节点同时写同一个 key（LangGraph 会直接报错，
    # 因为没配 reducer）。而计划本来就是 router 定的，router 顺手记下来最自然，
    # 也顺带把并发写这个坑绕开了。
    tried = list(state.get("tried") or []) + [w for w in plan if w not in (state.get("tried") or [])]

    # 这里**不**清空 results。它带 operator.add，往一个 add-型通道写 [] 等于没写；
    # 跨轮累加是 reducer 的语义，不是能靠赋值绕开的事。所以轮次只由 round 记录，
    # 由 aggregate 用 base.round_results 挑出本轮那份（那里有完整的来龙去脉）。
    return {
        "plan": plan,
        "plan_reason": result["reason"],
        "plan_confidence": result["confidence"],
        "router_source": result["source"],
        "round": round_no,
        "tried": tried,
        "router_tokens": result["tokens"],
    }


def _worker_node(name: str, fn=None, stream: bool = False):
    def node(state: dict) -> dict:
        plan = state.get("plan") or []
        if name not in plan:
            return {}  # 不在本轮计划里：什么都不做（见模块开头第 1 条）

        # 只有**本轮只有一个 worker** 时才把它的 token 直接流给用户。
        #
        # 多 worker 时不能流：那份答案是中间产物，汇总节点可能改写它，
        # 先显示再被整段替换就是"闪一下"。单 worker 时没有汇总可做，它
        # 就是最终答案——这时逐 token 流是真的，不是把长字符串切碎。
        emit = None
        if stream and len(plan) == 1:
            try:
                writer = get_stream_writer()
            except Exception as exc:  # noqa: BLE001 —— 拿不到就退回整段，不影响作答
                logger.warning("拿不到流式 writer，worker %s 按整段返回：%s", name, exc)
            else:
                # worker 只知道"一段文本"；事件形状是图这一层的契约（custom
                # 通道里混着 string 和 dict 的话，answer_stream 就得猜）。
                def emit(text: str) -> None:
                    writer({"type": "delta", "text": text})

        result = run(
            name,
            state.get("question") or "",
            state.get("chat_history") or [],
            fn=fn,
            customer_id=state.get("customer_id") or "",
            emit=emit,
        )
        return {"results": [stamp_round(result, state.get("round"))]}

    node.__name__ = _node_name(name)
    return node


def _aggregate_node(state: dict, fn=None, stream: bool = False) -> dict:
    """
    `stream=True` 时从 LangGraph 拿一个 writer，把汇总的增量直接推出去。

    **只有流式那条路才取 writer**：`invoke()` 里也有一个 writer（空操作），
    拿它不会报错，但会让汇总改走 `llm.stream()`——普通 `answer()` 的行为
    就悄悄变了。所以这里由 `build_graph(stream=...)` 显式开关，而不是
    "有 writer 就用"。
    """
    emit = None
    if stream:
        try:
            emit = get_stream_writer()
        except Exception as exc:  # noqa: BLE001 —— 拿不到就退化成不流，不影响作答
            logger.warning("拿不到流式 writer，本次按普通方式汇总：%s", exc)
    return aggregate_mod.aggregate(state, llm=fn, emit=emit)


def _after_aggregate(state: dict) -> str:
    return NODE_ROUTER if state.get("decision") == "re_route" else END


def build_graph(workers: dict = None, router_llm=None, synth_llm=None, stream: bool = False):
    """
    编译好的图。

    workers：{worker 名: 可调用的替身}。测试用；不传就按注册表取真的。
    router_llm / synth_llm：注入假模型（测试用），避免测试联网。
    stream：汇总节点是否把增量推给 `answer_stream()`（见 `_aggregate_node`）。
    """
    workers = workers or {}
    # 必须传 CSState（而不是裸 dict）：`results` 的合并语义写在它的
    # Annotated[list, operator.add] 上，换成 dict 就只剩"后写覆盖先写"，
    # 三个并行 worker 同时写 results 会直接报错。
    graph = StateGraph(CSState)

    graph.add_node(NODE_ROUTER, lambda s: _router_node(s, llm=router_llm))
    for name in worker_names():
        graph.add_node(
            _node_name(name), _worker_node(name, fn=workers.get(name), stream=stream)
        )
    graph.add_node(
        NODE_AGGREGATE, lambda s: _aggregate_node(s, fn=synth_llm, stream=stream)
    )

    graph.add_edge(START, NODE_ROUTER)
    for name in worker_names():
        # router → 每个 worker（并行分派），每个 worker → aggregate（汇聚）
        graph.add_edge(NODE_ROUTER, _node_name(name))
        graph.add_edge(_node_name(name), NODE_AGGREGATE)
    graph.add_conditional_edges(NODE_AGGREGATE, _after_aggregate, {NODE_ROUTER: NODE_ROUTER, END: END})

    return graph.compile()


def _prepare_question(question: str) -> str:
    """截断而不是拒绝：一句超长粘贴里通常前面就是问题，后面是误贴的日志。"""
    question = (question or "").strip()
    if len(question) > MAX_QUESTION_CHARS:
        logger.warning("问题超过 %d 字，已截断", MAX_QUESTION_CHARS)
        question = question[:MAX_QUESTION_CHARS]
    return question


def _recursion_limit() -> dict:
    # recursion_limit：LangGraph 自带的兜底。我们的环由 MAX_ROUNDS 有界，
    # 但编译器不知道这件事——真出现意外回环时，宁可由它抛错，也不要空转到天荒地老。
    return {"recursion_limit": MAX_ROUNDS * 4 + 6}


def _finalize(final: dict, question: str, request_id: str, record: bool) -> dict:
    """
    收尾：算总账、写 trace、打日志。

    **流式和非流式共用这一份**。两条路各写一份的话，token 汇总迟早会分叉
    （一边把重路由那轮算进去、另一边忘了），而 trace 一旦写两次，成本表的
    分母也会翻倍。这是"回答路径只有一条"这句话的落点。
    """
    # 记账要算**所有轮次**的调用，不是最后一轮的——重路由那一轮也花了钱，
    # 只算最后一轮会让成表比真实成本低。
    results = final.get("results") or []
    total = dict(EMPTY)
    for r in results:
        total = add(total, r.get("tokens") or EMPTY)
    total = add(total, final.get("router_tokens") or EMPTY)
    total = add(total, final.get("synth_tokens") or EMPTY)
    final["tokens"] = total

    if record:
        trace.record(
            {
                "request_id": request_id,
                "question": question[:200],
                "plan": final.get("plan"),
                "rounds": final.get("round"),
                "router_source": final.get("router_source"),
                "plan_confidence": final.get("plan_confidence"),
                "decision": final.get("decision"),
                "decision_reason": final.get("decision_reason"),
                "synth_source": final.get("synth_source"),
                "sources": final.get("sources"),
                "workers": trace.summarize_workers(results),
                "tokens": total,
            }
        )

    logger.info(
        "请求 %s：%d 轮，决策 %s（%s），token %s",
        request_id, final.get("round"), final.get("decision"),
        final.get("decision_reason"), total,
    )
    return final


# 用户明确要求人工时给的一句话。**工单号由 API 层补上**——图不知道身份、
# 也不知道队列，它只负责把"该转人工"这件事判出来。
HANDOFF_REQUESTED = "好的，需要人工客服。"


def _handoff_state(request_id: str) -> dict:
    """
    不跑图、直接转人工的 state。

    **不调模型**：用户已经说清要什么了，再路由一遍只是多花一次钱、多一个
    可能出错的地方。`tokens` 留空，账本如实记 0——这不是"漏算"。
    """
    return {
        "request_id": request_id,
        "question": "",
        "decision": "human",
        "answer": HANDOFF_REQUESTED,
        "sources": [],
        "decision_reason": "用户明确要求人工客服",
        "plan": [],
        "round": 0,
        "results": [],
        "chat_history": [],
        "tokens": dict(EMPTY),
    }


def answer(question: str, chat_history: list = None, workers: dict = None,
           router_llm=None, synth_llm=None, record: bool = True,
           customer_id: str = "") -> dict:
    """
    跑一次完整问答，返回最终 state（外加一个 tokens 汇总）。

    record=False 用于测试：不往 logs/requests.jsonl 里写。
    customer_id：已认证客户号；非空时订单 worker 只查这个客户名下的单。
    """
    question = _prepare_question(question)
    request_id = trace.new_request_id()
    # 明确要求人工：短路，不建图、不调模型。规则命中就转，理由见 utils/intent.py。
    if wants_human(question):
        return _finalize(_handoff_state(request_id), question, request_id, record)
    state = make_initial_state(
        question, chat_history, request_id=request_id, customer_id=customer_id
    )
    app = build_graph(workers=workers, router_llm=router_llm, synth_llm=synth_llm)
    final = app.invoke(state, config=_recursion_limit())
    return _finalize(final, question, request_id, record)


def answer_stream(question: str, chat_history: list = None, workers: dict = None,
                  router_llm=None, synth_llm=None, record: bool = True,
                  customer_id: str = ""):
    """
    跑一次完整问答，并把**最终答复的增量**和**思考过程**边生成边交出来（生成器）。

    事件契约（消费方只需要认这五种）：

        {"type": "start", "request_id": "..."}
        {"type": "thinking", "kind": "stage"|"reasoning", "text": "..."}
        {"type": "delta", "text": "..."}
        {"type": "reset"}   # 新一轮开始：上一轮流出的正文作废
        {"type": "final", "state": <和 answer() 返回的同一个 dict>,
         "thinking": "<完整思考过程>", "elapsed_ms": ...}

    【两种 thinking】
    - `stage`：路由、worker 完成/失败、最终决策。数据来自图每一步之后的
      完整 state，**是真实发生的事**，不是编出来的内心独白。
    - `reasoning`：模型自己吐出的思维链（provider 的 `reasoning_content`）。
      普通模型没有这一路，收不到就是没有，不补假内容。

    【哪些字是真的流出来的】
    两条路，都是"生成时就有"，不是把完成的字符串切碎：
    - **汇总模型**（多个 worker 出结论时）逐 token 流；
    - **单 worker** 时它就是最终答案，worker 支持 `on_token` 就直接把 token
      推出去（`_worker_node` 里 relay 到 custom 通道）。

    多 worker 时不流单个 worker 的答案：那份是中间产物，汇总会改写它，
    先显示再整段替换只会让界面闪一下。都没有可流的东西（直接转人工、
    worker 不支持 token、模型层整个挂了）才作为**一个** delta 整段交出。
    思考过程是另一条路：它从图每一步之后的真实 state 产生，所以单个
    worker 的等待时间也能被看见。

    【中途失败怎么办】
    汇总流到一半失败会退回拼接，此时最终的 `answer` 和已吐出的增量不一致。
    消费方必须以 `final.state["answer"]` 为准。异常本身**不吞**：模型挂了、
    图抛了，生成器就把异常抛给调用方，由它决定怎么告诉用户。

    【记账只有一处】
    收尾走 `_finalize()`，和 `answer()` 是同一份：trace 只写一条、token 只
    汇总一次。所以"加了一条流式路径"没有变成"两份成本账"。
    """
    question = _prepare_question(question)
    request_id = trace.new_request_id()

    started = time.perf_counter()
    yield {"type": "start", "request_id": request_id}

    emitted = False
    final_state = None
    thinking: list[str] = []

    def stage(text: str) -> dict:
        thinking.append(text)
        return {"type": "thinking", "kind": "stage", "text": text}

    # 明确要求人工：短路。事件形状保持和正常路径一致（start→thinking→
    # delta→final），消费方不需要为"这一轮没建图"写分支。
    if wants_human(question):
        final = _finalize(_handoff_state(request_id), question, request_id, record)
        yield stage("用户明确要求人工客服，跳过路由。\n")
        yield {"type": "delta", "text": final["answer"]}
        yield {
            "type": "final",
            "state": final,
            "thinking": "".join(thinking),
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
        return

    state = make_initial_state(
        question, chat_history, request_id=request_id, customer_id=customer_id
    )
    app = build_graph(
        workers=workers, router_llm=router_llm, synth_llm=synth_llm, stream=True
    )

    # 先给一条，用户按下发送之后立刻有东西可看——后面的阶段由图的每一步补。
    yield stage("正在理解问题，判断需要哪些单元…\n")

    seen_results = 0
    last_round = 0
    last_decision = ""
    # stream_mode 给两种：custom 是汇总节点推出来的增量，values 是每一步之后
    # 的完整 state。最后一帧 values 就是 answer() 里 invoke() 拿到的那个 dict。
    for mode, chunk in app.stream(
        state, config=_recursion_limit(), stream_mode=["custom", "values"]
    ):
        if mode == "custom":
            if not isinstance(chunk, dict):
                continue
            if chunk.get("type") == "delta":
                emitted = True
                yield chunk
            elif chunk.get("type") == "thinking":
                thinking.append(str(chunk.get("text") or ""))
                yield chunk
        elif mode == "values":
            final_state = chunk
            round_no = int(chunk.get("round") or 0)

            if round_no and round_no != last_round:
                last_round = round_no
                # 重路由：上一轮可能已经流出了半段答案（单 worker 直流的代价），
                # 那一段不再属于最终答复。显式让消费方清空，而不是靠下一次
                # delta 往上拼——拼出来的会是两轮答案的混合。
                if emitted:
                    emitted = False
                    yield {"type": "reset"}
                plan = chunk.get("plan") or []
                labels = "、".join(
                    aggregate_mod.WORKER_LABELS.get(name, name) for name in plan
                ) or "无"
                source = _ROUTER_SOURCE_LABELS.get(
                    str(chunk.get("router_source") or ""), "路由"
                )
                line = f"第 {round_no} 轮 · {source}：{labels}"
                confidence = chunk.get("plan_confidence")
                if isinstance(confidence, (int, float)) and confidence:
                    line += f"（置信度 {confidence:.2f}）"
                reason = str(chunk.get("plan_reason") or "").strip()
                if reason:
                    line += f"\n  {reason}"
                yield stage(line + "\n")

            results = chunk.get("results") or []
            if len(results) > seen_results:
                for result in results[seen_results:]:
                    label = aggregate_mod.WORKER_LABELS.get(
                        result.get("worker"), result.get("worker") or "单元"
                    )
                    if result.get("ok") is False:
                        line = f"「{label}」失败：{result.get('error') or '未知错误'}"
                    elif result.get("insufficient"):
                        line = f"「{label}」没有找到足够的信息"
                    else:
                        line = f"「{label}」完成（{int(result.get('elapsed_ms') or 0)} ms）"
                    yield stage(line + "\n")
                seen_results = len(results)

            decision = str(chunk.get("decision") or "")
            if decision and decision != last_decision:
                last_decision = decision
                line = f"决策：{_DECISION_LABELS.get(decision, decision)}"
                reason = str(chunk.get("decision_reason") or "").strip()
                if reason:
                    line += f"（{reason}）"
                yield stage(line + "\n")

    final = _finalize(final_state or {}, question, request_id, record)

    if not emitted and final.get("answer"):
        # 没有汇总可做（单结论 / 直接转人工）：把最终答复作为**一个**增量。
        # 这样消费方不必为"这次没流"写第二条渲染路径。
        yield {"type": "delta", "text": final["answer"]}

    yield {
        "type": "final",
        "state": final,
        "thinking": "".join(thinking),
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }
