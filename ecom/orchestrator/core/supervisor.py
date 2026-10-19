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
from langgraph.graph import END, START, StateGraph

from ecom.orchestrator.core import aggregate as aggregate_mod
from ecom.orchestrator.core.router import route
from ecom.orchestrator.core.state import CSState, make_initial_state
from ecom.orchestrator.settings import MAX_QUESTION_CHARS, MAX_ROUNDS
from ecom.orchestrator.utils import trace
from ecom.orchestrator.utils.logger import setup_logger
from ecom.orchestrator.utils.tokens import EMPTY, add
from ecom.orchestrator.workers.base import run, stamp_round, worker_names

logger = setup_logger(__name__)

NODE_ROUTER = "router"
NODE_AGGREGATE = "aggregate"


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


def _worker_node(name: str, fn=None):
    def node(state: dict) -> dict:
        if name not in (state.get("plan") or []):
            return {}  # 不在本轮计划里：什么都不做（见模块开头第 1 条）
        result = run(name, state.get("question") or "", state.get("chat_history") or [], fn=fn)
        return {"results": [stamp_round(result, state.get("round"))]}

    node.__name__ = _node_name(name)
    return node


def _aggregate_node(state: dict, fn=None) -> dict:
    return aggregate_mod.aggregate(state, llm=fn)


def _after_aggregate(state: dict) -> str:
    return NODE_ROUTER if state.get("decision") == "re_route" else END


def build_graph(workers: dict = None, router_llm=None, synth_llm=None):
    """
    编译好的图。

    workers：{worker 名: 可调用的替身}。测试用；不传就按注册表取真的。
    router_llm / synth_llm：注入假模型（测试用），避免测试联网。
    """
    workers = workers or {}
    # 必须传 CSState（而不是裸 dict）：`results` 的合并语义写在它的
    # Annotated[list, operator.add] 上，换成 dict 就只剩"后写覆盖先写"，
    # 三个并行 worker 同时写 results 会直接报错。
    graph = StateGraph(CSState)

    graph.add_node(NODE_ROUTER, lambda s: _router_node(s, llm=router_llm))
    for name in worker_names():
        graph.add_node(_node_name(name), _worker_node(name, fn=workers.get(name)))
    graph.add_node(NODE_AGGREGATE, lambda s: _aggregate_node(s, fn=synth_llm))

    graph.add_edge(START, NODE_ROUTER)
    for name in worker_names():
        # router → 每个 worker（并行分派），每个 worker → aggregate（汇聚）
        graph.add_edge(NODE_ROUTER, _node_name(name))
        graph.add_edge(_node_name(name), NODE_AGGREGATE)
    graph.add_conditional_edges(NODE_AGGREGATE, _after_aggregate, {NODE_ROUTER: NODE_ROUTER, END: END})

    return graph.compile()


def answer(question: str, chat_history: list = None, workers: dict = None,
           router_llm=None, synth_llm=None, record: bool = True) -> dict:
    """
    跑一次完整问答，返回最终 state（外加一个 tokens 汇总）。

    record=False 用于测试：不往 logs/requests.jsonl 里写。
    """
    question = (question or "").strip()
    if len(question) > MAX_QUESTION_CHARS:
        # 截断而不是拒绝：一句超长粘贴里通常前面就是问题，后面是误贴的日志。
        logger.warning("问题超过 %d 字，已截断", MAX_QUESTION_CHARS)
        question = question[:MAX_QUESTION_CHARS]

    request_id = trace.new_request_id()
    state = make_initial_state(question, chat_history, request_id=request_id)
    app = build_graph(workers=workers, router_llm=router_llm, synth_llm=synth_llm)

    # recursion_limit：LangGraph 自带的兜底。我们的环由 MAX_ROUNDS 有界，
    # 但编译器不知道这件事——真出现意外回环时，宁可由它抛错，也不要空转到天荒地老。
    final = app.invoke(state, config={"recursion_limit": MAX_ROUNDS * 4 + 6})

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
