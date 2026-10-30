"""整张图：并行分派 / 结果合并 / 回环有界 / 全链路不联网也能跑通。"""
import json

from ecom.orchestrator.core.supervisor import answer, answer_stream
from ecom.orchestrator.settings import MAX_ROUNDS
from ecom.orchestrator.utils.trace import summarize_workers
from tests.orchestrator.conftest import ExplodingLLM, FakeLLM


def workers(order=None, knowledge=None, analytics=None):
    """三个都塞满——留空的那个会被 base 走真注册表，测试就跑去连真库了。"""
    return {
        "order": order or (lambda q, h: {"answer": "订单结论", "sources": ["52644"]}),
        "knowledge": knowledge or (lambda q, h: {"answer": "知识库结论", "sources": ["faq.md"]}),
        "analytics": analytics or (lambda q, h: {"answer": "数据分析结论", "sources": ["sales.csv"]}),
    }


def plan_json(*plans):
    return [f'{{"workers": {p}, "confidence": 1.0}}' for p in plans]


# ---------------------------------------------------------------------------
# 并行分派与合并
# ---------------------------------------------------------------------------
def test_two_workers_run_in_one_round_and_get_merged():
    final = answer(
        "订单 52644 到哪了，顺便问下退货政策",
        workers=workers(),
        router_llm=FakeLLM('{"workers": ["order", "knowledge"], "confidence": 1.0}'),
        synth_llm=FakeLLM("合并后的答复"),
        record=False,
    )
    assert final["round"] == 1
    assert sorted(r["worker"] for r in final["results"]) == ["knowledge", "order"]
    assert final["decision"] == "answer"
    assert final["answer"] == "合并后的答复"


def test_workers_outside_the_plan_are_not_called_at_all():
    final = answer(
        "订单 52644 到哪了",
        workers=workers(),
        router_llm=FakeLLM('{"workers": ["order"], "confidence": 1.0}'),
        record=False,
    )
    assert [r["worker"] for r in final["results"]] == ["order"]
    assert final["answer"] == "订单结论"


def test_a_crashing_worker_becomes_a_footnote_not_a_500():
    def boom(q, h):
        raise RuntimeError("数据库文件被删了")

    final = answer(
        "订单 52644 到哪了，顺便问下退货政策",
        workers=workers(order=boom),
        router_llm=FakeLLM('{"workers": ["order", "knowledge"], "confidence": 1.0}'),
        synth_llm=FakeLLM("合并后的答复"),
        record=False,
    )
    assert final["decision"] == "answer"
    assert "知识库" in final["answer"] or "合并后" in final["answer"]
    assert "订单系统这块暂时查不了" in final["answer"]
    broken = [r for r in final["results"] if not r["ok"]]
    assert broken and "数据库文件被删了" in broken[0]["error"]


# ---------------------------------------------------------------------------
# 回环
# ---------------------------------------------------------------------------
def test_an_insufficient_worker_triggers_exactly_one_more_round():
    def ask_for_id(q, h):
        return {"answer": "请提供订单号", "insufficient": True, "sources": []}

    final = answer(
        "我这单到哪了",
        workers=workers(order=ask_for_id),
        router_llm=FakeLLM(*plan_json('["order"]', '["knowledge"]')),
        record=False,
    )
    assert final["round"] == 2
    assert final["tried"] == ["order", "knowledge"]
    assert "知识库结论" in final["answer"]
    # 第一轮攒下的澄清话术没有被丢掉
    assert "请提供订单号" in final["answer"]


def test_the_loop_cannot_outlive_max_rounds():
    calls = []

    def never_answers(q, h):
        calls.append(1)
        return {"answer": "没查到", "insufficient": True, "sources": []}

    fake = workers(order=never_answers, knowledge=never_answers, analytics=never_answers)
    final = answer(
        "帮我查点东西",
        workers=fake,
        # 路由每轮都往回指同一个 worker：tried 会把它过滤掉，计划变空，环就此停住
        router_llm=FakeLLM(*plan_json('["order"]', '["order"]', '["order"]', '["order"]')),
        record=False,
    )
    assert final["round"] <= MAX_ROUNDS
    assert final["decision"] in {"answer", "human"}
    assert len(calls) == 1  # order 只在第一轮真的跑过


def test_results_do_not_leak_from_one_round_into_the_next():
    """
    results 的 reducer 是 operator.add，跨轮**只增不减**（往 add-型通道写 [] 等于没写）。
    所以汇总前必须按 round 过滤，否则第一轮的结论会在第二轮被再汇总一次——
    症状是"同一段话说两遍"，而且只在第二轮出现，最难查。
    """
    from ecom.orchestrator.workers.base import round_results

    def order_fn(q, h):
        return {"answer": "订单结论", "insufficient": True, "sources": []}

    final = answer(
        "我这单到哪了",
        workers=workers(order=order_fn),
        router_llm=FakeLLM(*plan_json('["order"]', '["knowledge"]')),
        record=False,
    )
    # 账本上两轮都在（都真的调用过，成本要算），但只有第二轮的进汇总
    assert [(r["worker"], r["round"]) for r in final["results"]] == [("order", 1), ("knowledge", 2)]
    assert [r["worker"] for r in round_results(final["results"], final["round"])] == ["knowledge"]
    assert final["answer"].count("订单结论") == 1  # 只在"补充"里出现一次，不是被汇总了两遍


# ---------------------------------------------------------------------------
# 兜底：模型这条路整个不可用时，图还能不能跑通
# ---------------------------------------------------------------------------
def test_the_whole_graph_still_works_with_the_model_layer_dead():
    # 路由挂 → 关键词规则接住（"52644" 是强信号）；只有一个 worker 出结论
    # → 汇总不调模型。所以两个模型角色全挂，链路依然是通的。
    dead = ExplodingLLM()
    final = answer(
        "订单 52644 到哪了",
        workers=workers(),
        router_llm=dead,
        synth_llm=dead,
        record=False,
    )
    assert final["router_source"] == "rule"
    assert final["plan"] == ["order"]
    assert final["answer"] == "订单结论"


def test_an_empty_question_still_terminates_with_something_to_say():
    final = answer("", workers=workers(), router_llm=ExplodingLLM(), record=False)
    assert final["decision"] in {"answer", "human"}
    assert final["answer"].strip()


# ---------------------------------------------------------------------------
# 记账与 trace
# ---------------------------------------------------------------------------
def test_tokens_cover_router_workers_and_synthesis():
    final = answer(
        "订单 52644 到哪了，顺便问下退货政策",
        workers=workers(),
        router_llm=FakeLLM('{"workers": ["order", "knowledge"], "confidence": 1.0}'),
        synth_llm=FakeLLM("合并后的答复"),
        record=False,
    )
    # 路由 10/5 + 汇总 10/5；假 worker 不上报用量，按 0 算
    assert final["tokens"] == {"in": 20, "out": 10}


def test_one_request_writes_exactly_one_trace_line():
    # 必须从 ecom.orchestrator.utils.trace 取**它自己那个** LOGS_DIR：
    # conftest 把它换到了 tmp_path，而 ecom.settings.LOGS_DIR 还指着仓库里真实的
    # logs/。两个模块各有一份名字相同的绑定，取错了就会去读线上那份。
    from ecom.orchestrator.utils import trace

    final = answer(
        "订单 52644 到哪了",
        workers=workers(),
        router_llm=FakeLLM('{"workers": ["order"], "confidence": 1.0}'),
    )
    path = trace.LOGS_DIR / "requests.jsonl"
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").strip().splitlines()]
    assert len(lines) == 1
    entry = lines[0]
    assert entry["request_id"] == final["request_id"]
    assert entry["decision"] == "answer"
    assert entry["router_source"] == "llm"
    assert entry["workers"]["order"]["calls"] == 1


# ---------------------------------------------------------------------------
# 流式：同一份图、同一份收尾，只多了增量
# ---------------------------------------------------------------------------
def _two_worker_kwargs():
    # 每次新建模型：FakeLLM 按调用次数往下取，跨两次运行复用会让断言失去意义。
    return {
        "workers": workers(),
        "router_llm": FakeLLM('{"workers": ["order", "knowledge"], "confidence": 1.0}'),
        "synth_llm": FakeLLM("合并后的答复"),
    }


def test_streaming_ends_with_the_same_state_as_invoke():
    """**这是流式那条路唯一真正要守的契约**：和 `answer()` 跑出来一模一样。

    内容、来源、决策、token 汇总、request_id 全部对齐。流式不是"另一条回答
    路径"，只是同一条路径多了一个出口。
    """
    plain = answer("订单 52644 到哪了，顺便问下退货政策", record=False, **_two_worker_kwargs())
    events = list(
        answer_stream("订单 52644 到哪了，顺便问下退货政策", record=False, **_two_worker_kwargs())
    )
    final = events[-1]["state"]

    assert events[0]["type"] == "start"
    assert events[0]["request_id"] == final["request_id"]
    assert events[-1]["type"] == "final"
    assert final["answer"] == plain["answer"]
    assert final["sources"] == plain["sources"]
    assert final["decision"] == plain["decision"]
    assert final["tokens"] == plain["tokens"]
    assert "elapsed_ms" in events[-1]


def test_streaming_emits_stage_thinking_before_the_answer():
    """思考过程是真实 state 的投影：路由、每个 worker、决策都要出现，
    并且和答案正文分成两路（thinking 不能被拼进 answer）。"""
    events = list(
        answer_stream("订单 52644 到哪了，顺便问下退货政策", record=False, **_two_worker_kwargs())
    )
    thinking = [e for e in events if e["type"] == "thinking"]
    texts = [e["text"] for e in thinking]

    assert thinking and all(e["kind"] == "stage" for e in thinking)
    assert texts[0].startswith("正在理解问题")
    assert any("第 1 轮" in t and "订单" in t and "知识库" in t for t in texts)
    assert any("「订单」完成" in t for t in texts)
    assert any(t.startswith("决策：直接回答") for t in texts)

    first_thinking = next(i for i, e in enumerate(events) if e["type"] == "thinking")
    first_delta = next(i for i, e in enumerate(events) if e["type"] == "delta")
    assert first_thinking < first_delta, "等待期间就该有过程可看，而不是答完才出现"
    assert events[-1]["thinking"] == "".join(texts)


def test_streamed_deltas_rebuild_the_final_answer():
    events = list(
        answer_stream("订单 52644 到哪了，顺便问下退货政策", record=False, **_two_worker_kwargs())
    )
    deltas = [e for e in events if e["type"] == "delta"]

    assert len(deltas) > 1, "两个结论要走汇总模型，应当收到多块"
    assert "".join(d["text"] for d in deltas) == events[-1]["state"]["answer"]


def test_a_single_conclusion_arrives_as_exactly_one_delta():
    """没有汇总可做时**不切段**：一整段（假流式是在演延迟，不是在优化体验）。"""
    events = list(
        answer_stream(
            "订单 52644 到哪了",
            workers=workers(),
            router_llm=FakeLLM('{"workers": ["order"], "confidence": 1.0}'),
            record=False,
        )
    )
    deltas = [e for e in events if e["type"] == "delta"]
    assert [d["text"] for d in deltas] == ["订单结论"]


def test_a_single_worker_that_supports_tokens_streams_instead_of_one_chunk():
    """单 worker 时它就是最终答案，worker 的 token 可以真流——不是切字符串。"""
    def streaming_answer(q, h, on_token=None):
        assert on_token is not None, "单 worker 的流式路径必须把回调传进来"
        on_token("逐")
        on_token("字")
        return {"answer": "逐字结论", "sources": []}

    events = list(
        answer_stream(
            "退货政策是什么",
            workers=workers(knowledge=streaming_answer),
            router_llm=FakeLLM('{"workers": ["knowledge"], "confidence": 1.0}'),
            record=False,
        )
    )

    deltas = [e["text"] for e in events if e["type"] == "delta"]
    assert deltas == ["逐", "字"]
    assert events[-1]["state"]["answer"] == "逐字结论"


def test_a_reroute_resets_the_previous_rounds_streamed_text():
    """上一轮流出的中间产物不能被拼进下一轮；消费方需要一个显式的 reset。"""
    def insufficient(q, h, on_token=None):
        if on_token:
            on_token("这是第一轮的中间答复")
        return {"answer": "这是第一轮的中间答复", "insufficient": True, "sources": []}

    events = list(
        answer_stream(
            "我这单到哪了",
            workers=workers(knowledge=insufficient),
            router_llm=FakeLLM(*plan_json('["knowledge"]', '["order"]')),
            record=False,
        )
    )

    types = [e["type"] for e in events]
    assert "reset" in types
    assert types.index("reset") > types.index("delta")
    # 第一轮的澄清话术按设计会留在最终答复里（见上面"不丢澄清"那条测试）；
    # reset 管的是客户端已经画出来的那一段，不是服务端的拼接规则。
    assert "订单结论" in events[-1]["state"]["answer"]


def test_streaming_records_exactly_one_trace_line(monkeypatch):
    from ecom.orchestrator.core import supervisor

    recorded = []
    monkeypatch.setattr(supervisor.trace, "record", recorded.append)

    events = list(
        answer_stream("订单 52644 到哪了，顺便问下退货政策", record=True, **_two_worker_kwargs())
    )

    assert len(recorded) == 1, "流式和非流式共用 _finalize，不该写两条 trace"
    assert recorded[0]["request_id"] == events[-1]["state"]["request_id"]
    assert recorded[0]["tokens"] == events[-1]["state"]["tokens"]


def test_streaming_reports_usage_exactly_once_on_completion():
    """正常看完一条流，`on_usage` 只收到一次，数值和 final.state 对齐。"""
    usage = []
    events = list(
        answer_stream(
            "订单 52644 到哪了",
            workers=workers(),
            router_llm=FakeLLM('{"workers": ["order"], "confidence": 1.0}'),
            record=False,
            on_usage=usage.append,
        )
    )
    assert usage == [events[-1]["state"]["tokens"]]


def test_streaming_reports_usage_when_the_client_disconnects_early():
    """断线不能把已经花掉的路由 token 漏掉——这是预算的绕道口。"""
    usage = []
    stream = answer_stream(
        "订单 52644 到哪了",
        workers=workers(),
        router_llm=FakeLLM(
            '{"workers": ["order"], "confidence": 1.0}',
            usage={"input_tokens": 7, "output_tokens": 3},
        ),
        record=False,
        on_usage=usage.append,
    )
    # 路由那一帧已经处理完（token 已花），但 worker 还没跑完就断开。
    for event in stream:
        if event.get("type") == "thinking" and "第 1 轮" in event.get("text", ""):
            break
    assert usage == [], "还没到收尾，不该提前记账"
    stream.close()
    assert usage == [{"in": 7, "out": 3}]


def test_a_dead_synth_model_still_yields_a_complete_answer():
    """汇总挂了 → 退回拼接。流式这条路也必须给出完整答复，而不是半句。"""
    events = list(
        answer_stream(
            "订单 52644 到哪了，顺便问下退货政策",
            workers=workers(),
            router_llm=FakeLLM('{"workers": ["order", "knowledge"], "confidence": 1.0}'),
            synth_llm=ExplodingLLM(),
            record=False,
        )
    )
    final = events[-1]["state"]

    assert final["decision"] == "answer"
    assert "【订单】" in final["answer"]
    deltas = "".join(e["text"] for e in events if e["type"] == "delta")
    assert deltas.strip() == final["answer"], "降级路径也要把完整答复交出去"


def test_summarize_workers_aggregates_per_worker():
    from ecom.orchestrator.workers.base import error_result, ok_result

    summary = summarize_workers(
        [
            ok_result("order", elapsed_ms=120),
            error_result("order", "超时", elapsed_ms=20000),
            ok_result("knowledge", elapsed_ms=800),
        ]
    )
    assert summary["order"] == {"calls": 2, "ok": 1, "ms": 20120, "errors": ["超时"]}
    assert summary["knowledge"]["calls"] == 1


# ---------------------------------------------------------------------------
# 明确要求人工：短路，不建图、不调模型
# ---------------------------------------------------------------------------
def test_explicit_human_request_skips_the_graph_and_costs_nothing():
    """「转人工」是一个动作，不是一道要理解的问题。

    这里把 router/synth 都换成会抛错的 ExplodingLLM：如果哪天有人把短路去掉，
    这个测试会立刻炸，而不是"多花一次调用、结果看起来还行"。
    """
    from ecom.orchestrator.core.supervisor import HANDOFF_REQUESTED

    final = answer(
        "转人工",
        workers=workers(),
        router_llm=ExplodingLLM(),
        synth_llm=ExplodingLLM(),
        record=False,
    )

    assert final["decision"] == "human"
    assert final["answer"] == HANDOFF_REQUESTED
    assert final["tokens"].get("in", 0) == 0
    assert final["tokens"].get("out", 0) == 0
    assert final["results"] == []


def test_explicit_human_request_streams_the_same_event_shape():
    """流式消费方不需要为"这一轮没建图"写分支：start → delta → final。"""
    events = list(
        answer_stream(
            "我要人工客服",
            workers=workers(),
            router_llm=ExplodingLLM(),
            synth_llm=ExplodingLLM(),
            record=False,
        )
    )

    kinds = [e["type"] for e in events]
    assert kinds[0] == "start"
    assert kinds[-1] == "final"
    assert any(e["type"] == "delta" for e in events)
    assert events[-1]["state"]["decision"] == "human"
