"""整张图：并行分派 / 结果合并 / 回环有界 / 全链路不联网也能跑通。"""
from ecom.orchestrator.core.supervisor import answer
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
    import json

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
