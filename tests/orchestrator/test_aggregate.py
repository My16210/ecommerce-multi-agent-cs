"""汇总节点：三桶分类 / 单结论不调模型 / 汇总失败降级 / 决策的优先级。"""
import pytest

from ecom.orchestrator.core import aggregate as agg
from ecom.orchestrator.workers.base import error_result, ok_result
from tests.orchestrator.conftest import ExplodingLLM, FakeLLM


def state(**kwargs):
    base = {"question": "q", "plan": ["order"], "tried": ["order"], "round": 1, "results": []}
    base.update(kwargs)
    # worker 结果带 round 标记（生产代码里由 worker 节点盖），aggregate 靠它挑出本轮的。
    base["results"] = [dict(r, round=base["round"]) for r in base.get("results") or []]
    return base


def usable(worker, answer=None, sources=None):
    return ok_result(worker, answer=answer or f"{worker} 的结论", sources=sources or [worker])


def thin(worker, answer):
    return ok_result(worker, answer=answer, insufficient=True)


def broken(worker, error="RuntimeError: 连不上"):
    return error_result(worker, error)


# ---------------------------------------------------------------------------
# 三桶分类
# ---------------------------------------------------------------------------
def test_buckets_are_sorted_by_worker_for_stable_output():
    buckets = agg._split([usable("order"), usable("knowledge"), usable("analytics")])
    assert [r["worker"] for r in buckets["usable"]] == ["analytics", "knowledge", "order"]


def test_ok_but_silent_result_counts_as_thin_not_as_usable():
    # ok=True、insufficient=False、正文却是空的——契约被违反。
    # 当成"没答上来"，**不能静默丢掉**：丢掉的结果是用户看到一条空回答。
    buckets = agg._split([ok_result("order", answer="   ")])
    assert buckets["usable"] == []
    assert len(buckets["thin"]) == 1


# ---------------------------------------------------------------------------
# 合并
# ---------------------------------------------------------------------------
def test_single_conclusion_never_calls_the_model():
    # 没东西可合并时调模型 = 凭空多一次幻觉机会、零信息增益。
    exploding = ExplodingLLM()
    result = agg.aggregate(state(results=[usable("order", "订单 52644 已签收")]), llm=exploding)
    assert result["answer"] == "订单 52644 已签收"
    assert result["synth_source"] == "concat"
    assert exploding.calls == 0


def test_two_conclusions_are_merged_by_the_model():
    llm = FakeLLM("订单 52644 已签收。退货政策方面，七天无理由。")
    result = agg.aggregate(state(results=[usable("order"), usable("knowledge")]), llm=llm)
    assert result["synth_source"] == "llm"
    assert result["answer"].startswith("订单 52644")
    assert llm.call_count == 1


def test_merge_falls_back_to_concatenation_when_the_model_dies():
    result = agg.aggregate(state(results=[usable("order"), usable("knowledge")]), llm=ExplodingLLM())
    assert result["synth_source"] == "concat"
    # 降级后仍要能分清哪句是哪来的——两段不同系统的话连排会读成一件事。
    assert "【订单】" in result["answer"]
    assert "【知识库】" in result["answer"]


def test_merge_falls_back_when_the_model_returns_an_empty_body():
    result = agg.aggregate(state(results=[usable("order"), usable("knowledge")]), llm=FakeLLM("   "))
    assert result["synth_source"] == "concat"
    assert result["answer"].strip()


def test_sources_are_deduped_but_keep_their_order():
    result = agg.aggregate(
        state(results=[usable("order", sources=["52644", "100003"]), usable("knowledge", sources=["52644"])]),
        llm=ExplodingLLM(),
    )
    assert result["sources"] == ["52644", "100003"]


# ---------------------------------------------------------------------------
# 决策
# ---------------------------------------------------------------------------
def test_a_broken_worker_does_not_kill_the_answer():
    result = agg.aggregate(
        state(results=[usable("order", "订单 52644 已签收"), broken("analytics")]),
        llm=ExplodingLLM(),
    )
    assert result["decision"] == "answer"
    assert "订单 52644 已签收" in result["answer"]
    # 用户看到的是人话，不是异常原文——异常原文等于把内部结构报给外面
    assert "RuntimeError" not in result["answer"]
    assert "数据分析这块暂时跑不了" in result["answer"]


def test_insufficient_with_untried_workers_triggers_exactly_one_more_round():
    result = agg.aggregate(state(results=[thin("order", "请提供订单号")]))
    assert result["decision"] == "re_route"
    # 已经攒下的话必须带进下一轮，否则下一轮空手而归时这句就没了
    assert result["pending_note"] == "请提供订单号"


def test_no_more_rounds_left_means_we_answer_with_what_we_have():
    result = agg.aggregate(state(round=2, results=[thin("order", "请提供订单号")]))
    assert result["decision"] == "answer"
    assert "请提供订单号" in result["answer"]


def test_all_workers_down_does_not_reroute_it_handsoff():
    # 全挂 = 系统性的坏，换个 worker 只会在同一个网络上再挂一次。
    result = agg.aggregate(state(plan=["order", "analytics"], results=[broken("order"), broken("analytics")]))
    assert result["decision"] == "answer"
    assert agg.HANDOFF in result["answer"]
    assert "订单系统这块暂时查不了" in result["answer"]


def test_empty_plan_is_not_retried_because_it_would_be_empty_again():
    # plan 为空 → 这轮什么都没跑 → 重来一遍还是空（tried 已经排除了它们）。
    result = agg.aggregate(state(plan=[], tried=["order"], results=[]))
    assert result["decision"] == "human"
    assert result["answer"] == agg.HANDOFF_ONLY


def test_pending_note_survives_a_final_empty_round():
    result = agg.aggregate(state(plan=[], tried=["order"], results=[], pending_note="请提供订单号"))
    assert "请提供订单号" in result["answer"]


def test_a_usable_answer_still_carries_the_earlier_clarification():
    result = agg.aggregate(
        state(plan=["knowledge"], tried=["order", "knowledge"], round=2,
              results=[usable("knowledge", "七天无理由退货。")], pending_note="请提供订单号"),
        llm=ExplodingLLM(),
    )
    assert "七天无理由退货" in result["answer"]
    assert "请提供订单号" in result["answer"]


@pytest.mark.parametrize("decision", ["answer", "human", "re_route"])
def test_every_branch_reports_why(decision):
    # decision_reason 不是装饰：面试时"为什么这次转人工"要靠它回答，日志里也只有它。
    cases = {
        "answer": state(results=[usable("order")]),
        "human": state(plan=[], tried=["order", "knowledge", "analytics"], results=[]),
        "re_route": state(results=[thin("order", "请提供订单号")]),
    }
    result = agg.aggregate(cases[decision], llm=ExplodingLLM())
    assert result["decision"] == decision
    assert result["decision_reason"]
