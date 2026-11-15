"""单价表：模型名归一 + 算不出来返回 None + 三种「零」分开。"""
import pytest

from ecom.orchestrator.utils.pricing import (
    cost_of,
    cost_of_call,
    resolve_model,
    summarize_cost,
)


def test_resolve_model_maps_provider_names_to_price_buckets():
    assert resolve_model("qwen-plus") == "qwen"
    assert resolve_model("Qwen-Plus") == "qwen"
    assert resolve_model("deepseek-chat") == "deepseek"
    assert resolve_model("gpt-4o") == "gpt-4o"
    assert resolve_model("  QWEN-TURBO  ") == "qwen"
    assert resolve_model(None) is None
    assert resolve_model("") is None


def test_cost_of_normalizes_the_model_name_before_lookup():
    # qwen 单价故意留 None（未核实），两个名字都必须落到「未知」，而不是
    # 一个未知、一个查不到 key——后者看起来也像未知，但其实是两条不同的错。
    assert cost_of("qwen-plus", 100, 50) == cost_of("qwen", 100, 50)
    assert cost_of("qwen-plus", 100, 50) is None

    # deepseek 有价，这条才真正证明归一发生过：不归一会查不到 "deepseek-chat"。
    assert cost_of("deepseek-chat", 100, 50) == cost_of("deepseek", 100, 50)
    assert cost_of("deepseek-chat", 100, 50) is not None


def test_unknown_model_costs_none_not_zero():
    assert cost_of("some-new-model", 100, 50) is None


def test_cost_of_call_rule_with_zero_tokens_is_free_not_unknown():
    call = {"source": "rule", "model": None, "tokens_in": 0, "tokens_out": 0}
    assert cost_of_call(call) == 0.0
    assert cost_of_call(call) is not None


def test_cost_of_call_rule_with_tokens_goes_through_pricer():
    # 规则路径也可能带着用量（例如模型调过、结论退回规则）。不能被短路成 0。
    call = {
        "source": "rule", "model": "deepseek-chat",
        "tokens_in": 1000, "tokens_out": 1000,
    }
    assert cost_of_call(call) == cost_of("deepseek-chat", 1000, 1000)
    assert cost_of_call(call) > 0


def test_cost_of_call_unknown_without_model_is_unknown():
    assert cost_of_call(
        {"source": "unknown", "model": None, "tokens_in": 0, "tokens_out": 0}
    ) is None
    assert cost_of_call(
        {"source": "llm", "model": None, "tokens_in": 10, "tokens_out": 5}
    ) is None


def test_summarize_cost_keeps_known_and_unknown_separate():
    calls = [
        # 两条规则调用：已知，各 0 元
        {"source": "rule", "model": None, "tokens_in": 0, "tokens_out": 0},
        {"source": "rule", "model": None, "tokens_in": 0, "tokens_out": 0},
        # 一条有价的模型调用：已知
        {"source": "llm", "model": "deepseek-chat", "tokens_in": 1000, "tokens_out": 1000},
        # 两条未知：model=None / source=unknown
        {"source": "llm", "model": None, "tokens_in": 10, "tokens_out": 5},
        {"source": "unknown", "model": None, "tokens_in": 0, "tokens_out": 0},
    ]
    summary = summarize_cost(calls)
    assert summary["known_usd"] == pytest.approx(
        cost_of("deepseek-chat", 1000, 1000)
    )
    assert summary["unknown_calls"] == 2
    assert summary["complete"] is False
