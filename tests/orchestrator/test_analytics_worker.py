"""analytics worker：把子进程的用量按 provider 拆成 call 记录。"""
from ecom.orchestrator.workers.analytics import _calls_from_payload


def test_payload_is_split_into_one_call_per_provider():
    payload = {
        "tokens": {"in": 100, "out": 10},
        "token_detail": {
            "deepseek": {"in": 80, "out": 8},
            "qwen": {"in": 20, "out": 2},
        },
        "models": {"deepseek": "deepseek-chat", "qwen": "qwen-plus"},
        "elapsed_ms": 1234,
    }
    calls = _calls_from_payload(payload)

    assert [call["model"] for call in calls] == ["deepseek-chat", "qwen-plus"]
    # 拆开后的 token 之和必须等于原子进程给的合计——否则成本表会少算/多算。
    assert sum(call["tokens_in"] for call in calls) == payload["tokens"]["in"]
    assert sum(call["tokens_out"] for call in calls) == payload["tokens"]["out"]
    # 子进程只报总耗时，记在第一条上；第二条为 0，避免相加时翻倍。
    assert calls[0]["latency_ms"] == 1234
    assert calls[1]["latency_ms"] == 0
    assert all(call["source"] == "llm" and call["usage_given"] for call in calls)


def test_provider_without_tokens_gets_no_call():
    payload = {
        "tokens": {"in": 80, "out": 8},
        "token_detail": {
            "deepseek": {"in": 80, "out": 8},
            "qwen": {"in": 0, "out": 0},
        },
        "models": {"deepseek": "deepseek-chat", "qwen": "qwen-plus"},
        "elapsed_ms": 500,
    }
    calls = _calls_from_payload(payload)

    assert len(calls) == 1
    assert calls[0]["model"] == "deepseek-chat"


def test_payload_without_detail_falls_back_to_a_single_call():
    payload = {"tokens": {"in": 10, "out": 5}, "elapsed_ms": 100}
    calls = _calls_from_payload(payload)

    assert len(calls) == 1
    assert calls[0]["model"] is None
    assert calls[0]["source"] == "llm"
    assert calls[0]["latency_ms"] == 100


def test_empty_payload_is_unknown_not_free():
    calls = _calls_from_payload({})

    assert len(calls) == 1
    assert calls[0]["source"] == "unknown"
    assert calls[0]["usage_given"] is False
