"""CLI 入口的转人工提示：没有身份层就**明说没有人工**，不假装转接。

`main.py` 直接调 supervisor，不经过 API——没有身份、没有会话库，也就开不出
工单。supervisor 给的「好的，需要人工客服。」如果原样打出来，用户会以为有人
接了，然后永远等不到回复。这里钉住降级后的那句话，以及"真带了工单就不动它"
这个边界（将来 CLI 接上 API 的工单时，降级不能把真工单号盖掉）。

流式那条单独测，是因为占位话术会**先于** final 作为 delta 吐出来；终端打出去
的字收不回来，得在 `_run_streaming` 里扣到 final 再决定。
"""
import main
from ecom.orchestrator.core.supervisor import HANDOFF_REQUESTED, HANDOFF_UNAVAILABLE


def _human_state(**extra):
    return {
        "decision": "human",
        "answer": HANDOFF_REQUESTED,
        "sources": [],
        "tokens": {},
        **extra,
    }


def test_cli_does_not_pretend_a_human_was_reached(monkeypatch, capsys):
    monkeypatch.setattr(main, "answer", lambda question, history: _human_state())

    assert main.main(["转人工"]) == 0

    out = capsys.readouterr().out
    assert HANDOFF_UNAVAILABLE in out
    assert HANDOFF_REQUESTED not in out, "没有工单还打出'好的，需要人工客服'就是在骗人"


def test_cli_keeps_the_ticket_number_when_there_is_one(monkeypatch, capsys):
    """真带了 handoff 就原样保留——降级只处理"什么都没有"的那一种情况。"""
    ticket_answer = "已为你转接人工客服，工单号 T260101-ABC123，当前排在第 1 位。"
    monkeypatch.setattr(
        main,
        "answer",
        lambda question, history: _human_state(
            answer=ticket_answer,
            handoff={
                "ticket_id": "T260101-ABC123",
                "status": "open",
                "queue_position": 0,
            },
        ),
    )

    main.main(["转人工"])

    out = capsys.readouterr().out
    assert "T260101-ABC123" in out
    assert HANDOFF_UNAVAILABLE not in out


def test_cli_stream_never_prints_the_placeholder(monkeypatch, capsys):
    """流式会把占位话术先当 delta 吐出来。不扣住的话，用户先看到"好的，需要
    人工客服"，紧接着又被告知"这里没开通"——两句话互相打脸。"""

    def fake_stream(question, history):  # noqa: ANN001
        yield {"type": "start", "request_id": "req-test"}
        yield {"type": "delta", "text": HANDOFF_REQUESTED}
        yield {"type": "final", "state": _human_state(), "elapsed_ms": 1}

    monkeypatch.setattr(main, "answer_stream", fake_stream)

    main.main(["转人工", "--stream"])

    out = capsys.readouterr().out
    assert HANDOFF_UNAVAILABLE in out
    assert HANDOFF_REQUESTED not in out
