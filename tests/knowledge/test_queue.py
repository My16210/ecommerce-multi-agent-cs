"""
待审核队列测试。

每条用例都把队列指向 tmp_path 下的独立 SQLite 文件，不碰真实的
data/review_queue.db（那里有真实的待审记录）。
"""
import pytest

from ecom.knowledge.review import queue


@pytest.fixture
def que(monkeypatch, tmp_path):
    """把队列指到临时库文件

    必须 patch queue 模块**自己的** REVIEW_DB_PATH：connect() 是在调用时
    读这个模块级名字的。

    connect() 现在会缓存连接，所以前后都要 reset_connection()：
    否则第二条用例拿到的是上一条那条已连到别的库文件上的连接。
    """
    monkeypatch.setattr(queue, "REVIEW_DB_PATH", str(tmp_path / "review_queue.db"))
    queue.reset_connection()
    yield queue
    queue.reset_connection()


def _result(answer="答案", contexts=None, web_contexts=None, sources=None):
    return {
        "answer": answer,
        "contexts": contexts if contexts is not None else [],
        "web_contexts": web_contexts if web_contexts is not None else [],
        "sources": sources if sources is not None else [],
    }


def _verdict(worthy=True, category="kb_gap", score=0.8, reason="理由"):
    return {"worthy": worthy, "category": category, "score": score, "reason": reason}


def test_fingerprint_ignores_whitespace_and_case(que):
    assert queue.fingerprint("  退货 政策 ") == queue.fingerprint("退货政策")
    assert queue.fingerprint("Refund Policy") == queue.fingerprint("refundpolicy")


def test_enqueue_inserts_then_updates_the_same_row(que):
    row_id, is_new = que.enqueue("退货政策是什么", _result(), _verdict())
    assert is_new is True

    row_id_again, is_new_again = que.enqueue("退货政策是什么", _result(), _verdict())
    assert is_new_again is False
    assert row_id_again == row_id

    assert que.get_item(row_id)["ask_count"] == 2
    assert len(que.list_items(worthy_only=False, status=None)) == 1


def test_review_decision_survives_a_repeat_question(que):
    """人工批准过的问题又被问到，status 不能被冲回 pending

    这是知识库回灌链路里最要紧的一条：人已经审过的结论，
    不该被下一次提问推翻。写成 upsert 之后尤其要防这个。
    """
    row_id, _ = que.enqueue("退货政策是什么", _result(), _verdict())
    que.set_status(row_id, "approved", note="已收录")

    que.enqueue("退货政策是什么", _result(), _verdict())

    item = que.get_item(row_id)
    assert item["status"] == "approved"
    assert item["review_note"] == "已收录"
    assert item["doc_path"] is None


def test_stats_counts_by_status(que):
    first, _ = que.enqueue("问题一", _result(), _verdict())
    que.enqueue("问题二", _result(), _verdict())
    que.set_status(first, "rejected")

    assert que.stats() == {"pending": 1, "rejected": 1}


def test_list_items_filters_by_worthy(que):
    que.enqueue("值得的", _result(), _verdict(worthy=True))
    que.enqueue("不值得的", _result(), _verdict(worthy=False, category="neither"))

    assert len(que.list_items(worthy_only=True)) == 1
    assert len(que.list_items(worthy_only=False)) == 2


def test_set_status_rejects_unknown_status(que):
    row_id, _ = que.enqueue("问题", _result(), _verdict())
    with pytest.raises(ValueError):
        que.set_status(row_id, "随便写的")


def test_records_the_retrieval_signals(que):
    result = _result(contexts=[{"score": 0.31}, {"score": 0.55}], web_contexts=[{"url": "u"}])
    row_id, _ = que.enqueue("问题", result, _verdict())

    item = que.get_item(row_id)
    assert item["kb_hits"] == 2
    assert item["kb_top_score"] == pytest.approx(0.55)
    assert item["web_hits"] == 1
