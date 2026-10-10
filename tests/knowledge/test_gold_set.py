"""
金标准工具的两条命门。

这个脚本的价值不在它算了什么，而在它**没有**算什么：没有把模型答案当标准答案、
没有把没标过的条目塞进分母。这两条一旦失守，工具照常出分，而且出得更好看——
是那种最不该有的失败方式。所以这里只钉这两条，外加一条"抽样别换题"。

不联网、不碰向量库：只测纯函数。
"""
from ecom.knowledge.cli.gold_set import confirmed_cases, stratified_sample


def _case(case_id: str, kind: str) -> dict:
    return {"id": case_id, "question": f"问题 {case_id}", "type": kind}


def _cases(kind: str, count: int, prefix: str) -> list:
    return [_case(f"{prefix}-{i}", kind) for i in range(count)]


# --------------------------------------------------------------------------
# 分层抽样
# --------------------------------------------------------------------------


def test_sample_covers_both_types():
    """两个题型都得抽到——整体随机抽 10 条时，很容易整批倒向一边"""
    cases = _cases("term", 50, "t") + _cases("paraphrase", 50, "p")

    sample = stratified_sample(cases, 10)

    kinds = {case["type"] for case in sample}
    assert kinds == {"term", "paraphrase"}
    assert len(sample) == 10


def test_sample_ignores_input_order():
    """评测集重排一次，压测的那几十道题不能跟着换"""
    cases = _cases("term", 30, "t") + _cases("paraphrase", 30, "p")

    forward = stratified_sample(cases, 12)
    backward = stratified_sample(list(reversed(cases)), 12)

    assert [c["id"] for c in forward] == [c["id"] for c in backward]


def test_sample_is_repeatable():
    cases = _cases("term", 40, "t") + _cases("paraphrase", 40, "p")

    assert ([c["id"] for c in stratified_sample(cases, 15)]
            == [c["id"] for c in stratified_sample(cases, 15)])


def test_short_bucket_still_fills_the_quota():
    """某个题型不够数时，用另一类的题补满——不能悄悄少抽几条"""
    cases = _cases("term", 3, "t") + _cases("paraphrase", 50, "p")

    sample = stratified_sample(cases, 10)

    assert len(sample) == 10
    assert sum(1 for c in sample if c["type"] == "term") == 3


def test_asking_for_more_than_exists_returns_everything():
    cases = _cases("term", 4, "t") + _cases("paraphrase", 4, "p")

    sample = stratified_sample(cases, 50)

    assert len(sample) == 8


# --------------------------------------------------------------------------
# 未标注的条目不进报告
# --------------------------------------------------------------------------


def test_only_confirmed_entries_count():
    """pending 和 discarded 必须被挡住——它们进分母就是让模型自证"""
    records = [
        {"id": "a", "status": "confirmed", "expected_source": "x.md"},
        {"id": "b", "status": "pending"},
        {"id": "c", "status": "pending"},
        {"id": "d", "status": "discarded"},
    ]

    confirmed, pending, discarded = confirmed_cases(records)

    assert [r["id"] for r in confirmed] == ["a"]
    assert [r["id"] for r in pending] == ["b", "c"]
    assert [r["id"] for r in discarded] == ["d"]


def test_none_of_above_is_confirmed_but_has_no_answer():
    """「前 N 条都不对」算已确认，但没有期望段——它得进分母、记为未命中"""
    records = [{"id": "a", "status": "confirmed", "expected_source": None}]

    confirmed, _, _ = confirmed_cases(records)

    labeled = [r for r in confirmed if r.get("expected_source")]
    assert confirmed == records
    assert labeled == [], "「都不对」不该被当成一条可命中的题"


def test_empty_gold_set_is_not_an_error():
    confirmed, pending, discarded = confirmed_cases([])

    assert (confirmed, pending, discarded) == ([], [], [])
