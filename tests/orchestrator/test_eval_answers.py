"""答案评测器：判定、JSON 解析、运行结果落盘。"""
import json

from ecom.orchestrator.cli.eval_answers import (
    cost_text,
    judge,
    parse_result,
    trace_diagnostics,
    write_run,
)


def test_judge_requires_all_must_include_any_group_and_no_forbidden():
    item = {
        "must_include": ["7", "签收"],
        "must_include_any": ["鲜活易腐", "生鲜"],
        "must_not_include": ["不支持退货"],
    }
    assert judge("签收后 7 天内，生鲜不适用", item)
    assert not judge("签收后 14 天内，生鲜不适用", item)      # 缺 must_include
    assert not judge("签收后 7 天内，定作商品不适用", item)   # any 组没中
    assert not judge("签收后 7 天，生鲜，不支持退货", item)   # 命中禁词


def test_judge_is_case_insensitive():
    item = {"must_include": [], "must_include_any": ["Perfect Fitness"], "must_not_include": []}
    assert judge("perfect fitness perfect rip deck", item)


def test_parse_result_takes_the_last_json_object_and_ignores_log_lines():
    stdout = """INFO something happened
{"answer": "旧答案", "tokens": {"in": 1, "out": 2}}
not json
{"answer": "最终答案", "tokens": {"in": 3, "out": 4}, "cost": {"known_usd": 0.1}}
"""
    result = parse_result(stdout)

    assert result["answer"] == "最终答案"
    assert result["tokens"] == {"in": 3, "out": 4}


def test_parse_result_returns_none_when_nothing_parses():
    assert parse_result("INFO only\nnot json") is None


def test_write_run_persists_every_answer(tmp_path):
    rows = [
        {"id": "P01", "question": "q", "answer": "完整答案一", "passed": True},
        {"id": "P02", "question": "q", "answer": "完整答案二", "passed": False},
    ]
    path = write_run(rows, tmp_path / "nested" / "run.jsonl")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == rows


def test_trace_diagnostics_picks_the_matching_request(tmp_path):
    log = tmp_path / "requests.jsonl"
    log.write_text(
        '{"request_id": "old", "workers": {"analytics": {"retry_count": 2}}}\n'
        "not json\n"
        '{"request_id": "new", "workers": {"analytics": {"retry_count": 1, "review_round": 2}}}\n',
        encoding="utf-8",
    )

    assert trace_diagnostics("new", log) == {"retry_count": 1, "review_round": 2}
    assert trace_diagnostics("missing", log) == {}
    assert trace_diagnostics("", log) == {}


def test_cost_text_never_hides_unknown_calls():
    assert cost_text({"known_usd": 0.000303, "unknown_calls": 0}) == "$0.000303"
    assert cost_text({"known_usd": 0.000303, "unknown_calls": 2}) == "$0.000303 + 2 条未知"
    assert cost_text(None) == "未知"
