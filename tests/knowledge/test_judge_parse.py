"""
提问价值判定的输出解析测试。

只测 `_parse` 这个纯函数——容错逻辑全部集中在它身上，而且它不碰网络、
不碰 LLM，可以放心地喂各种畸形输入。真正的 judge() 要调千问，不在这里测。

解析的立场是"宁可漏收，不能污染队列"：任何解析不出来的输出，
一律按「不值得」处理。
"""
from ecom.knowledge.review.judge import _parse


def test_plain_json():
    verdict = _parse(
        '{"worthy": true, "category": "kb_gap", "score": 0.8, "reason": "库里缺这块"}',
        "q",
    )
    assert verdict["worthy"] is True
    assert verdict["category"] == "kb_gap"
    assert verdict["score"] == 0.8
    assert verdict["reason"] == "库里缺这块"


def test_markdown_fence_is_stripped():
    """prompt 里说了别用围栏，但模型经常还是加"""
    raw = '```json\n{"worthy": true, "category": "reusable_faq", "score": 0.5}\n```'
    assert _parse(raw, "q")["category"] == "reusable_faq"


def test_surrounding_prose_is_ignored():
    raw = '好的，我的判断如下：\n{"worthy": true, "category": "kb_gap", "score": 0.7}\n以上。'
    assert _parse(raw, "q")["category"] == "kb_gap"


def test_string_false_is_not_truthy():
    """真实踩过的坑：模型把布尔写成字符串 "false"，而 bool("false") 是 True"""
    verdict = _parse('{"worthy": "false", "category": "neither", "score": 0.1}', "q")
    assert verdict["worthy"] is False


def test_string_true_is_recognized():
    verdict = _parse('{"worthy": "true", "category": "kb_gap", "score": 0.9}', "q")
    assert verdict["worthy"] is True


def test_invalid_json_falls_back_to_not_worthy():
    verdict = _parse("模型今天不想输出 JSON", "q")
    assert verdict["worthy"] is False
    assert verdict["category"] == "neither"
    assert verdict["score"] == 0.0


def test_unknown_category_falls_back_to_neither():
    """category 和 worthy 冲突时以 category 为准"""
    verdict = _parse('{"worthy": true, "category": "something_else", "score": 0.9}', "q")
    assert verdict["category"] == "neither"
    assert verdict["worthy"] is False


def test_score_is_clamped_into_range():
    assert _parse('{"category": "kb_gap", "score": 5}', "q")["score"] == 1.0
    assert _parse('{"category": "kb_gap", "score": -3}', "q")["score"] == 0.0


def test_non_numeric_score_falls_back_to_zero():
    assert _parse('{"category": "kb_gap", "score": "很高"}', "q")["score"] == 0.0


def test_none_and_empty_input_are_handled():
    assert _parse(None, "q")["worthy"] is False
    assert _parse("", "q")["worthy"] is False
