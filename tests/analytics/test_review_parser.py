

from ecom.analytics.utils.review_parser import parse_review


def test_plain_pass():
    assert parse_review("正确")[0] is True
    assert parse_review("判定：通过")[0] is True
    assert parse_review("通过")[0] is True


def test_plain_fail_with_colon_extracts_feedback():
    passed, feedback = parse_review("错误：排序方向反了")
    assert passed is False
    assert feedback == "排序方向反了"


def test_verdict_style_fail():
    passed, feedback = parse_review("判定：不通过\n销售额没有剔除退货，口径和问题不符")
    assert passed is False
    assert "未剔除" in feedback or "口径" in feedback


def test_fail_without_colon_keeps_whole_text():
    passed, feedback = parse_review("这段代码有问题")
    assert passed is False
    assert feedback


def test_markdown_fence_and_list_prefix_are_stripped():
    assert parse_review("```\n判定：通过\n```")[0] is True
    assert parse_review("- 正确")[0] is True


def test_fail_anywhere_in_text_is_caught():
    passed, _ = parse_review("我看了一下\n这段代码计算逻辑错误")
    assert passed is False


def test_empty_output_fails_open():
    passed, _ = parse_review("")
    assert passed is True
    assert parse_review(None)[0] is True


def test_unparseable_output_fails_open():
    # 无法判定 → 保守按通过处理，避免无谓地烧一次 DeepSeek + Qwen
    assert parse_review("嗯，我再想想")[0] is True


def test_negation_not_mistaken_for_pass():
    # 「不通过」里含「通过」，不能被误判成通过
    assert parse_review("不通过")[0] is False
