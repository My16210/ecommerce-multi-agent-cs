

from ecom.analytics.agents.question_classifier import question_classifier_node


def needs_review(question: str) -> bool:
    return question_classifier_node({"question": question})["need_review"]


# 复杂问题 → 需要 Qwen 审查
def test_ranking_questions_need_review():
    assert needs_review("哪个地区卖得最好？") is True
    assert needs_review("销量最高的商品是什么？") is True
    assert needs_review("哪个地区的退货率最高？") is True
    assert needs_review("iPhone15在哪个地区卖得最多？") is True


def test_trend_and_comparison_need_review():
    assert needs_review("各月销售额趋势怎么样？") is True
    assert needs_review("华东和华北的销量对比") is True
    assert needs_review("各地区销售占比是多少？") is True


def test_top_is_case_insensitive():
    assert needs_review("top5的商品") is True
    assert needs_review("Top 5 商品") is True
    assert needs_review("TOP10") is True


# 简单问题 → 跳过审查，省一次 Qwen 调用
def test_simple_questions_skip_review():
    assert needs_review("总销售额是多少？") is False
    assert needs_review("一共卖了多少件？") is False


def test_多少_alone_does_not_trigger():
    # 「多少」不带「最」不应被当成排名问题
    assert needs_review("平均金额是多少") is False
