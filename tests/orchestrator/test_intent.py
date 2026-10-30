"""「用户是不是在要求人工客服」的规则测试。

这条判断的难点不在召回，在**误报**：中文里「人工」本身不表示任何诉求
（「退货需要人工审核吗」是政策咨询）。误报的表现是"问政策，系统说已为你
转接人工"——比答不上来更难解释。所以这里正反两面都要钉住。
"""
import pytest

from ecom.orchestrator.utils.intent import wants_human


@pytest.mark.parametrize(
    "question",
    [
        "转人工",
        "帮我转人工",
        "我要转人工客服",
        "人工客服",
        "找人工",
        "叫真人客服",
        "请人工服务",
        "人工",
        "转 人 工",
        "在吗，人工客服呢？",
    ],
)
def test_explicit_requests_are_detected(question):
    assert wants_human(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "退货需要人工审核吗",
        "退换货政策是什么",
        "订单 52644 到哪了",
        "人工成本怎么算",
        "退款需要人工处理吗",
        "",
        "   ",
    ],
)
def test_policy_and_ordinary_questions_are_not_handoffs(question):
    assert wants_human(question) is False
