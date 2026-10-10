from langgraph.graph import END, StateGraph

from ecom.analytics.agents.analyst import analyst_node
from ecom.analytics.agents.code_executor import code_executor_node
from ecom.analytics.agents.data_loader import data_loader_node
from ecom.analytics.agents.question_classifier import question_classifier_node
from ecom.analytics.agents.report_writer import report_writer_node
from ecom.analytics.agents.reviewer import reviewer_node
from ecom.analytics.agents.self_corrector import self_corrector_node
from ecom.analytics.core.state import SalesState

# 执行报错最多重写几次代码
MAX_EXEC_RETRY = 3
# reviewer 最多跑几轮（首审 + 复审），即审查驱动的纠正最多 1 次
MAX_REVIEW_ROUNDS = 2


def after_execute(state: SalesState) -> str:
    """执行之后的走向：报错就重写，成功则按需审查，无需审查就直接出报告。"""
    if state.get("exec_error"):
        if state.get("retry_count", 0) < MAX_EXEC_RETRY:
            return "self_correct"
        # 重试额度耗尽仍失败：交给报告节点给出「诚实失败」，不再无谓消耗
        return "report"

    # 成功。简单问题不审查，省下一次 Qwen 调用
    if state.get("need_review") and state.get("review_round", 0) < MAX_REVIEW_ROUNDS:
        return "reviewer"
    return "report"


def after_review(state: SalesState) -> str:
    """审查之后的走向：通过就出报告，不通过就改代码重跑，额度用尽则带风险出报告。"""
    if state.get("review_passed"):
        return "report"
    if state.get("review_round", 0) < MAX_REVIEW_ROUNDS:
        return "self_correct"
    return "report"


def build_graph():
    workflow = StateGraph(SalesState)

    workflow.add_node("load_data", data_loader_node)
    workflow.add_node("classify", question_classifier_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("execute", code_executor_node)
    workflow.add_node("self_correct", self_corrector_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("report", report_writer_node)

    workflow.set_entry_point("load_data")
    workflow.add_edge("load_data", "classify")
    workflow.add_edge("classify", "analyst")
    workflow.add_edge("analyst", "execute")

    workflow.add_conditional_edges("execute", after_execute, {
        "self_correct": "self_correct",
        "reviewer": "reviewer",
        "report": "report",
    })
    # 两个纠错来源（执行报错 / 审查不通过）都走这条边回到 execute，
    # 回到 execute 后再由 after_execute 依据当前状态重新决策，因此不需要记住来源。
    workflow.add_edge("self_correct", "execute")

    workflow.add_conditional_edges("reviewer", after_review, {
        "self_correct": "self_correct",
        "report": "report",
    })
    workflow.add_edge("report", END)

    return workflow.compile()
