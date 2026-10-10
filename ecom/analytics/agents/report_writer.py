from langchain_openai import ChatOpenAI

from ecom.analytics.core.state import SalesState
from ecom.analytics.settings import settings
from ecom.analytics.utils.logger import setup_logger
from ecom.analytics.utils.tokens import extract_usage

logger = setup_logger()


def report_writer_node(state: SalesState) -> dict:
    exec_error = state.get("exec_error")
    result = (state.get("exec_result") or "").strip()

    # 1. 执行失败 / 没有结果：给一份诚实的失败说明，不调 LLM 编结论
    if exec_error or not result or result == "没有返回结果":
        reason = exec_error or "代码没有产出结果"
        report = (
            f"分析未能完成：{reason}。"
            f"已自动重试 {state.get('retry_count', 0)} 次。"
            "可以换个问法，或确认问题里的维度和数据列是否对得上。"
        )
        logger.warning(f"生成失败报告：{reason}")
        return {
            "final_report": report,
            "chart_generated": False,
            "chart_path": "",
            "messages": state.get("messages", []) + [
                {"role": "用户", "content": state["question"]},
                {"role": "助手", "content": report},
            ],
        }

    # 2. 成功：让 DeepSeek 把结果写成一段人话结论
    llm = ChatOpenAI(**settings.get_llm_kwargs("deepseek"))
    prompt = f"""用户问的是：{state['question']}
数据分析结果是：{result}
本次分析共消耗{state.get('total_tokens', 0)}个token。

请把这个结果写成一段简洁的中文结论，不要超过3句话，最后提一下本次token消耗。
"""
    response = llm.invoke(prompt)
    prompt_tokens, completion_tokens = extract_usage(response)

    final_report = response.content

    # 3. 审查没通过且额度已用尽：在报告里明确提示风险，不要假装没事
    if state.get("need_review") and not state.get("review_passed"):
        final_report += f"\n\n（提示：本次结果未通过 Qwen 审查，审查意见：{state.get('review_feedback', '')}）"

    # 4. 图只有在本次真的画出来时才展示
    chart_generated = bool(state.get("chart_generated"))
    return {
        "final_report": final_report,
        "chart_generated": chart_generated,
        "chart_path": state.get("chart_path", "") if chart_generated else "",
        "total_tokens": state.get("total_tokens", 0) + prompt_tokens + completion_tokens,
        "deepseek_prompt_tokens": state.get("deepseek_prompt_tokens", 0) + prompt_tokens,
        "deepseek_completion_tokens": state.get("deepseek_completion_tokens", 0) + completion_tokens,
        "messages": state.get("messages", []) + [
            {"role": "用户", "content": state["question"]},
            {"role": "助手", "content": final_report},
        ],
    }
