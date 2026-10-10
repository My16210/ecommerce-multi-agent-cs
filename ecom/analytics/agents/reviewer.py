from langchain_openai import ChatOpenAI

from ecom.analytics.core.state import SalesState
from ecom.analytics.settings import settings
from ecom.analytics.utils.logger import setup_logger
from ecom.analytics.utils.review_parser import parse_review
from ecom.analytics.utils.tokens import extract_usage

logger = setup_logger()


def reviewer_node(state: SalesState) -> dict:
    logger.info("Qwen开始审查代码")
    # 1.初始化Qwen模型
    llm = ChatOpenAI(**settings.get_llm_kwargs("qwen"))

    # 2. 写prompt：首行要求机器可读，便于解析成布尔判定
    prompt = f"""你是一个代码审查员。
用户的问题是：{state['question']}
数据列名有：{state['columns']}
退货列含义：是否退货 = 1 表示该笔订单退货；默认口径是统计全部订单，不剔除退货。

DeepSeek写的pandas代码是：
{state['generated_code']}

执行结果是：
{state['exec_result']}

请检查：
1. 列名有没有写错
2. 计算逻辑和口径对不对（尤其是是否该剔除退货、排序方向、过滤条件）
3. 结果是否真的回答了用户的问题

第一行只写「判定：通过」或「判定：不通过」。
如果不通过，从第二行开始用一句话说明具体问题。
"""

    # 3. 调模型
    response = llm.invoke(prompt)
    passed, feedback = parse_review(response.content)
    prompt_tokens, completion_tokens = extract_usage(response)
    logger.info(f"Qwen审查结果：{'通过' if passed else '不通过'} | {feedback[:80]}")

    # 4. 返回结构化判定 + 计数 + 分模型token
    return {
        "review_result": response.content,
        "review_comment": response.content,
        "review_passed": passed,
        "review_feedback": feedback,
        "review_round": state.get("review_round", 0) + 1,
        "total_tokens": state.get("total_tokens", 0) + prompt_tokens + completion_tokens,
        "qwen_prompt_tokens": state.get("qwen_prompt_tokens", 0) + prompt_tokens,
        "qwen_completion_tokens": state.get("qwen_completion_tokens", 0) + completion_tokens,
    }
