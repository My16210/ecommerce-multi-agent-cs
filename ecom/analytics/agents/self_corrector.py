from langchain_openai import ChatOpenAI

from ecom.analytics.agents.prompts import build_code_rules
from ecom.analytics.core.state import SalesState
from ecom.analytics.settings import settings
from ecom.analytics.utils.logger import setup_logger
from ecom.analytics.utils.tokens import extract_usage

logger = setup_logger()


def self_corrector_node(state: SalesState) -> dict:
    """纠错节点，两种来源共用：

    - exec_error 有值：上次代码执行报错，重写并让 retry_count +1（受 MAX_EXEC_RETRY 限制）
    - 否则：审查不通过，按审查意见改逻辑。这条不计入 retry_count，否则两个循环会互相污染计数。
    """
    exec_error = state.get("exec_error")

    if exec_error:
        source = "exec_error"
        retry_count = state.get("retry_count", 0) + 1
        logger.info(f"第{retry_count}次自动纠错（执行报错）")
        tail = f"""下面的代码执行时报错了，请修正：
错误信息：{exec_error}"""
    else:
        source = "review"
        retry_count = state.get("retry_count", 0)
        feedback = state.get("review_feedback", "")
        logger.info(f"审查不通过，按审查意见纠正：{feedback}")
        tail = f"""这段代码能跑通，但审查认为结果不可信，请按审查意见修改：
审查意见：{feedback}

注意：这不是报错，而是口径、过滤条件、排序方向或列名理解上的偏差。"""

    # 1. 初始化DeepSeek
    llm = ChatOpenAI(**settings.get_llm_kwargs("deepseek"))

    # 2. 共用头 + 分支尾，两处规则保持一致
    #    口径走和 analyst 同一个 build_code_rules：改代码的模型和写代码的模型
    #    拿到的是**同一份**口径，不会出现"按 A 套写、按 B 套改"。
    code_rules = build_code_rules(state.get("dataset_rules", ""))
    prompt = f"""你之前写的 pandas 代码有问题。
用户问题：{state['question']}
数据列名：{state['columns']}
前5行数据：{state['data_sample']}

你之前写的代码：
{state['generated_code']}

{tail}

{code_rules}"""

    # 3. 调模型
    response = llm.invoke(prompt)
    prompt_tokens, completion_tokens = extract_usage(response)

    # 4. 返回新代码 + 计数 + 清掉旧错误（交给 execute 重新判定）
    return {
        "generated_code": response.content,
        "fix_source": source,
        "retry_count": retry_count,
        "exec_error": None,
        "total_tokens": state.get("total_tokens", 0) + prompt_tokens + completion_tokens,
        "deepseek_prompt_tokens": state.get("deepseek_prompt_tokens", 0) + prompt_tokens,
        "deepseek_completion_tokens": state.get("deepseek_completion_tokens", 0) + completion_tokens,
    }
