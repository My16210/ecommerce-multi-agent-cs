from langchain_openai import ChatOpenAI

from ecom.analytics.agents.prompts import build_code_rules
from ecom.analytics.core.state import SalesState
from ecom.analytics.settings import settings
from ecom.analytics.utils.logger import setup_logger
from ecom.analytics.utils.tokens import extract_usage

logger = setup_logger()


def analyst_node(state: SalesState) -> dict:
    logger.info(f"开始分析: {state['question']}")
    # 1.初始化DeepSeek模型
    llm = ChatOpenAI(**settings.get_llm_kwargs("deepseek"))

    # 2. 拼接历史对话
    history = "".join(
        f"{msg['role']}: {msg['content']}\n" for msg in state.get("messages", [])
    )

    # 3.写prompt，告诉模型这是哪份数据、有哪些列、用户问什么、以及代码要遵守的约定
    #   数据集名字和口径都从 state 来——不在这里写死"这是一个销售数据表"，
    #   否则换数据集时这句会留在 prompt 里误导模型。
    code_rules = build_code_rules(state.get("dataset_rules", ""))
    prompt = f"""你是一个数据分析助手。
当前数据集：{state.get('dataset_label') or state.get('dataset_name', '')}
列名有：{state['columns']}
前5行数据：{state['data_sample']}

之前的对话：
{history}

用户的问题是：{state['question']}

请写一段 pandas 代码来回答这个问题。

{code_rules}"""

    # 4. 调模型
    response = llm.invoke(prompt)
    prompt_tokens, completion_tokens = extract_usage(response)
    logger.info("DeepSeek生成代码成功")

    # 5.统计token消耗（按模型分开记，方便算成本）
    return {
        "generated_code": response.content,
        "total_tokens": state.get("total_tokens", 0) + prompt_tokens + completion_tokens,
        "deepseek_prompt_tokens": state.get("deepseek_prompt_tokens", 0) + prompt_tokens,
        "deepseek_completion_tokens": state.get("deepseek_completion_tokens", 0) + completion_tokens,
    }
