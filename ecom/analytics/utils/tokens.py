"""LLM token 统计工具。

统一在这里取 usage，避免各节点直接 `response.usage_metadata.get(...)`：
部分 OpenAI 兼容端点不返回 usage，`usage_metadata` 会是 None，直接 .get 会崩掉整个图。
"""


def extract_usage(response) -> tuple[int, int]:
    """从 ChatOpenAI 的响应里取 (prompt_tokens, completion_tokens)，取不到返回 (0, 0)。

    【为什么先读 input_tokens / output_tokens】

    这两个才是 LangChain 统一的 `usage_metadata` 里的键名。
    `prompt_tokens` / `completion_tokens` 是 **OpenAI 原始响应**里的名字，
    只出现在 `response_metadata["token_usage"]` 里。

    这个函数原来只读后者，所以 `usage_metadata` 明明有值、取出来却永远是 (0, 0)。
    症状是**完全静默**的：图照跑、报告照出、图照画，只有 eval.py 的成本列
    永远显示 $0.0000，以及报告结尾那句"本次分析共消耗 0 个 token"。

    全项目其余四处读的都是 input_tokens（orchestrator/utils/tokens.py、
    knowledge 的 answer_eval / judge / generator），只有这里读错了——
    这正是"同一个概念在两处各实现一遍"会付的代价。

    两级取值：先 LangChain 的标准键，再退回 OpenAI 原始键（少数兼容端点
    只填后者）。两个都没有就当 0，但要清楚那是"这次没记上"，不是"这次没花钱"。
    """
    usage = getattr(response, "usage_metadata", None) or {}
    if usage:
        prompt = usage.get("input_tokens")
        completion = usage.get("output_tokens")
        if prompt is not None or completion is not None:
            return int(prompt or 0), int(completion or 0)

    raw = (getattr(response, "response_metadata", None) or {}).get("token_usage") or {}
    return int(raw.get("prompt_tokens") or 0), int(raw.get("completion_tokens") or 0)
