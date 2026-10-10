"""LLM token 统计工具。

统一在这里取 usage，避免各节点直接 `response.usage_metadata.get(...)`：
部分 OpenAI 兼容端点不返回 usage，`usage_metadata` 会是 None，直接 .get 会崩掉整个图。
"""


def extract_usage(response) -> tuple[int, int]:
    """从 ChatOpenAI 的响应里取 (prompt_tokens, completion_tokens)，取不到返回 (0, 0)。"""
    usage = getattr(response, "usage_metadata", None) or {}
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
