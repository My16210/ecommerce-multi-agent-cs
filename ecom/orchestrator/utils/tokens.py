"""从 LangChain 的响应里挖出 token 用量。

**必须容错**：`usage_metadata` 不是每个 provider、每次调用都给。挖不到就当 0，
但**不要**因此认为"这次不花钱"——它更可能是"这次没记上"。所以调用方汇总时
要把"有没有 usage"和"用了多少"分开看，否则成本表会静静地少算。
"""


def usage_of(response) -> dict:
    """返回 {"in": int, "out": int}；缺字段按 0 计"""
    usage = getattr(response, "usage_metadata", None) or {}
    return {
        "in": int(usage.get("input_tokens", 0) or 0),
        "out": int(usage.get("output_tokens", 0) or 0),
    }


def add(a: dict, b: dict) -> dict:
    """两份用量的合并。路由会重试，重试那次的 token 也得算进去。"""
    return {
        "in": int(a.get("in", 0)) + int(b.get("in", 0)),
        "out": int(a.get("out", 0)) + int(b.get("out", 0)),
    }


EMPTY = {"in": 0, "out": 0}
