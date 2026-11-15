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


def usage_given(response) -> bool:
    """provider 到底有没有返回 usage。

    和 `usage_of` 的 0 必须分开：没返回 usage 时它也是 0，但那是"没记上"，
    不是"没花钱"。成本表要能区分这两种零。
    """
    return bool(getattr(response, "usage_metadata", None))


def model_name_of(llm) -> str | None:
    """从模型对象上取**实际使用的**模型标识；取不到返回 None。

    **不读配置**：配置里有默认值，而这里要记录的是这一次真正拿到的客户端。
    假模型（测试替身）没有这个属性时返回 None——不要编一个名字，
    编了会让成本表看起来有数，实际全是假的。
    """
    for attr in ("model_name", "model"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def add(a: dict, b: dict) -> dict:
    """两份用量的合并。路由会重试，重试那次的 token 也得算进去。"""
    return {
        "in": int(a.get("in", 0)) + int(b.get("in", 0)),
        "out": int(a.get("out", 0)) + int(b.get("out", 0)),
    }


EMPTY = {"in": 0, "out": 0}
