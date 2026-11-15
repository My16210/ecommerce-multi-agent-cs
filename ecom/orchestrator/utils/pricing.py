"""模型单价与成本换算。

**必须区分三种「零」**：
  1. 走了规则（source="rule"）—— 真的没调模型，成本是 0；
  2. 调了模型但 provider 没返回 usage —— 成本未知，不是 0；
  3. 模型单价没填 —— 成本算不出来，也不是 0。

把后两种算成 0，成本表就会静静地少算，而且看不出来——和 `utils/tokens.py`
里那条注释是同一个问题。

所以这里的原则是：**算不出来就返回 None，绝不返回 0 顶替。**

注：本模块目前放在第一个项目的仓库里，因为它要接那个项目的 trace。
将来抽成独立工具时整块搬走即可。
"""


# 改了单价必须升这个版本号，否则旧实验和新实验的钱不可比。
PRICE_VERSION = "v1-2026-11"

# 单位：美元 / 百万 token
# key 是 resolve_model() 归一后的**档位名**，不是 provider 报的完整模型名：
# qwen-plus / qwen-turbo 都会落到 "qwen"，deepseek-chat 落到 "deepseek"。
# 同一档位的不同版本共用一行，改价只有一个地方。
PRICES = {
    "deepseek": {"input": 0.27, "output": 1.10},
    # 百炼官方原价（2026-10-10 文档，输入 ≤128k 档）：输入 0.8 元/M、输出 2 元/M。
    # 按 7.2 的汇率折成美元；**7.2 是假设值**，实际结算汇率不同要同步改这里。
    # **思考模式**：输出价是 8 元/M（4 倍，折 1.1111）。当前 call 记录没有单独
    # 区分 reasoning token，也没确认 qwen-plus 在本账号是否默认开思考——如果
    # 账单显示是思考模式，必须把 output 改成 1.1111，否则成本会低估 4 倍。
    "qwen": {"input": 0.1111, "output": 0.2778},
}


def resolve_model(name: str | None) -> str | None:
    """把 provider 报的具体模型名归一到**单价档位**。

    调用记录里的 `model` 是真实标识（如 `qwen-plus`），而 PRICES 按档位
    （`qwen` / `deepseek`）维护：同一家的不同版本共用一个档位时，只该改一处。
    不做归一的话，`cost_of("qwen-plus", ...)` 会查不到 key，即使价格填了
    也永远返回 None——而且是静默的。

    不认识的模型原样（小写）返回：表里没有就是没有，`cost_of` 会返回 None。
    """
    if not name:
        return None
    value = str(name).strip().lower()
    if not value:
        return None
    if value.startswith("qwen"):
        return "qwen"
    if value.startswith("deepseek"):
        return "deepseek"
    return value


def price_of(model: str) -> dict | None:
    """取某模型的单价表；没填或未知模型返回 None。先归一再查。"""
    return PRICES.get(resolve_model(model))


def unpriced_models() -> list[str]:
    """返回还没填单价的模型。

    启动时打印一次。不打印的话，未填单价的模型会静默漏算，
    而报表看起来一切正常。
    """
    return [name for name, price in PRICES.items() if price is None]


def cost_of(model: str, tokens_in: int, tokens_out: int) -> float | None:
    """算一次调用的成本（美元）。

    返回 None 的两种情况，含义不同但都必须如实表达：
      - 模型没填单价（unpriced）
      - 模型不在表里（unknown）
    调用方拿到 None 时，只能用「未知」参与汇总，不许当 0 累加。
    """
    price = price_of(model)
    if price is None:
        return None
    return (
        int(tokens_in or 0) / 1_000_000 * price["input"]
        + int(tokens_out or 0) / 1_000_000 * price["output"]
    )


def fmt_cost(cost: float | None) -> str:
    """成本的人类可读格式。None 显式写成「未知」，不留空。"""
    if cost is None:
        return "未知"
    return f"${cost:.6f}"


def cost_of_call(call: dict) -> float | None:
    """一条 call 的成本。三种「零」必须分开：
      1. source="rule" 且 tokens 全 0 → 0.0（真的没花钱）
      2. model 有价 → 实际成本
      3. 其余（model=None / source=unknown / 没拿到 usage）→ None（未知）
    第 1 类绝不能被算成 None，否则每次订单问答都带一条假「未知」，
    真正的黑洞就被稀释了。
    """
    if (
        call.get("source") == "rule"
        and not call.get("tokens_in")
        and not call.get("tokens_out")
    ):
        return 0.0
    return cost_of(call.get("model"), call.get("tokens_in", 0), call.get("tokens_out", 0))


def summarize_cost(calls: list[dict]) -> dict:
    """已知部分求和，未知条数单列。unknown 绝不当 0 混进 known。"""
    known = 0.0
    unknown = 0
    for call in calls:
        value = cost_of_call(call)
        if value is None:
            unknown += 1
        else:
            known += value
    return {"known_usd": known, "unknown_calls": unknown, "complete": unknown == 0}
