"""图片 → 结构化意图：**唯一一处**调用视觉模型的地方。

【为什么图片不直接进对话主链路】
主链路（router / worker / aggregate）看到的应该是一句"用户想干什么"，而不是
一张图。直接把图塞进去，等于让每个 worker 都要会看图、都要为视觉模型的输出
格式负责；而且图片里的订单号会被当成"用户当轮说的号"，绕过写操作那道闸。
所以这里只做一次提取：订单号 / 金额 / 商品 / 诉求 → `ImageIntent`，再由
`utils/image_intent.augment_question_with_image()` 拼成一段带标记的文本。

【失败是返回值，不是异常】
没 key、网络超时、模型吐了非 JSON、图太大、格式不支持——全部返回
`ImageIntent(ok=False, error=...)`。调用方据此提示"请手动输入"，主链路不必
为一个可选功能加 try/except。这是 `understand_image` 唯一的错误契约。

【为什么用 ChatOpenAI 而不是裸 openai SDK】
和 `ecom/orchestrator/utils/llm.py`、`ecom/knowledge/utils/llm.py` 保持同一套
客户端构造方式：按参数缓存、`max_retries=0`（重试权收回到调用方）、超时显式。
视觉模型走的是 OpenAI 兼容接口，`ChatOpenAI` 只是那层薄适配，没有引入新依赖。

【模型名不是常量】
`VISION_MODEL` 从 settings 读，默认 `Qwen/Qwen3-VL-30B-A3B-Instruct`。平台
上下架模型是常态，写死在代码里会让"换一个模型"变成一次改代码 + 发版；放进
.env 之后它只是一个配置项。跑不通时的症状是一次可读的识别失败，不是崩溃。
"""
import base64

from langchain_openai import ChatOpenAI

from ecom.orchestrator.utils.ids import ORDER_ID_RE
from ecom.orchestrator.utils.image_intent import ImageIntent
from ecom.orchestrator.utils.json_utils import loads_or_none
from ecom.orchestrator.utils.tokens import usage_of
from ecom.settings import (
    VISION_API_KEY,
    VISION_BASE_URL,
    VISION_MAX_IMAGE_MB,
    VISION_MODEL,
    VISION_TIMEOUT,
)

# 只收这几种。浏览器/Streamlit 给的 MIME 一般是准的；`image/jpg` 是历史写法，
# 统一归一成 `image/jpeg` 再发给模型。
_ALLOWED_MIME = {"image/png", "image/jpeg", "image/webp"}

_PROMPT = """你在做电商客服的图片信息提取。只输出一个 JSON 对象，不要 Markdown，不要解释。

字段：
- order_id: 订单号，只保留数字；没有就空字符串
- amount: 金额，保留数字和货币符号；没有就空字符串
- product: 商品名称；没有就空字符串
- request: 用户诉求，用一句短语（例如 查物流 / 退货 / 退款 / 改地址 / 咨询政策）；没有就空字符串
- confidence: high / medium / low
- notes: 看不清或不确定的地方；没有就空字符串

规则：不要编造。看不清的字段留空，并在 notes 说明。"""

# 按超时缓存。和另外两个 llm.py 一样：同参数复用同一个连接池。
_clients: dict[int, ChatOpenAI] = {}


def get_vision_llm(timeout: int = VISION_TIMEOUT) -> ChatOpenAI:
    """取视觉模型客户端（按超时缓存）。

    缺 key 时抛的错要能照着改：错误信息里写清去哪配。这个异常会被
    `understand_image` 接住，最终变成界面上的一句"请手动输入"。
    """
    if timeout not in _clients:
        if not VISION_API_KEY:
            raise RuntimeError(
                "缺少 VISION_API_KEY（或 SILICONFLOW_API_KEY）。"
                "请在项目根目录的 .env 中配置，或直接填 SILICONFLOW_API_KEY。"
            )
        _clients[timeout] = ChatOpenAI(
            model=VISION_MODEL,
            api_key=VISION_API_KEY,
            base_url=VISION_BASE_URL,
            temperature=0,      # 提取任务，要稳定，不要"有创意"
            timeout=timeout,
            max_retries=0,      # 见模块开头：重试只能有一层，那一层是我们
        )
    return _clients[timeout]


def _message_text(response) -> str:
    """从 LangChain 响应里取文本。

    `content` 可能是 str，也可能是 content blocks 列表（有些 provider 在多模态
    调用里会这么回）。两种都收，取不到就返回空串——解析层会把它判成失败。
    """
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content or "")


def _clean(value, limit: int = 200) -> str:
    return " ".join(str(value or "").split())[:limit]


def _to_intent(parsed: dict, raw: str, tokens: dict) -> ImageIntent:
    """把模型给的 JSON 归一成 `ImageIntent`。

    订单号**再过一次形状校验**：模型可能把运单号、金额、客户号写进 order_id。
    校验不过就留空——空号会让主链路问用户要，错号会让它去查一笔不存在的订单。
    两种都比"相信模型"好。
    """
    raw_id = _clean(parsed.get("order_id"), 40)
    match = ORDER_ID_RE.search(raw_id)
    order_id = match.group(1) if match else ""

    confidence = _clean(parsed.get("confidence"), 20).lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    return ImageIntent(
        ok=True,
        order_id=order_id,
        amount=_clean(parsed.get("amount"), 40),
        product=_clean(parsed.get("product"), 200),
        request=_clean(parsed.get("request"), 200),
        confidence=confidence,
        notes=_clean(parsed.get("notes"), 300),
        raw=raw[:2000],
        tokens=tokens,
    )


def understand_image(
    data: bytes,
    mime_type: str = "image/png",
    *,
    llm=None,
) -> ImageIntent:
    """把一张图提取成结构化意图。**任何失败都返回 ok=False，不抛异常。**

    `llm` 只给测试注入假模型用；生产路径不传，走 `get_vision_llm()`。
    """
    if not data:
        return ImageIntent(ok=False, error="图片是空的")

    limit = int(VISION_MAX_IMAGE_MB * 1024 * 1024)
    if len(data) > limit:
        return ImageIntent(ok=False, error=f"图片超过 {VISION_MAX_IMAGE_MB:g}MB 上限")

    mime = (mime_type or "image/png").strip().lower()
    if mime == "image/jpg":
        mime = "image/jpeg"
    if mime not in _ALLOWED_MIME:
        return ImageIntent(ok=False, error=f"不支持的图片类型：{mime_type}")

    try:
        client = llm or get_vision_llm()
        data_url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        response = client.invoke(
            [
                ("system", "你是电商客服的图片信息提取器，只输出 JSON。"),
                (
                    "human",
                    [
                        {"type": "text", "text": _PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                ),
            ]
        )
    except Exception as exc:  # noqa: BLE001 —— 这是可选功能，失败必须降级
        return ImageIntent(ok=False, error=f"{type(exc).__name__}: {exc}")

    text = _message_text(response)
    parsed = loads_or_none(text)
    if not isinstance(parsed, dict):
        return ImageIntent(ok=False, error="视觉模型没有返回可解析的 JSON", raw=text[:2000])

    return _to_intent(parsed, text, usage_of(response))
