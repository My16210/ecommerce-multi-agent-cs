"""图片理解结果的**结构化契约**，以及它和用户消息之间的拼接规则。

【为什么单独一个模块】
`workers/order.py` 只需要知道"这段文本里有没有图片块、块里的订单号是多少"，
不需要视觉模型客户端。把契约和客户端分开，订单 worker 就不会因为一个图片功能
把 `langchain_openai` / 网络配置拖进来；测试也可以只测纯文本规则。

【图片里的订单号为什么单独标来源】
图片里的订单号是**模型从像素里认出来的**，不是用户在这句话里打出来的。
它和"从上一轮猜来的订单号"属于同一类：可以用来查，不能用来写。写操作
（提交退货申请）必须要求用户在当前消息里手动输入订单号——这是
`workers/order.py` 第三道闸的依据。

【为什么块要带标记，而不是把字段直接拼成一句话】
主链路拿到的仍然是一个字符串（`answer()` 的契约没变），所以"哪些字来自图片"
只能靠文本里的结构区分。`strip_image_block()` 把整块摘掉再交给规则/模型，
就是为了避免图片里的订单号被当成"用户当轮说的号"——那会绕过写操作那道闸。
"""
import re
from dataclasses import dataclass, field

from ecom.orchestrator.utils.ids import ORDER_ID_RE

IMAGE_BLOCK_START = "【图片识别结果】"
IMAGE_BLOCK_END = "【/图片识别结果】"

_BLOCK_RE = re.compile(
    re.escape(IMAGE_BLOCK_START) + r".*?" + re.escape(IMAGE_BLOCK_END),
    re.S,
)


@dataclass(frozen=True)
class ImageIntent:
    """一次图片识别的结果。

    `ok=False` 表示**没识别成**（没 key、网络失败、模型吐了非 JSON、图太大），
    调用方应提示手动输入；`ok=True` 但 `order_id == ""` 是**识别成了，只是图里
    没有订单号**，这是正常情况，不是故障。

    `tokens` 只是记账用：图片理解发生在主链路之前，不进 supervisor 的 token
    汇总，所以单独带在这里，需要时由调用方自己累加。
    """

    ok: bool
    order_id: str = ""
    amount: str = ""
    product: str = ""
    request: str = ""
    confidence: str = ""
    notes: str = ""
    error: str = ""
    raw: str = ""
    tokens: dict = field(default_factory=dict)

    @property
    def has_content(self) -> bool:
        """除了 ok 之外，是否至少有一个可用的字段。"""
        return any((self.order_id, self.amount, self.product, self.request, self.notes))

    @property
    def needs_manual_order_id(self) -> bool:
        """识别成功、但图里没有订单号——提示用户手动补充，不是失败。"""
        return self.ok and not self.order_id


def _one_line(value, limit: int = 200) -> str:
    """把模型输出压成一行：块是逐行解析的，字段里混进换行会破坏结构。"""
    return " ".join(str(value or "").split())[:limit]


def sanitize_image_text(text: str) -> str:
    """去掉模型输出里可能被误当成"用户输入"的订单号和块标记。

    典型场景：视觉模型在 `request` 里复述了「我要退货，订单号 52644」。
    如果原样拼进用户消息，规则层会把这个号标成 `source="question"`，从而
    绕过"图片里的号不能写"那道闸。所以拼进正文前先把号替换掉，真正的号
    留在块里、由 `image_order_id()` 按 `source="image"` 取。
    """
    clean = ORDER_ID_RE.sub("[订单号]", str(text or ""))
    return clean.replace(IMAGE_BLOCK_START, "").replace(IMAGE_BLOCK_END, "")


def format_image_context(intent: ImageIntent) -> str:
    """把结构化意图写成主链路能读的文本块；没有可用字段时返回空串。

    字段用全角冒号，解析函数按行读。每个值都压成一行并截断，既防止模型
    输出里混进换行破坏结构，也防止一段超长文本把 prompt 撑大。
    """
    if not intent.ok or not intent.has_content:
        return ""

    lines = [IMAGE_BLOCK_START]
    for label, value in (
        ("订单号", intent.order_id),
        ("金额", intent.amount),
        ("商品", intent.product),
        ("诉求", intent.request),
        ("置信度", intent.confidence),
        ("备注", intent.notes),
    ):
        value = _one_line(value)
        if value:
            lines.append(f"{label}：{value}")
    lines.append(IMAGE_BLOCK_END)
    return "\n".join(lines)


def augment_question_with_image(text: str, intent: ImageIntent) -> str:
    """把图片意图拼进用户消息；没有可说的内容时返回空串。

    - 用户打了字就用用户的字，**原样保留**——包括用户自己打的订单号。那是
      用户当轮明确给出的号，写操作可以用。
    - 只传图时用图片里的诉求当问题；这一段来自模型，所以要先
      `sanitize_image_text`，把模型可能复述的订单号替换掉。
    - `intent.ok=False` 时返回空串：调用方应当走"识别失败，请手动输入"的分支，
      而不是把一条空问题送进主链路。
    """
    if not intent.ok:
        return ""

    body = (text or "").strip() or sanitize_image_text(intent.request).strip()
    if not body:
        return ""

    context = format_image_context(intent)
    return f"{body}\n\n{context}".strip() if context else body


def _block(question: str) -> str:
    match = _BLOCK_RE.search(question or "")
    return match.group(0) if match else ""


def _field(block: str, label: str) -> str:
    """按行取字段。`^` + MULTILINE 保证只在行首匹配，备注里的同名字样不算。"""
    match = re.search(rf"^{re.escape(label)}[：:]\s*(.*)$", block, re.M)
    return match.group(1).strip() if match else ""


def image_order_id(question: str) -> str:
    """块里的订单号；没有、或形状不合法时返回空串。

    形状校验用全仓库唯一那份 `ORDER_ID_RE`，不在这里另写一条正则。
    """
    candidate = _field(_block(question), "订单号")
    return candidate if ORDER_ID_RE.fullmatch(candidate) else ""


def image_request(question: str) -> str:
    """块里的诉求原文（未清洗）；没有时返回空串。"""
    return _field(_block(question), "诉求")


def strip_image_block(question: str) -> str:
    """摘掉所有图片块，只留下用户在输入框里真正打的字。

    规则提取和 LLM 提取都跑在摘掉之后的文本上——图片里的订单号不能以
    "用户当轮说的"身份进入写操作。
    """
    return _BLOCK_RE.sub("", question or "").strip()
