"""把 Qwen 审查的自由文本解析成 (是否通过, 具体意见)。

单独放这里（而不是塞在 reviewer 节点里）是为了能脱离 langchain 单独测试。
"""
import re

from ecom.analytics.utils.logger import setup_logger

logger = setup_logger()

# 首行「通过」的写法
_PASS_RE = re.compile(r"^(判定[:：]?\s*)?(通过|正确|没问题|没有问题|pass|ok|correct)", re.IGNORECASE)
# 「不通过」的写法
_FAIL_RE = re.compile(r"(不通过|错误|不正确|有问题|fail|incorrect|wrong)", re.IGNORECASE)

# 冒号后面**只有一个结论词**（「判定：不通过」而不是「错误：排序方向反了」）。
#
# 两者必须分开：前者说明原因被写在了第二行，后者原因就在冒号后面。
# 所以这里是**整串锚定**（^...$），不能只匹配前缀——`不通过，因为X` 有原因，
# 判成结论词就会把它丢掉。
_VERDICT_ONLY_RE = re.compile(
    r"^\s*(判定|结论|结果|审查|review)?\s*[:：]?\s*"
    r"(不通过|通过|错误|不正确|有问题|没问题|没有问题|正确|fail|incorrect|wrong|pass|ok|correct)"
    r"\s*[。.!！]?\s*$",
    re.IGNORECASE,
)


def parse_review(content: str) -> tuple[bool, str]:
    """容忍 markdown 围栏、列表前缀、「判定：」前缀、中英文混写。

    **无法判定时按「通过」处理（fail-open）**：解析失败就回炉重造会多烧一次
    DeepSeek + Qwen 且可能反复，而审查本身已被 need_review 门槛过滤过。
    """
    text = (content or "").strip()
    text = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "").strip()
    text = text.lstrip("-*># \t").strip()

    if not text:
        logger.warning("审查输出为空，保守放行")
        return True, "审查输出为空，保守放行"

    first_line = text.splitlines()[0].strip()

    if _PASS_RE.match(first_line):
        return True, text

    if _FAIL_RE.search(first_line):
        # 取「错误：xxx」里的 xxx 作为意见；没有冒号就用整句
        parts = re.split(r"[:：]", first_line, maxsplit=1)
        has_colon = len(parts) > 1
        feedback = parts[1].strip() if has_colon else ""

        # 有冒号、但冒号后面只剩一个结论词（或干脆是空的），说明**原因写在第二行**——
        # 提示词就是这么要求的，见 analytics/agents/reviewer.py。
        #
        # 不往下翻的后果不是"少点信息"，而是 feedback 退化成「不通过」三个字，
        # 被喂进 self_corrector 那一轮：它拿到的等于一句废话，改无可改。
        if has_colon and (not feedback or _VERDICT_ONLY_RE.match(feedback)):
            feedback = "\n".join(text.splitlines()[1:]).strip() or feedback

        # 翻完还是空（比如整段就只有「判定：不通过」一行）就退回整段。
        # **不能返回空串**：上面「审查输出为空 → 保守放行」那条语义是靠非空来区分的。
        return False, feedback or text

    # 首行判不出来就扫全文
    if _FAIL_RE.search(text):
        return False, text

    logger.warning(f"审查输出无法判定，按通过处理：{text[:100]}")
    return True, text
