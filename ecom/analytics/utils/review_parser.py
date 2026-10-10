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
        feedback = parts[1].strip() if len(parts) > 1 and parts[1].strip() else text
        return False, feedback

    # 首行判不出来就扫全文
    if _FAIL_RE.search(text):
        return False, text

    logger.warning(f"审查输出无法判定，按通过处理：{text[:100]}")
    return True, text
