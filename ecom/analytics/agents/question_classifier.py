import re

from ecom.analytics.core.state import SalesState

# 判定「需要第二模型审查」的问题特征。
# 用正则而不是关键词列表，是为了大小写不敏感、并覆盖「最高/最少」这类中文比较级
# —— 原来的 ["Top","排名",...] 里 Top 是大写，且「哪个地区退货率最高」根本命中不了。
_COMPLEX_PATTERNS = [
    r"最(高|低|多|少|好|差|大|小)",   # 最高/卖得最好/卖得最多……
    r"排(名|序|行)",                  # 排名/排序
    r"\btop\b",                       # top / Top
    r"top\s*\d",                      # top5
    r"趋势",
    r"trend",
    r"对比",
    r"比较",
    r"占比",
    r"分布",
    r"构成",
    r"为什么",
    r"原因",
    r"分析一下",
]
_COMPLEX_RE = re.compile("|".join(_COMPLEX_PATTERNS), re.IGNORECASE)


def question_classifier_node(state: SalesState) -> dict:
    # 结果由 graph 的 after_execute 真正读取：判 False 就直接出报告，跳过 Qwen 审查
    need_review = bool(_COMPLEX_RE.search(state["question"]))
    return {"need_review": need_review}
