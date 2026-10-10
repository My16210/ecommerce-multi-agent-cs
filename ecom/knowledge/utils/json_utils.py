"""
从大模型的自由文本里剥出 JSON。

prompt 里写了"只输出 JSON，不要 markdown 代码块"，但模型照样会加围栏、
会在前面写一句"好的，结果如下"、会在后面补一句"以上"。这类脏输出在两个
地方都要处理（judge.py 剥一个对象，build_eval_set.py 剥一个数组），
而且处理得不干净的症状是**静默丢数据**——判定全变成"不值得"、出题全被丢掉，
两种都不报错。所以剥法只能有一份。

这里**只定位、不兜底**：解析失败怎么处理是各调用方自己的事
（判定宁可漏收也不能污染队列；出题丢掉这一批就行），
所以这里不吞成默认值，而是明确返回 None。
"""
import json
import re

# 围栏块。**注意这里不"取第一个"**——载荷本身经常包含 ``` 包起来的代码
# （判定模型把代码当证据引回来的时候就是这样），那时候"第一个围栏"是一条
# 从载荷中间开始、被截断的片段，拿它去解析永远失败。
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _first_json(text: str) -> str | None:
    """从 text 的任意位置起，剥出第一个**完整** JSON 值；没有返回 None

    用 JSONDecoder().raw_decode 逐个候选位置试，而不是简单地
    `find("{")` 配 `rfind("}")`：后者在"JSON 后面还有一段带花括号的解释"
    时会把两段一起框进来，直接解析失败。raw_decode 能停在 JSON 真正结束的位置。
    """
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        return text[index:end]
    return None


def strip_to_json(text: str) -> str | None:
    """返回 text 里第一个**完整** JSON 值（对象或数组）的原文；找不到返回 None

    **先扫全文，再逐块试围栏**。这个顺序是踩出来的，不是想出来的——

    载荷的外层就是 JSON 时（模型直接吐、或裹一层围栏吐，都是常态），扫全文
    在第一个 `{`/`[` 处 raw_decode 就把整个载荷吃下来了，一步到位。反过来先
    "逐块试围栏"会出事：载荷的字符串字段里常含 ``` 围栏（判定把代码当证据
    引回来时必然如此），而围栏里的代码又含 `[...]`/`{...}` 这样的字面量
    （`filter="id in [463480757150366907, 463480757150366908]"`）。非贪婪地
    取"第一个围栏块"拿到的正是那段代码，raw_decode 欣然把里面那个 `[...]`
    当成"一个合法 JSON 值"返回——调用方于是拿到一个数组，而它要的是对象，
    直接判成解析失败。实测 `ans-0015` 就是这么坏的：全文可解析（json.loads
    通），但围栏优先把它剥成了证据里的 ID 数组。

    围栏那一步退到兜底，治的是**反过来**的情况：正文里先出现一个合法 JSON
    示例、真正的载荷才在后面围栏里，扫全文会先撞上那个示例。这种很少见，
    但留一手不吃亏——而且它现在只在全文扫不出来时才生效，不会再抢答。
    """
    raw = (text or "").strip()

    found = _first_json(raw)
    if found is not None:
        return found

    for block in _FENCE.finditer(raw):
        found = _first_json(block.group(1))
        if found is not None:
            return found

    return None
