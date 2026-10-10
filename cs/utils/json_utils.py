"""
从大模型的自由文本里剥出 JSON。

算法与 rag_knowledge_base 的 `src/utils/json_utils.py` **同源**（那两版 bug 也是
那边修完搬过来的）。这里没有直接 import 它，是因为路由这一步必须在"知识库仓库
根本不在场"时也能工作——它是 supervisor 自己的零件，不该被一个 sibling 仓库的
存在与否绑住。代价是三十行重复代码，值得。

**【两步的顺序是踩出来的，不是想出来的】先扫全文，再逐块试围栏。**

载荷的外层就是 JSON 时（模型直接吐、或裹一层围栏吐，都是常态），扫全文在第一个
`{`/`[` 处 raw_decode 就把整个载荷吃下来了。反过来先"逐块试围栏"会出事：载荷的
字符串字段里常含 ``` 围栏（比如订单号备注里贴了段代码、或路由 reason 里引了例子），
而围栏里又含 `[...]`/`{...}` 字面量，非贪婪取到的"第一个围栏块"等于那段内部文本，
raw_decode 欣然把它当成"一个合法 JSON 值"返回——调用方要对象、拿到数组，
直接判成解析失败，而全文其实可解析。
"""
import json
import re

# 围栏块。注意这里不"取第一个"就返回——见上面那段。
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _first_json(text: str) -> str | None:
    """从 text 的任意位置起，剥出第一个**完整** JSON 值；没有返回 None

    用 JSONDecoder().raw_decode 逐个候选位置试，而不是 `find("{")` 配 `rfind("}")`：
    后者在"JSON 后面还有一段带花括号的解释"时会把两段一起框进来，直接解析失败。
    raw_decode 能停在 JSON 真正结束的位置。
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
    """返回 text 里第一个**完整** JSON 值的原文；找不到返回 None。

    不吞成默认值：解析失败怎么处理是调用方的事（路由可以重试或退回规则兜底），
    在这里假装"成功"会让故障看不出来。
    """
    raw = (text or "").strip()

    found = _first_json(raw)
    if found is not None:
        return found

    # 兜底：正文里先出现一个合法 JSON 示例、真载荷在后面围栏里。很少见，留一手。
    for block in _FENCE.finditer(raw):
        found = _first_json(block.group(1))
        if found is not None:
            return found

    return None


def loads_or_none(text: str):
    """剥出来再解析，任一环节失败都返回 None（不抛）。"""
    stripped = strip_to_json(text)
    if stripped is None:
        return None
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return None
