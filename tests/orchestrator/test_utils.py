"""基础设施：剥 JSON 的两步顺序 / token 记账的容错。"""
import json

import pytest

from ecom.orchestrator.utils.json_utils import loads_or_none, strip_to_json
from ecom.orchestrator.utils.tokens import EMPTY, add, usage_of


# ---------------------------------------------------------------------------
# strip_to_json
# ---------------------------------------------------------------------------
def test_plain_payload():
    assert strip_to_json('{"a": 1}') == '{"a": 1}'


def test_fenced_payload():
    assert strip_to_json('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_payload_sitting_in_prose():
    text = '好的，判断如下：{"workers": ["order"]} 就这些。'
    assert strip_to_json(text) == '{"workers": ["order"]}'


def test_json_followed_by_a_braced_explanation_is_not_over_captured():
    # find("{") + rfind("}") 会在这里把两段一起框进来，直接解析失败。
    text = '{"a": 1} 补充说明：如果 x 则 {b} 否则 {c}'
    assert strip_to_json(text) == '{"a": 1}'


def test_string_field_containing_a_fenced_bracket_literal():
    """
    这条是一条真实事故的回归测试（从 rag_knowledge_base 搬过来的）。

    载荷的字符串字段里贴了一段围栏代码，围栏里又有 `[...]` 字面量。
    如果实现是"先逐块试围栏"，`_FENCE` 非贪婪取到的第一块正是那段内部文本，
    raw_decode 欣然把它当成一个合法 JSON **数组**返回——调用方要对象却拿到数组，
    判成解析失败，而全文其实好得很。所以顺序必须是"先扫全文，围栏只当兜底"。
    """
    inner = '```\nfilter="id in [463480757150366907, 463480757150366908]"\n```'
    text = json.dumps(
        {"verdict": "unsupported", "evidence": f"文档里写的是 {inner} 和答案对不上"},
        ensure_ascii=False,
    )
    parsed = loads_or_none(text)
    assert isinstance(parsed, dict)
    assert parsed["verdict"] == "unsupported"

    # 顺带留个反例：载荷本身坏了（字符串里有裸换行，JSON 不允许）时，
    # 扫描会一路滑到内层那个数组上，raw_decode 认它是合法 JSON，于是返回一个 list。
    # 这正是"调用方要对象、拿到数组"的那次事故的症状——所以**解析成功后还要校验类型**，
    # 不能因为 strip_to_json 没返回 None 就当它是对的。
    broken = '{"v": "x\ny [1, 2]"}'
    assert not isinstance(loads_or_none(broken), dict)


def test_returns_none_instead_of_a_default():
    # 解析失败怎么处理是调用方的事（路由要重试/兜底）。这里假装成功会把故障藏起来。
    assert strip_to_json("我建议交给订单系统") is None
    assert loads_or_none("") is None
    # 扫描只从 `{` / `[` 起（结构化载荷就是这样），所以裸标量不算数
    assert loads_or_none("1 2 3") is None


# ---------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------
class _Response:
    def __init__(self, usage):
        self.usage_metadata = usage


def test_usage_is_read_when_present():
    assert usage_of(_Response({"input_tokens": 7, "output_tokens": 3})) == {"in": 7, "out": 3}


def test_missing_usage_counts_as_zero_without_raising():
    # 挖不到就当 0——但**别**因此认为"这次不花钱"，它更可能是"这次没记上"。
    assert usage_of(_Response(None)) == EMPTY
    assert usage_of(object()) == EMPTY


def test_add_merges_both_directions():
    assert add({"in": 1, "out": 2}, {"in": 10, "out": 20}) == {"in": 11, "out": 22}
    assert add(EMPTY, EMPTY) == EMPTY


@pytest.mark.parametrize("bad", [None, {}])
def test_add_tolerates_a_missing_side(bad):
    assert add({"in": 1, "out": 2}, bad or {}) == {"in": 1, "out": 2}
