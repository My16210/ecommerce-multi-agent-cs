"""
`strip_to_json` 的单测。全离线。

**不变量：只要文本里存在一个完整的 JSON 值，就不能返回 None。**
听起来像废话，但这里真栽过：载荷自己含 ``` 代码块时，旧实现会返回 None，
而调用方（判定 / 出题 / 出题校验）把 None 当"这批坏了"处理——**静默丢数据**，
正是这个模块开头那段警告的场景。所以下面那组"载荷自带围栏"的用例是重点。
"""
import json

from ecom.knowledge.utils.json_utils import strip_to_json


class TestPlainJson:
    def test_bare_object(self):
        assert json.loads(strip_to_json('{"a": 1}')) == {"a": 1}

    def test_bare_array(self):
        assert json.loads(strip_to_json('[{"a": 1}]')) == [{"a": 1}]

    def test_leading_and_trailing_prose(self):
        """模型爱在前后各补一句人话——那是散文，不是载荷"""
        text = '好的，结果如下：\n{"a": 1}\n以上。'
        assert json.loads(strip_to_json(text)) == {"a": 1}

    def test_stops_where_the_json_ends(self):
        """JSON 后面还有一段带花括号的解释时，不能把两段一起框进来——
        这正是用 raw_decode 而不是 find/rfind 的理由"""
        text = '{"a": 1}\n其中 {b} 表示另一个含义。'
        assert json.loads(strip_to_json(text)) == {"a": 1}

    def test_braces_inside_a_string_do_not_break_it(self):
        """字符串里的花括号不算结构——find("{")+rfind("}") 会在这里翻车"""
        text = '{"cmd": "curl {url} -d {body}"}'
        assert json.loads(strip_to_json(text))["cmd"] == "curl {url} -d {body}"


class TestFencedJson:
    def test_json_inside_a_fence(self):
        assert json.loads(strip_to_json('```json\n{"a": 1}\n```')) == {"a": 1}

    def test_payload_that_itself_contains_a_code_fence(self):
        """**这条钉的是一个真踩过的坑。** 判定模型把代码当证据引回来时，
        载荷里的 evidence 字符串自己就带着 ``` 代码块。旧实现非贪婪地取
        "第一个围栏块"，抓到的是一条从载荷中间开始、被截断的碎片，
        于是返回 None，判定被误记成"解析失败"。
        实测同一条答案：0 个围栏时正常，4～6 个围栏时必失败。"""
        payload = {"claims": [
            {"claim": "提交任务", "label": "supported",
             "evidence": "```python\nurl = f\".../{batch_id}\"\n```"},
        ], "score": 1.0}
        text = "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"
        assert text.count("```") == 4          # 载荷自带一对，外面再包一对
        assert json.loads(strip_to_json(text)) == payload

    def test_prose_with_a_braced_example_before_the_real_payload(self):
        """反过来的一条：正文里先出现一个带花括号的示例，真载荷在后面围栏里。
        走的是"扫全文"那条路——`{示例}` 不是合法 JSON，raw_decode 跳过它，
        接着就撞上真正的载荷。围栏兜底那步用不上，但它也不该抢答。"""
        text = '例如 {示例} 这种写法不行。真正的结果：\n```json\n{"a": 1}\n```'
        assert json.loads(strip_to_json(text)) == {"a": 1}

    def test_object_whose_evidence_contains_a_fenced_code_block_with_a_bracket_literal(self):
        """**这条钉的是 `ans-0015` 那个真踩过的坑，也是"先围栏后全文"的死刑判决。**

        载荷是对象，但它的 evidence 字段里嵌了一段 ``` 围栏，围栏里的代码又含
        `[...]` 字面量。先"逐块试围栏"的话，非贪婪取到的第一个围栏块就是那段代码，
        raw_decode 把里面的 `[463..., 463...]` 当成合法 JSON 数组返回——调用方拿到
        数组、要的是对象，直接判成解析失败（实测 json.loads(全文) 是通的）。
        扫全文优先后，第一个 `{` 就把整个对象吃下来了。"""
        payload = {"claims": [
            {"claim": "支持按 ID 批量删除", "label": "supported",
             "evidence": "可以通过ID列表删除：\n```python\n"
                         '    filter="id in [463480757150366907, 463480757150366908]",\n'
                         "```"},
        ], "score": 1.0}
        text = json.dumps(payload, ensure_ascii=False)
        assert json.loads(strip_to_json(text)) == payload


class TestNoJson:
    def test_returns_none_not_an_exception(self):
        """剥不出来是调用方的事，不是这里的事——所以明确返回 None，
        不吞成默认值（吞了会让"判定全变成不值得"这种故障看不出来）"""
        assert strip_to_json("模型今天不想说 JSON") is None
        assert strip_to_json("") is None
        assert strip_to_json(None) is None

    def test_incomplete_json_returns_none(self):
        """被 token 上限截断的半截 JSON 属于"没有完整值"，同样是 None"""
        assert strip_to_json('{"claims": [{"claim": "a"') is None
