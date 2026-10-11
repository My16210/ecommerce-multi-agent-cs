"""守 token 统计的两个名字。

这条 bug 在真容器里跑出来：报告结尾写着"本次分析共消耗 0 个 token"，
eval.py 的成本列永远是 $0.0000——因为代码读的是 OpenAI **原始响应**里的
prompt_tokens，而 LangChain 统一的 usage_metadata 用的是 input_tokens。
两个名字都存在，只是不在同一个地方，读错了不报错、只返回 0。

全项目另外四处读的都是 input_tokens，所以这里的第二组测试其实在守
"别再退回那个名字"。
"""
from ecom.analytics.utils.tokens import extract_usage


class _Response:
    def __init__(self, usage_metadata=None, response_metadata=None):
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata


def test_reads_langchain_standard_keys():
    # 真跑一次 DeepSeek/Qwen 拿到的就是这个形状
    prompt, completion = extract_usage(_Response(usage_metadata={"input_tokens": 812, "output_tokens": 143}))
    assert (prompt, completion) == (812, 143)


def test_reads_the_same_keys_as_the_rest_of_the_repo():
    """和 orchestrator 那份实现在同一个输入上必须给出同一个答案。

    两份实现、一个概念——这正是当初读错名字能藏住的原因。
    真去比对，而不是"看起来一样"。
    """
    from ecom.orchestrator.utils.tokens import usage_of

    response = _Response(usage_metadata={"input_tokens": 7, "output_tokens": 3})
    prompt, completion = extract_usage(response)
    assert (prompt, completion) == (usage_of(response)["in"], usage_of(response)["out"])


def test_falls_back_to_the_raw_openai_names():
    """少数兼容端点只填 response_metadata，这时退回原始键名。"""
    response = _Response(response_metadata={"token_usage": {"prompt_tokens": 55, "completion_tokens": 9}})
    assert extract_usage(response) == (55, 9)


def test_missing_usage_is_zero_and_not_an_exception():
    # usage_metadata 为 None 是合法的（有些端点不给），不能因此崩掉整张图
    assert extract_usage(_Response()) == (0, 0)
    assert extract_usage(_Response(usage_metadata=None)) == (0, 0)


def test_empty_usage_metadata_falls_through_to_the_raw_metadata():
    """usage_metadata 是空 dict 时不能就此认定"没有用量"，还要看原始字段。"""
    response = _Response(
        usage_metadata={},
        response_metadata={"token_usage": {"prompt_tokens": 12, "completion_tokens": 4}},
    )
    assert extract_usage(response) == (12, 4)
