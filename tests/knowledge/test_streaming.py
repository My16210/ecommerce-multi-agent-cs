"""
token 汇 + 流式生成。

这块的全部风险都压在**一条不变量**上：

    挂了汇时"边收边发"拼出来的文本，必须和没挂汇时一次性返回的文本**逐字相同**。

因为挂汇只该改变"用户什么时候看到答案"，绝不该改变"答案是什么"。它一旦不等，
出问题的不是界面——是 `state["answer"]`：judge 拿它判空（识别服务故障）、
多轮记忆把它写进历史、评测读它。所以这里用假模型把两条路的输出直接对着比。

另外钉住"没挂汇时是真空操作"，理由和 tests/test_timing.py 完全一样：
它挂在生产链路的每次生成上，关着的时候不能有任何成本。
"""
import threading
import time

import pytest

from ecom.knowledge.generator import answer as answer_module
from ecom.knowledge.utils import streaming
from ecom.knowledge.utils.timing import collect

# --------------------------------------------------------------------------
# 假模型
# --------------------------------------------------------------------------

PIECES = ["根据", "资料", "[1]", "，退货需要", "在 7 天内", "申请。"]
FULL_TEXT = "".join(PIECES)


class _Response:
    def __init__(self, content, usage):
        self.content = content
        self.usage_metadata = usage


class _Chunk:
    """一个流式片段。字段名和真 chunk 对齐（.text / .content / .usage_metadata）"""

    def __init__(self, text, usage=None, content=None):
        self.text = text
        if content is not None:
            self.content = content
        self.usage_metadata = usage


class FakeLLM:
    """按"怎么被调用"分别给出一次性结果和流式片段"""

    def __init__(self):
        self.invoked = 0
        self.streamed = 0

    def invoke(self, messages):
        self.invoked += 1
        return _Response(FULL_TEXT, {"input_tokens": 888, "output_tokens": 248})

    def stream(self, messages):
        self.streamed += 1
        for piece in PIECES:
            yield _Chunk(piece)
        # 开了 stream_usage 的真接口，用量挂在**最后一个** chunk 上
        yield _Chunk("", usage={"input_tokens": 888, "output_tokens": 248})


@pytest.fixture
def fake_llm(monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr(answer_module, "get_llm", lambda *a, **k: llm)
    return llm


# --------------------------------------------------------------------------
# 汇本身
# --------------------------------------------------------------------------


def test_emit_reaches_the_sink():
    received = []
    with streaming.sink(received.append):
        streaming.emit("甲")
        streaming.emit("乙")

    assert received == ["甲", "乙"]


def test_emit_without_sink_is_a_noop():
    """没挂汇时 emit 不该抛、不该留痕——它是挂在生产链路上的"""
    assert streaming.current() is None
    streaming.emit("没人听")
    assert streaming.current() is None


def test_emit_drops_empty_pieces():
    """空片段丢掉：否则界面会为一个空字符串白白重绘一次"""
    received = []
    with streaming.sink(received.append):
        streaming.emit("")
        streaming.emit("有内容")

    assert received == ["有内容"]


def test_nested_sink_shadows_then_restores():
    outer, inner = [], []
    with streaming.sink(outer.append):
        streaming.emit("外1")
        with streaming.sink(inner.append):
            streaming.emit("内")
        streaming.emit("外2")

    assert outer == ["外1", "外2"]
    assert inner == ["内"]


def test_sink_is_isolated_between_threads():
    """ContextVar 的核心承诺：两个会话的 token 不会串到一起

    Streamlit 每次交互都在自己的线程里重跑脚本。这里若用模块级全局变量，
    两个会话同时提问时，A 的答案会出现在 B 的气泡里——而且是偶发的。
    """
    seen = {}

    def worker(name: str) -> None:
        pieces = []
        with streaming.sink(pieces.append):
            for index in range(3):
                streaming.emit(f"{name}{index}")
                time.sleep(0.005)
        seen[name] = pieces

    threads = [threading.Thread(target=worker, args=(n,)) for n in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert seen["a"] == ["a0", "a1", "a2"]
    assert seen["b"] == ["b0", "b1", "b2"]


# --------------------------------------------------------------------------
# 两条路必须给出同一个答案
# --------------------------------------------------------------------------


def test_streaming_and_non_streaming_produce_identical_text(fake_llm):
    """本文件最重要的一条：挂汇只改变"什么时候看到"，不改变答案"""
    plain = answer_module.generate_answer("退货政策是什么", ["7天无理由退货"])

    pieces = []
    with streaming.sink(pieces.append):
        streamed = answer_module.generate_answer("退货政策是什么", ["7天无理由退货"])

    assert streamed == plain == FULL_TEXT
    assert "".join(pieces) == FULL_TEXT       # 汇里收到的拼起来也得是全文


def test_streaming_reports_usage(fake_llm):
    """流式下 token 用量仍要进采集器

    不钉这条的话，llm.py 里 stream_usage=True 哪天被删掉都没人知道——
    症状是 profile.py 的成本表变成 ￥0.0000/题，而它打印得毫无异常。
    """
    with collect() as collector:
        with streaming.sink(lambda _: None):
            answer_module.generate_answer("问题", ["context"])

    assert collector.metrics["llm_input_tokens"] == 888
    assert collector.metrics["llm_output_tokens"] == 248


def test_without_sink_the_model_is_not_streamed(fake_llm):
    """没挂汇时不该走流式——那条路径要一字不动地保持原样"""
    answer_module.generate_answer("问题", ["context"])
    assert (fake_llm.invoked, fake_llm.streamed) == (1, 0)


def test_with_sink_the_model_is_streamed(fake_llm):
    with streaming.sink(lambda _: None):
        answer_module.generate_answer("问题", ["context"])
    assert (fake_llm.invoked, fake_llm.streamed) == (0, 1)


# --------------------------------------------------------------------------
# 从 chunk 里取文本
# --------------------------------------------------------------------------


def test_chunk_text_reads_the_text_attribute():
    assert answer_module._chunk_text(_Chunk("你好")) == "你好"


def test_chunk_text_reads_string_content():
    """.text 拿不到时回退到 .content（字符串形态）"""
    chunk = _Chunk("", content="来自 content 的文本")
    # 真 chunk 的 .text 会把 content 里的文本聚合出来，这里刻意把 .text 置空，
    # 逼它走 .content 那一支
    assert answer_module._chunk_text(chunk) == "来自 content 的文本"


def test_chunk_text_reads_block_list_content():
    """分块内容（带推理字段的模型会这样）只取文本块

    漏了这条的症状是"答案缺一段"——看起来像模型自己没说完，基本追不到这里。
    """
    chunk = _Chunk("", content=[
        {"type": "reasoning", "text": "这里是想的过程，不该出现在答案里"},
        {"type": "text", "text": "这里才是答案"},
    ])
    assert answer_module._chunk_text(chunk) == "这里才是答案"


def test_chunk_text_of_an_empty_chunk_is_empty():
    assert answer_module._chunk_text(_Chunk("")) == ""
