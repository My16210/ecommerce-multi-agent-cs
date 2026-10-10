"""
生成过程的 token 汇（和 ecom/knowledge/utils/timing.py 是同一个模式）

为什么需要它：`generate_answer()` 是**同步返回整段文本**的，而界面要让答案
逐字出现。两条路：

  a) 让 LangGraph 用 `stream_mode="messages"` 把节点内模型调用的 token 流出来；
  b) 在应用层显式挂一个"每收到一个 token 就调一下"的汇。

选 b，两个理由：

  1. **a 的行为不好把握。** 它依赖 LangGraph 对节点内模型调用的插桩，而这个
     行为在不同版本间变过；而且它会把图里**所有**模型调用一起流出来——哪天
     判定（judge）也挪进图里，用户会先看到一段判定理由再看到答案，得靠 tag
     去过滤才分得清是谁在说。b 是确定的：谁设了汇、汇收到了什么，测试里能直接断言。
  2. **只改"什么时候看到"，不改"答案是什么"。** 流式分支自己把 token 拼回全文，
     拼出来的那份就是 `state["answer"]`。所以 judge、多轮记忆、eval、非流式调用方
     全都看不到区别——有测试钉住"流式拼出来的文本 == 非流式的文本"。

`ContextVar` 而不是模块级全局，理由和 timing.py 一样：Streamlit 每交互一次就在
自己的线程里从头重跑脚本，用全局变量会让两个会话的 token 串到一起。

**调用方只有一个**：`ecom/knowledge/generator/answer.py`。它读 `current()` 决定走不走流式。
别的模块不要 emit——汇的语义是"答案是这么长出来的"，掺进别的东西就废了。
"""
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

# 汇的签名：收到一个 token 片段就被调一次
Sink = Callable[[str], None]

_sink: ContextVar[Sink | None] = ContextVar("token_sink", default=None)


@contextmanager
def sink(callback: Sink) -> Iterator[Sink]:
    """在 with 块里把 callback 设为当前汇，退出时还原（嵌套时还原成外层那个）"""
    token = _sink.set(callback)
    try:
        yield callback
    finally:
        _sink.reset(token)


def current() -> Sink | None:
    """取当前的汇；没挂汇时返回 None——调用方据此走"一次性返回"的老路径"""
    return _sink.get()


def emit(text: str) -> None:
    """
    把一个 token 交给当前的汇。

    和 `timing.stage()` 一样：**没挂汇时是一次 `get()` 就返回**——不分配、
    不加锁、不碰字符串。所以非流式路径上留着这个调用不花钱。
    空片段直接丢掉，免得界面多一次无意义的重绘。
    """
    callback = _sink.get()
    if callback is not None and text:
        callback(text)
