"""控制台输出的字符集保险：打不出来的字符换成替代符。

【这个坑】Windows 控制台默认 cp936（GBK）。回答里的 `¥`（U+00A5）、`✅` 这类字符
不在 GBK 里，`print` 会直接抛 `UnicodeEncodeError`——**而且是在回答已经生成之后
才崩**，用户看到的是崩溃而不是答案。

【为什么是"替换"而不是把 stdout 强行 reconfigure 成 UTF-8】
理由和 `knowledge/cli/review.py` / `orchestrator/cli/returns.py` 那两处写的一样：
在别人的终端上强行改口径，比按终端自己的能力降级更容易炸。

【为什么收在一个函数里】
`main.py`（打印模型生成的答案）和 `cli/memory.py`（打印模型生成的记忆摘要）都要它，
而它们打印的**都是模型产出的文本**——那是最容易夹带怪字符的地方。两个私有副本迟早
分叉，而且这种分叉**只在别人的终端上才看得见**，本地永远是绿的。
"""
import sys


def safe(text) -> str:
    """打不出来的字符换成替代符，保证 `print` 不抛。"""
    text = str(text)
    encoding = sys.stdout.encoding or "utf-8"
    try:
        text.encode(encoding)
        return text
    except (UnicodeEncodeError, LookupError):
        return text.encode(encoding, errors="replace").decode(encoding, errors="replace")
