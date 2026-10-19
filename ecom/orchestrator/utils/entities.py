"""跨轮对话里的实体 —— **指代消解的判定只写这一份**。

为什么单独一个模块
------------------
`core/router.py`（决定交给谁）和 `workers/order.py`（挑出要查的那一笔）都要
"接着上一轮说"。两处各写一份判定就会分叉，**而且分叉不报错**——这和
`utils/ids.py` 模块 docstring 里记的那次事故是同一个形状（订单号位数门槛两处
各写一份，路由认得出的抽取层认不出）。所以只写一份，两边 import 同一个函数。

【为什么判据这么保守：消解不了就不猜】
消解错了比消解不了更糟。消解不了只是把现状维持住（多问一句 / 答不出来），
消解错了会答非所问，甚至在写操作上替用户对错误订单办事——所以下面三条规则
**任何一条不满足就返回空**，宁可携带不上。
"""
import re

from ecom.orchestrator.settings import HISTORY_TURNS
from ecom.orchestrator.utils.ids import ORDER_ID_RE

# 指代词。出现它，这句话才**可能**是在说上一轮那件事；没有它就不携带。
#
# 【已知的不精确，是选的方向不是漏的】「那」「这个」在中文里也当连词和普通
# 指示词用（"那我先看看"），所以会**偶尔多带一个 order worker**。代价是答案里
# 多一句澄清（order worker 的 _ask_before_writing），比"整个订单问题答不出来"
# 轻得多。这和 router 里"漏调一个 worker = 少答一半、多调一个 = 多花一次调用"
# 是同一条不对称，只是这次多出来的那次调用用户看得见——接受它。
_ANAPHORA_RE = re.compile(r"那|它|这单|这笔|该单|这个|刚才|上面|此单|刚刚")


def carried_order_id(question: str, chat_history: list) -> str:
    """
    当前这句话没说订单号、但用了指代词时，从最近的**用户发言**里取回唯一候选。

    三条保守规则，任何一条不满足就返回 `""`：

    1. **只扫 user 消息，不扫 assistant**。assistant 的回答里 5 位以上的数字
       太多（销售额、总条数、退货率），`ORDER_ID_RE` 会把它们当成订单号。
    2. **只扫最近 `HISTORY_TURNS` 条用户发言**，从后往前。再往前的是另一件
       事了，接着说不等于还在这件事上。
    3. **那条消息里若有 ≥2 个不同的订单号**（"对比 52644 和 33333"），返回空 ——
       消解不了的歧义就不要替用户猜。
    """
    if not _ANAPHORA_RE.search(question):
        return ""
    if ORDER_ID_RE.search(question):
        # 这一轮自己说了订单号，用当轮那个，不用猜。
        return ""

    user_msgs = [
        str(turn.get("content") or "")
        for turn in (chat_history or [])
        if str(turn.get("role") or "") == "user"
    ]
    for content in reversed(user_msgs[-HISTORY_TURNS:]):
        ids = {m.group(1) for m in ORDER_ID_RE.finditer(content)}
        if not ids:
            continue
        # 最近那条"提过订单号"的用户发言就是答案：唯一 → 用它，多个 → 歧义，不猜。
        return ids.pop() if len(ids) == 1 else ""
    return ""
