"""会话内长期记忆：把滑出窗口的消息蒸馏成"可执行的事实"，需要时按语义召回。

【它解决的是哪一件事】
`session.history()` 只带最近 `HISTORY_TURNS` 轮。聊到第 20 轮时，第 3 轮说过的
订单号、诉求、结论都已经滑出窗口——模型看不见了，于是会重复问、或者答非所问。
这一层把**滑出窗口的那一段**压成一条摘要存进 `memories` 表，需要时按语义召回，
拼在真实历史**前面**。

注意它和 `utils/entities.carried_order_id`（指代消解）是**两件不同的事**：
那个管"这句话在说上一轮提过的哪一笔"（第 2 轮就发生），这个管"早期说过的话还在不在"
（超过 `HISTORY_TURNS` 轮才发生）。两者要改的地方不重叠。

【为什么存"事实"而不是"关键词"】
"超出范围就把前文提炼成关键字另存"这个想法很自然（便宜、好检索），但它**只能
告诉你聊过什么话题，不能告诉你结论是什么**——而多轮对话里真正要继承的恰恰是
结论（"用户要为订单 52644 退货，但还没确认"）。所以存的是**摘要**，关键词只当
**索引**：CLI 列表里给人扫一眼，不参与回答。

【为什么是"按需召回"而不是"每轮全灌"】
全灌 = 每轮把整段记忆塞进 prompt：token 随会话长度线性涨，而且大部分记忆和当前
问题无关，是纯噪声。召回 = 先按语义挑最相关的几条（`MEMORY_TOP_K`）。

**不设相似度阈值。** 池子是这个会话自己的、条数天然少；加一个没实测过的阈值，
只会静默丢掉记忆——"丢了还不报错"是这里最难查的一种失败。

【它为什么和 session.py 平级、不进 core/】
它要联网（模型 + embedding），而 `session.py` 的定位是"只有 SQLite 的笨存储"
（那边模块 docstring 明说这个边界）。把网络依赖塞进那条边界里，`session.py`
的测试就没法离线跑了。

【依赖边界：不引 numpy】
浮点 → BLOB 用标准库 `array`，`_cosine` 纯 Python。全仓库除 analytics 沙箱外
没有数值栈（`embedder.py` 也没有）。为了存个向量把 numpy 拉进依赖，是拿一整条
依赖边界换一行代码——**这是依赖边界的决定，不是性能取舍**。
"""
import array

from ecom.knowledge.embedding.embedder import get_embedding
from ecom.orchestrator import session
from ecom.orchestrator.settings import MEMORY_DISTILL_MAX_MESSAGES, MEMORY_TOP_K
from ecom.orchestrator.utils.json_utils import loads_or_none
from ecom.orchestrator.utils.llm import get_llm
from ecom.orchestrator.utils.logger import setup_logger

logger = setup_logger(__name__)

DISTILL_PROMPT = """下面是一段电商客服对话的**较早部分**（已经滑出上下文窗口）。
把它压成一条**以后还会用到**的记忆。

只输出 JSON，不要代码块：
{{"summary": "一到三句话", "keywords": ["词", "词"]}}

要求：
- **summary 写"可执行的事实"，不是话题**：哪一笔订单（订单号原样保留）、
  用户的诉求、最后落到什么结论。这些是后面的对话要**接着用**的。
- 不确定的信息**不要写**。宁可短，也不要把没出现过的订单号、金额、结论编进去——
  这条摘要以后会被当成事实喂给模型，编进去的东西**没有人会再核一遍**。
- 寒暄、客套、"客服说会去查一下"这种没有结论的过场，**不要写**。
- keywords 只做检索索引和列表展示，三五个词就够，**不要把它塞进 summary**。

【对话】
{transcript}"""


# ---------------------------------------------------------------------------
# 向量：打包 / 解包 / 余弦（全是纯 Python，见模块开头那条依赖边界）
# ---------------------------------------------------------------------------
def _pack(vec: list) -> bytes:
    """float32 序列化。float32 够用：BGE-M3 出来的向量做余弦排序，双精度那几位
    有效数字影响不了名次，但存储要翻倍。"""
    return array.array("f", vec).tobytes()


def _unpack(blob: bytes) -> list:
    vec = array.array("f")
    vec.frombytes(blob)
    return list(vec)


def _cosine(a: list, b: list) -> float:
    """
    余弦相似度。长度对不上时返回 0.0，**不抛异常**。

    长度对不上只可能来自"换过 embedding 模型"或脏数据。那种情况下让这一条排到
    最后（0.0 在余弦里就是"毫不相关"），比让整次召回失败合理——一条过期的记忆
    不该把整个会话的记忆功能弄挂。这是**已知限制**（没有 embedding 版本号落点，
    做不了自动重算），不是漏处理。
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# 蒸馏：把滑出窗口的消息压成一条记忆（只在 CLI 里调，问答路径不碰它）
# ---------------------------------------------------------------------------
def _transcript(rows: list) -> str:
    return "\n".join(
        f"{'用户' if row['role'] == 'user' else '客服'}：{row['content']}" for row in rows
    )


def _summarize(rows: list, llm) -> tuple:
    """让模型把这段对话压成 (摘要, 关键词串)。解析不出来返回 `("", "")`。"""
    response = llm.invoke([
        ("system", "你在为客服系统整理对话摘要，只输出 JSON。"),
        ("human", DISTILL_PROMPT.format(transcript=_transcript(rows))),
    ])
    parsed = loads_or_none(response.content)
    if not isinstance(parsed, dict):
        return "", ""

    summary = str(parsed.get("summary") or "").strip()
    raw = parsed.get("keywords")
    if isinstance(raw, list):
        keywords = "，".join(str(k).strip() for k in raw if str(k).strip())
    else:
        keywords = str(raw or "").strip()
    return summary, keywords


def distill(session_id: str, llm=None) -> dict:
    """
    把这个会话**滑出窗口、且还没蒸馏过**的消息压成一条记忆，写库。

    返回 `{"created", "through_message_id", "messages", "reason"}`。

    `created=False` 要能区分两种情况，否则 CLI 报不出人话：
      · `reason="无可蒸馏"`——会话还短，没有滑出窗口的消息（**这不是错误**）
      · `reason="已被蒸馏"`——另一个进程先跑了，`UNIQUE` 把这次挡掉了（幂等生效）

    **embedding 失败会让它抛出去**（水位线不推进，下次可重跑）——绝不写一条
    永远召不回的 NULL 向量的记忆：那条约等于"这条记忆存在但永远选不上"，
    比没有更糟，因为 `list` 里看得到它。
    """
    sid = (session_id or "").strip()
    if not sid:
        return {"created": False, "through_message_id": 0, "messages": 0, "reason": "没有会话号"}

    rows = session.out_of_window_messages(sid, MEMORY_DISTILL_MAX_MESSAGES)
    if not rows:
        return {"created": False, "through_message_id": 0, "messages": 0, "reason": "无可蒸馏"}

    llm = llm or get_llm(0)
    summary, keywords = _summarize(rows, llm)
    through = max(row["id"] for row in rows)
    if not summary:
        # 模型没给出可用的摘要。**不写、水位线不推进**，下次可重跑——
        # 写一条空记忆会让它永远占着水位线，把这段对话吞掉。
        return {
            "created": False, "through_message_id": 0, "messages": len(rows),
            "reason": "模型没给出可用摘要",
        }

    embedding = _pack(get_embedding([summary])[0])
    created = session.add_memory(sid, summary, keywords, embedding, through)
    return {
        "created": created,
        "through_message_id": through,
        "messages": len(rows),
        "reason": "" if created else "已被蒸馏",
    }


# ---------------------------------------------------------------------------
# 召回：按语义挑最相关的几条（每轮问答都会调，所以"没有记忆"必须最便宜）
# ---------------------------------------------------------------------------
def recall(session_id: str, question: str) -> list:
    """
    召回这个会话里和 `question` 最相关的若干条记忆，**最相关的在前**。

    **池子为空就直接返回 `[]`，一次 embedding 都不调。** 这条是硬约束，两个理由：
    老会话（没跑过 distill）每问一句都白花一次 embedding 调用；以及离线测试要能
    在没有记忆的库上跑——不早退的话，每个用例都会去连硅基流动。

    返回的行比 `memories_for` 多一个 `score`（余弦），CLI 的 dry-run 直接打印它。
    """
    pool = session.memories_for(session_id)
    if not pool:
        return []

    query_vec = get_embedding([question])[0]
    scored = [
        {**row, "score": _cosine(query_vec, _unpack(row["embedding"]))}
        for row in pool
        if row.get("embedding")
    ]
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:MEMORY_TOP_K]


def augmented_history(session_id: str, question: str, history: list) -> list:
    """
    把召回的记忆拼在**真实历史前面**，返回一份**新的**列表交给 `answer()`。

    【为什么是"每轮现拼、不落盘、不累积"】
    记忆用 `role="memory"` 走 `chat_history` 这条现成的通道（worker 的通用契约
    本来就是 `fn(question, chat_history)`），**不新增 `CSState` 字段**。
    省下十几个文件的改动不是主要理由——**多一条要与 `chat_history` 长期保持同步
    的平行管道才是**（少一个能分叉的地方才挡得住，见 `utils/ids.py` 的教训）。

    代价是调用方必须守一条规矩：**拿到返回值直接传给 `answer()`，不要再存回去**。
    尤其 `main.py` 的交互模式自己维护一个内存 `history` 列表——存回去就会每轮
    重复累积一份记忆。

    记忆不进 `messages` 表（`append_turn` 只写真轮次），所以它不会出现在
    `cli.session show` 里，也不会被 `history()` 的 SQL 截断。
    """
    history = list(history or [])
    memories = recall(session_id, question)
    if not memories:
        return history
    return [{"role": "memory", "content": m["summary"]} for m in memories] + history
