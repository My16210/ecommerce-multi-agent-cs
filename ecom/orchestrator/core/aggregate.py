"""
汇总节点：把各 worker 的结论收成一条给用户的话，并决定**下一步**。

它是 supervisor 图里唯一一个"看着全部结果做决定"的地方，所以三件事都在这里：
合并答复、判断要不要重路由、判断要不要转人工。

【判据是有优先级的一串，短路返回，顺序不能换】
  1. 有可用结论                          → answer
  2. 没有，但还有没试过的单元、没到轮次上限 → re_route
  3. 没有，但手上有话可说（澄清/故障说明） → answer
  4. 什么都没有                          → human（转人工）

【为什么"坏了"不触发重路由，"没有"才触发】
ok=False 是"这个 worker 本身没跑起来"（embedding 服务挂、数据库文件锁住、子进程超时）。
另一头多半在同一个网络、同一台机器上，再试一次大概率还是坏——那就不叫重试，叫
"服务一抖动就换个 worker 再抖一次"，白烧一轮时间还把用户拖住。
insufficient=True 才是**有信息量的失败**：它看过了，这个单元的知识范围里确实没有。
这时换一个单元是有依据的赌注。

所以还有一条：**全部 ok=False 时不给重路由**（`alive` 那道门），直接告诉用户
这块暂时用不了并转人工。系统性的坏，换人不解决。

【为什么只有一个结论时不叫 LLM】
没有东西可合并时调模型，只会把一段本来忠实的话改写成可能不忠实的另一段话——
**凭空多一次幻觉机会，换来零信息增益**。所以 len(usable) == 1 时原样返回。

【汇总输出为什么是自由文本，不是 JSON】
早先版本要求汇总也返回 {"answer": ...}，结果模型经常把整段正文塞进 JSON 字符串里，
换行和引号全被转义，反而更容易崩、更难读。**人要读的正文就让它直接是正文。**

【这里**刻意不读** `chat_history`】
调用方（API / 界面 / CLI）会传一整份对话历史进来，`CSState` 里有这个字段——
但这个文件一次都不该碰它。理由和上面那条同源：汇总只做"各说各的、不替谁判对错"，
它手上已经有三个 worker 的**结构化结论**了；再把历史塞给它，等于给它素材去
改写和脑补，**多一次幻觉机会、零信息增益**。

指代消解（「那能退吗」指哪一笔）是 router 和 order worker 的事，已经在那两处
做完了——轮到汇总时，"这笔订单"早就被解析成了具体订单号。汇总不需要知道用户
上一轮说过什么。所以"历史只在汇总时用一次"那句老话是**说反了**：它一次都没用过，
真正用的是 router、order worker 和 knowledge worker。
"""
from ecom.orchestrator.settings import MAX_ROUNDS, SYNTH_TEMPERATURE, SYNTH_TIMEOUT
from ecom.orchestrator.utils.llm import get_llm
from ecom.orchestrator.utils.logger import setup_logger
from ecom.orchestrator.utils.tokens import EMPTY, usage_of
from ecom.orchestrator.workers.base import round_results, worker_names

logger = setup_logger(__name__)

# worker 名 → 给人看的短标签。汇总时分节用，CLI/前端也用它。
WORKER_LABELS = {
    "knowledge": "知识库",
    "order": "订单",
    "analytics": "数据分析",
}

# worker 挂了时对用户说的话。**不把异常原文给用户**——"ModuleNotFoundError:
# No module named 'pymysql'" 对用户没有意义，还等于把内部结构报给外面。
# 具体错误进日志和 trace，用户只该看到"这块暂时不可用"。
_FAIL_NOTE = {
    "knowledge": "知识库这块暂时查不了（服务异常）。",
    "order": "订单系统这块暂时查不了（服务异常）。",
    "analytics": "数据分析这块暂时跑不了（服务异常）。",
}
_FAIL_NOTE_DEFAULT = "有一块处理没跑起来（服务异常）。"

HANDOFF = "已经为您转接人工客服，请稍等。"
HANDOFF_ONLY = "抱歉，这次没能查到您要的信息，已经为您转接人工客服，请稍等。"

SYNTH_PROMPT = """你是电商客服的答复汇总器。下面有几个处理单元各自给出的结论，把它们合成**一段**给用户的话。

硬要求：
1. **只做整理**：合并重复、调整顺序、统一人称和语气。
2. **一个字的新事实都不许加。** 不许补常识、不许推测、不许把没说清的说圆。
   各单元没提到的东西，就当作不存在。
3. 结论之间**有冲突时如实指出冲突**，不要替用户挑一个。
4. 把字段名换成中文说法（order_id → 订单号，amount → 金额）。
5. 直接给答复正文。不要"根据以上信息"这类开头，不要 markdown 标题，不要罗列单元名字。

【用户的话】
{question}

【各单元结论】
{blocks}"""


def _split(results: list) -> dict:
    """
    把结果分成三堆。

    usable：真答上来了（ok 且没被标 insufficient，且有正文）
    thin  ：通了，但明确说"查不到 / 需要更多信息"。它的 answer 是**澄清话术**，
            不是知识内容——所以不能当正文拼进去，但用户仍需要看到
    failed：这个单元本身没跑起来
    """
    usable, thin, failed = [], [], []
    for r in results or []:
        if not r.get("ok"):
            failed.append(r)
        elif r.get("insufficient") or not str(r.get("answer") or "").strip():
            if not r.get("insufficient"):
                # ok=True、insufficient=False、却一个字没有——契约被违反了。
                # 当成"没答上来"处理，别让它静默消失（静默消失 = 用户看到空回答）。
                logger.warning("worker %s 返回了空结论", r.get("worker"))
            thin.append(r)
        else:
            usable.append(r)
    for bucket in (usable, thin, failed):
        bucket.sort(key=lambda r: r.get("worker", ""))
    return {"usable": usable, "thin": thin, "failed": failed}


def _blocks(rows: list) -> str:
    lines = []
    for r in rows:
        label = WORKER_LABELS.get(r.get("worker"), r.get("worker"))
        lines.append(f"- 来自「{label}」：{str(r.get('answer') or '').strip()}")
    return "\n".join(lines)


def _concat(rows: list) -> str:
    """
    不用 LLM 的拼接。**汇总的兜底，也是"只有一个结论"时的正路。**

    多段时带单元名分节：不是为了好看——两段来自不同系统的话直接连排，
    读起来像同一件事的延续，实际可能是两件不相干的事。
    """
    parts = []
    for r in rows:
        text = str(r.get("answer") or "").strip()
        if not text:
            continue
        if len(rows) > 1:
            label = WORKER_LABELS.get(r.get("worker"), r.get("worker"))
            parts.append(f"【{label}】{text}")
        else:
            parts.append(text)
    return "\n\n".join(parts)


def synthesize(question: str, usable: list, llm=None) -> dict:
    """
    合并多个结论。返回 {"answer", "source", "tokens"}，source ∈ {"llm","concat"}。

    只有一个结论时**直接返回，不调模型**（见模块开头那段）。
    """
    if not usable:
        return {"answer": "", "source": "concat", "tokens": dict(EMPTY)}
    if len(usable) == 1:
        return {
            "answer": str(usable[0].get("answer") or "").strip(),
            "source": "concat",
            "tokens": dict(EMPTY),
        }

    prompt = SYNTH_PROMPT.format(question=question, blocks=_blocks(usable))
    try:
        llm = llm or get_llm(SYNTH_TEMPERATURE, SYNTH_TIMEOUT)
        response = llm.invoke(
            [("system", "你是客服答复汇总器，不许引入任何新事实。"), ("human", prompt)]
        )
        text = str(response.content or "").strip()
        if text:
            return {"answer": text, "source": "llm", "tokens": usage_of(response)}
        logger.warning("汇总模型返回了空正文，改用直接拼接")
    except Exception as exc:  # noqa: BLE001 —— 汇总失败不该让整次请求失败
        logger.warning("汇总模型不可用，改用直接拼接：%s: %s", type(exc).__name__, exc)

    return {"answer": _concat(usable), "source": "concat", "tokens": dict(EMPTY)}


def _dedupe(values: list) -> list:
    """来源去重且保序。保序是有意的：先出现的往往是主来源。"""
    seen, out = set(), []
    for value in values or []:
        key = str(value)
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _join(parts: list) -> str:
    return "\n\n".join(p for p in parts if p and p.strip()).strip()


def aggregate(state: dict, llm=None) -> dict:
    """
    汇总节点。返回 state 的更新片段。

    llm：可注入（测试用假模型）。默认按配置取。
    """
    plan = state.get("plan") or []
    tried = state.get("tried") or []
    round_no = int(state.get("round") or 0)
    pending = str(state.get("pending_note") or "").strip()
    # 只看本轮的结果：results 跨轮累加（reducer 是 operator.add），
    # 不过滤的话第一轮的结论会在第二轮被再汇总一次——"同一段话说两遍"的来源。
    results = round_results(state.get("results") or [], round_no)

    buckets = _split(results)
    usable, thin, failed = buckets["usable"], buckets["thin"], buckets["failed"]
    alive = any(r.get("ok") for r in results)

    no_result = {
        "synth_source": "",
        "synth_tokens": dict(EMPTY),
    }

    # ① 有真结论 —— 合并后答复
    if usable:
        synth = synthesize(state.get("question") or "", usable, llm=llm)
        tail = [_FAIL_NOTE.get(r.get("worker"), _FAIL_NOTE_DEFAULT) for r in failed]
        tail += [str(r.get("answer") or "").strip() for r in thin if str(r.get("answer") or "").strip()]
        # 上一轮攒下的澄清话术不能丢：它可能正是用户需要的那句"请提供订单号"
        if pending and pending not in tail:
            tail.insert(0, pending)
        return {
            "decision": "answer",
            "answer": _join([synth["answer"], *tail]),
            "sources": _dedupe([s for r in usable for s in r.get("sources") or []]),
            "decision_reason": f"{len(usable)} 个单元给出结论，汇总方式 {synth['source']}",
            **no_result,
            "synth_source": synth["source"],
            "synth_tokens": synth["tokens"],
        }

    # ② 没有结论，看还能不能再试一轮
    untried = [n for n in worker_names() if n not in tried]
    # round 由 router 自增、从 1 起，所以 `round_no < MAX_ROUNDS` 就是
    # "还没跑满 MAX_ROUNDS 轮，可以再来一轮"。
    can_retry = (
        bool(plan)        # 这一轮本来就是空计划，重来一遍还是空
        and bool(untried)  # 没有没试过的单元，再路由就是原地打转
        and round_no < MAX_ROUNDS
        and alive         # 全挂了就别换人再挂一次
    )

    notes = [_FAIL_NOTE.get(r.get("worker"), _FAIL_NOTE_DEFAULT) for r in failed]
    notes += [str(r.get("answer") or "").strip() for r in thin if str(r.get("answer") or "").strip()]
    text = _join([pending, *notes])

    if can_retry:
        # 把已有的话带进下一轮：万一下一轮也没问出东西，至少手里还有这段澄清话术
        return {
            "decision": "re_route",
            "decision_reason": f"本轮无结论，改试未用过的单元：{'、'.join(untried)}",
            "pending_note": text,
            **no_result,
        }

    # ③ 手上有话可说（澄清话术 / 故障说明）—— 这就是答复
    if text:
        return {
            "decision": "answer",
            "answer": _join([text, HANDOFF]) if not alive else text,
            "sources": [],
            "decision_reason": "无结论，但把已知情况如实转达" if alive else "全部单元失败，说明并转人工",
            **no_result,
        }

    # ④ 什么都说不出 —— 唯一真正该转人工的情况
    return {
        "decision": "human",
        "answer": HANDOFF_ONLY,
        "sources": [],
        "decision_reason": "没有任何可转达的信息",
        **no_result,
    }
