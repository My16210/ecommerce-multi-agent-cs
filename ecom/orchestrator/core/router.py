"""
Supervisor 的路由：这句话该交给谁处理。

【三层，从"聪明"到"兜底"，顺序就是优先级】

1. **LLM 分类**（主路径）。能理解改写、口语、多意图（"我上周买的那个耳机怎么还没到，
   顺便问下退货政策" → order + knowledge）。
2. **规则兜底**（LLM 调用失败，或输出解析不出来时）。正则+关键词，笨但不花钱、
   不联网、不会失败——它存在的意义就是"模型这条路整个不可用时还有得用"。
3. **默认最小计划**（规则也没命中）。交给 knowledge——它是唯一"什么都能聊两句"的 worker，
   答不出来会走 insufficient，比直接甩"我帮您转人工"体验好。

【为什么"置信度低于阈值就补 knowledge"】
路由错一次的代价不对称：漏调一个 worker = 少答一半（用户看得出来），
多调一个 = 多花一次调用（用户看不出来）。所以低置信时宁可多带一个安全的。
这是**故意的不对称**，不是没调好阈值。

【为什么排除 tried】
回环重路由时必须排除已经试过的 worker，否则"没答上来 → 再路由 → 还是它"
就是个死循环。所有权的约束在 supervisor 那层用 MAX_ROUNDS 兜底，这里负责不原地打转。

【为什么路由也要看上一轮】
「那能退吗」「那它的金额是多少」这类追问，**光看当轮那句字符串是判断不了的**——
它既没有订单号，也没有"订单"两个字。上一轮提到的那笔订单就是这句话的主语。
所以 route 收一个 chat_history（由 `_router_node` 从 state 传进来）：LLM 层拿到
一段**有界**的「上文」（见 `_context_block`），规则层走 `utils/entities` 的
`carried_order_id`——**那是"携带上一轮提到的实体"，不是往关键词表里加词**。

这条同时纠正了仓库里一处说反了的文档：`state.py` / README / 技术文档一直写着
"历史只在汇总那一步用一次"，但汇总节点（`core/aggregate.py`）**一次都没读过**
`chat_history`；真正读它的是 knowledge worker，和（从这一版起）router。
"""
import re
import time

from ecom import demo
from ecom.orchestrator.settings import (
    ROUTER_ATTEMPTS,
    ROUTER_BUDGET,
    ROUTER_TEMPERATURE,
    ROUTER_TIMEOUT,
)
from ecom.orchestrator.utils.entities import carried_order_id
from ecom.orchestrator.utils.ids import ORDER_ID_RE
from ecom.orchestrator.utils.json_utils import loads_or_none
from ecom.orchestrator.utils.llm import get_llm
from ecom.orchestrator.utils.logger import setup_logger
from ecom.orchestrator.utils.tokens import EMPTY, add, usage_of

logger = setup_logger(__name__)

# worker 名 → 给 LLM 看的职责说明。这份说明就是路由的"知识"，改它等于改路由行为，
# 所以放在最显眼的地方，和 WORKER_NAMES 挨着。
WORKER_DESCRIPTIONS = {
    "knowledge": "售后政策、退换货规则、发票物流说明、商品使用问题等**文字规定**类问题",
    "order": "具体某一笔订单：查状态、查物流到哪了、申请退货、查某个客户的订单",
    "analytics": "销售数据的**统计与分析**：销量排名、趋势、对比、退货率、总销售额",
}
WORKER_NAMES = list(WORKER_DESCRIPTIONS)

# 低于这个置信度就往计划里补一个 knowledge（见模块开头那段"不对称"）
CONFIDENCE_FLOOR = 0.6

ROUTER_PROMPT = """你是电商客服系统的调度中心。理解用户这句话，判断该交给下面哪些处理单元。

【可用的处理单元】
{workers}

【输出要求】
只输出一个 JSON 对象，不要 markdown 代码块，不要解释：
{{"workers": ["单元名", ...], "reason": "一句话理由", "confidence": 0.0~1.0}}

规则：
- 一句话里有**多个不相关**的需求就返回多个单元（如"订单到哪了 + 退货政策是什么"）。
- 只有一处需求就只返回一个。
- 拿不准时返回 knowledge。
- workers 只能从上面这几个名字里选，不要造新名字。
- 若用户这句话里有指代（"它""那笔订单""刚才那个"），指的是【上文】里的东西，
  按【上文】说的是哪件事来判断交给谁。

【上文】
{context}

【用户的话】
{question}"""

# 规则兜底用的关键词。宁可粗糙——它存在的意义是"LLM 挂了也不至于全瘫"，
# 不是"替代 LLM"。
_RULES = [
    # 订单号是强信号：一串数字，基本只可能是订单号。
    #
    # 【形状定义在 utils/ids.py，这里只引用】以前这里和 workers/order.py 各写了
    # 一份，换真实数据时放宽位数只改了一边，**分叉不报错**——路由认得出的订单号，
    # 抽取层认不出。那次事故的完整经过记在 utils/ids.py 的模块 docstring 里。
    (ORDER_ID_RE, "order"),
    (
        re.compile(
            r"订单|物流|快递|发货|签收|运单|到哪|到货|退回|退货申请|退款进度|包裹|谁买的"
        ),
        "order",
    ),
    (
        re.compile(r"销量|销售额|排名|趋势|同比|环比|哪个地区|最好卖|卖得最好|统计|占比|退货率"),
        "analytics",
    ),
    (
        re.compile(r"政策|规则|能不能退|可以退吗|发票|保修|售后|说明|怎么办|如何|流程|多久"),
        "knowledge",
    ),
]


# 「上文」给多少：轮数和每轮的字数都封顶。这是**成本参数**，不是随手取的——
# 路由每一轮都跑，喂进去的每一个字都要付钱，而且喂多了还有副作用（见 _context_block）。
_CONTEXT_TURNS = 2
_CONTEXT_CHARS = 160
# 更早的对话摘要（记忆，见 ecom/orchestrator/memory.py）最多带几条进路由提示词。
_MAX_MEMORY_LINES = 3
_ROLE_LABEL = {"user": "用户", "assistant": "客服", "memory": "更早的对话摘要"}


def _context_block(chat_history: list) -> str:
    """
    把最近几轮压成一段有界的「上文」，给 LLM 层做指代消解用。

    **只给最近 `_CONTEXT_TURNS` 轮，不给整段历史**，两个理由：
    1. 这一层是 `temperature=0` 的分类任务，而 assistant 的历史回答里 5 位以上的
       数字最多（销售额、总条数、退货率），整段塞进去会把路由往 order 带。
    2. 路由每轮都跑，整段历史每轮都要付一次 token。

    记忆（`role="memory"`）单独排在前面——它们本来就是"更早的对话"，位置上也该在上。
    """
    history = list(chat_history or [])
    memories = [t for t in history if str(t.get("role") or "") == "memory"]
    turns = [t for t in history if str(t.get("role") or "") != "memory"]

    lines = []
    for turn in memories[:_MAX_MEMORY_LINES] + turns[-_CONTEXT_TURNS * 2:]:
        content = str(turn.get("content") or "").strip()
        if not content:
            continue
        who = _ROLE_LABEL.get(str(turn.get("role") or ""), "上文")
        lines.append(f"{who}：{content[:_CONTEXT_CHARS]}")
    return "\n".join(lines) if lines else "（无）"


def _rule_route(question: str, carried_id: str = "") -> list:
    """
    规则兜底：命中的都返回（可能多意图），一个都没命中返回空。

    `carried_id` 非空 = "上一轮提过的那笔订单，而且这一轮用了指代词没再说订单号"
    （判定在 `utils/entities.carried_order_id`，这里不重复判）。这时补一个 order。

    **这不是往 `_RULES` 里加词**：加词是"看到某个字就路由到某个 worker"，
    携带实体是"接着上一轮说"。两者形状不同，所以走独立入口，不混进关键词表
    （README 的「已知限制」点名了不要用加词的办法做这件事）。
    """
    plan = []
    for pattern, worker in _RULES:
        if pattern.search(question) and worker not in plan:
            plan.append(worker)
    if carried_id and "order" not in plan:
        plan.append("order")
    return plan


def _cold_plan() -> list:
    """彻底没辙时的最小计划。knowledge 是唯一"什么都能聊两句"的。"""
    return ["knowledge"]


def _sanitize(raw: dict) -> dict:
    """
    把 LLM 的返回收拾成可信的形状。

    认不出的 worker 名**直接丢**而不是猜——猜错了等于调错单元，比少调一个更贵。
    置信度缺失或非数按 0 处理（宁可当低置信去补 knowledge，也不要凭空相信）。
    """
    workers = raw.get("workers")
    plan = []
    if isinstance(workers, list):
        for name in workers:
            name = str(name).strip().lower()
            if name in WORKER_NAMES and name not in plan:
                plan.append(name)

    try:
        confidence = float(raw.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    # clamp：模型偶尔会给 95 这种"百分比"写法
    if confidence > 1.0:
        confidence = confidence / 100.0 if confidence <= 100 else 1.0
    confidence = min(max(confidence, 0.0), 1.0)

    reason = str(raw.get("reason") or "").strip()[:200]
    return {"plan": plan, "confidence": confidence, "reason": reason}


def parse_route(content: str) -> dict:
    """解析路由输出。解析不出来返回 plan=[]（调用方据此走规则兜底）。"""
    data = loads_or_none(content)
    if not isinstance(data, dict):
        return {"plan": [], "confidence": 0.0, "reason": "", "parse_error": True}
    out = _sanitize(data)
    out["parse_error"] = not out["plan"]
    return out


def route(question: str, tried: list = None, llm=None, chat_history: list = None) -> dict:
    """
    决定这一轮调用哪些 worker。

    返回 {"plan", "reason", "confidence", "source", "tokens"}。
    source ∈ {"llm", "rule", "fallback"}——一定要记录它是谁给的结论，
    因为"规则兜底的比例"是判断路由健康度的关键指标：它一涨，说明 LLM 那边在坏。

    chat_history：上一轮的对话，指代消解要用（见模块开头）。不传就是单轮。
    llm：可注入（测试用假模型）。默认按配置取。
    """
    tried = list(tried or [])
    # 演示模式下连提示词里都不列数据分析：让模型去选一个已经关掉的单元，
    # 只会多一次解析失败/一次白跑。规则层和最终计划下面还会再过滤一遍——
    # 提示词是建议，不是保证。
    workers_block = "\n".join(
        f"- {n}：{d}" for n, d in WORKER_DESCRIPTIONS.items() if demo.worker_allowed(n)
    )
    prompt = ROUTER_PROMPT.format(
        workers=workers_block, context=_context_block(chat_history), question=question
    )

    total = dict(EMPTY)
    parsed = None
    # 重试的理由：temperature=0 **不等于**确定性（MoE 路由、批处理都会让它抖），
    # 所以偶尔一次解析失败是常态，不是缺陷。把偶发当缺陷处理（重试）比调低标准强。
    #
    # 整段包在 try 里，是为了让"模型这条路彻底不可用"（没配 key、超时、网络断）
    # 也走同一个出口——退回规则。**这一步不能漏**：三层设计里"规则兜底"存在的
    # 全部意义就是"LLM 挂了还能路由"，如果异常直接冒出去，第三层从来没被触发过。
    #
    # 除了 try，还有一道**总预算**：重试次数本身不封顶时间。单次超时 8 秒 × 3 次
    # 本来也还好，但之前单次超时取的是默认 60 秒，于是"模型这条路不通"要拖
    # 180 秒才落到规则上——用户已经走了。次数和总时长是两把锁，都要有。
    started = time.perf_counter()
    try:
        llm = llm or get_llm(ROUTER_TEMPERATURE, ROUTER_TIMEOUT)
        for _ in range(ROUTER_ATTEMPTS):
            response = llm.invoke([("system", "你是调度中心，只输出 JSON。"), ("human", prompt)])
            total = add(total, usage_of(response))
            parsed = parse_route(response.content)
            if not parsed["parse_error"]:
                break
            if time.perf_counter() - started > ROUTER_BUDGET:
                logger.warning("路由重试已超过 %ss 预算，不再重试", ROUTER_BUDGET)
                break
    except Exception as exc:  # noqa: BLE001 —— 见上：这里是"模型不可用"的统一出口
        logger.warning("路由模型不可用，退回关键词规则：%s: %s", type(exc).__name__, exc)
        parsed = {"plan": [], "confidence": 0.0, "reason": "", "parse_error": True}

    source = "llm"
    if parsed["parse_error"]:
        # 走到这里说明 LLM 这路没给出可用结论，退回规则。
        source = "rule"
        plan = _rule_route(question, carried_order_id(question, chat_history))
        reason = "LLM 路由输出不可用，退回关键词规则"
        confidence = 0.0
        if not plan:
            source = "fallback"
            plan = _cold_plan()
            reason = "规则也没命中，默认交给知识库（答不出来会走 insufficient）"
    else:
        plan = parsed["plan"]
        reason = parsed["reason"]
        confidence = parsed["confidence"]
        if confidence < CONFIDENCE_FLOOR:
            for name in _cold_plan():
                if name not in plan and name not in tried:
                    plan.append(name)
            reason = f"{reason}（置信度 {confidence:.2f} 偏低，补一个 knowledge 兜底）"

    # 演示模式：把禁用的 worker 从计划里摘掉（LLM 和规则两条路都可能选到它）。
    raw_plan = list(plan)
    plan = demo.filter_workers(plan)
    if plan != raw_plan:
        reason = f"{reason}（演示环境只读：已停用数据分析）"

    # 排除已试过的，防止回环原地打转
    plan = [n for n in plan if n not in tried]

    # 摘掉之后计划可能空了（例如用户只问了销售数据）。这时退回知识库：
    # 它答不上来会走 insufficient，比直接甩一句"转人工"好，也不会执行代码。
    if demo.enabled() and not plan and "knowledge" not in tried:
        plan = _cold_plan()
        reason = f"{reason}（原计划在演示模式下不可用，改走知识库）"

    logger.info("路由：%s → %s（%s，置信度 %.2f）", question[:40], plan, source, confidence)
    return {
        "plan": plan,
        "reason": reason,
        "confidence": confidence,
        "source": source,
        "tokens": total,
    }
