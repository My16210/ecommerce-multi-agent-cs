"""
订单 worker：查状态 / 查物流 / 申请退货 / 查某客户的订单。

【为什么不用 Function Calling，而用"规则优先、LLM 兜底"】
方案初稿写的是"LLM 提取订单号 + 操作类型"。落地时改成了规则优先，理由是
**这两件事的难度不对称**：

  - 认订单号：一串 6 位以上纯数字，正则的准确率是 100%，LLM 是 95%——
    而且它会把"2025 年 9 月"里的数字也当成订单号。规则在这件事上更强。
  - 认操作类型：中文口语里"我这单咋还没动""我要退了这个"变化很多，
    关键词表一定会漏。这里是 LLM 真正划算的地方。

所以：**规则先认**（订单号 + 高频动作词），认不全才把整句话交给 LLM 判断。
好处有三——常见问法零模型调用、离线可测、LLM 挂了仍能处理大部分请求。

【"办不了"不是"坏了"】
订单不存在、状态不允许退货、重复申请，这些是**业务结论**，以 ok=True 返回，
只是 answer 里如实说明。只有"查库超时/连不上库"才是 ok=False。
把这两类混在一起，监控上就看不出"业务规则拦下了多少"和"系统坏了几次"。
"""
import re

from ecom.orchestrator.db.seed import seed
from ecom.orchestrator.tools.order_tools import (
    OrderNotFound,
    ReturnNotAllowed,
    create_return,
    describe_status,
    get_order_status,
    list_customer_orders,
    query_order,
)
from ecom.orchestrator.utils.json_utils import loads_or_none
from ecom.orchestrator.utils.llm import get_llm
from ecom.orchestrator.utils.logger import setup_logger
from ecom.orchestrator.utils.tokens import EMPTY, usage_of

logger = setup_logger(__name__)

# 订单号：5 位以上纯数字。太短的（如 2025）不当订单号，那多半是年份。
#
# 【为什么是 5 位】：DataCo 的 Order Id 实测是 1~5 位（最小 1，最大 77204，不补零）。
# 门槛只能取"最长的那个量级"——放宽到 4 位会让「2025 年买的」里的 2025 被判成订单号，
# 那比认不出短订单号更糟。所以 1~4 位的订单号在对话里**认不出来**，这是一条已知
# 边界（写在 README 的"明确不做"里），不是在正则里偷偷补零绕过去。
#
# 【为什么是 (?<!\w) 而不是原来的 (?<!\d)】：客户号是 C20755 这个形状。
# 如果只挡"前面是数字"，`C20755` 里那串 20755 会被当成订单号——一句话里同时
# 提到客户号和订单号时，认出来的会是客户号。挡"前面是字母或数字"才分得开。
#
# 【这个形状和 core/router.py 的 `_RULES` 第一条必须一致】，改一处就要改另一处。
_ORDER_ID = re.compile(r"(?<!\w)(\d{5,})(?!\d)")

# 【"我想退"和"能不能退"是两件事，写死的写操作不能靠一个"退货"关键词触发】
# 这是真踩过的：一句「订单 52644 到哪了，顺便问下你们的退货政策是什么」，
# 因为含"退货"两个字，被判成 create_return，**真的往库里插了一条退货申请**。
# 用户只是问政策，系统却替他办了件事——这是这个 worker 能犯的最严重的错。
#
# 所以 create_return 只在**第一人称诉求**（我要退/帮我退/申请退货…）时才触发；
# 光出现"退货"两个字不算。政策类问法（能不能退/怎么退/退货政策）由知识库答。
CREATE_RETURN_HINT = re.compile(
    r"我要退|我想退|帮我退|给我退|麻烦退|申请退|要退款|退货申请|退掉这|退了这|不想要了|退款吧|要求退"
)
# 只要句子里有这些、又**没有**第一人称诉求，就不认成"要办退货"——它们在问规则。
POLICY_HINT = re.compile(
    r"政策|规则|能不能退|能退吗|可以退吗|怎么退|如何退|多久|流程|条件|说明|规定"
)

# 反过来：明确说了"我要退"就算数，哪怕同一句里还夹了个政策问题——
# 「我要退货，顺便问下政策」是两条需求，用户点名的那个照办。

# 剩下的三个都是"读"（不改库），可以放心按词表扫。
ACTION_KEYWORDS = [
    ("tracking", re.compile(r"物流|快递|运单|包裹|到哪|到哪儿|到货|送到了|发货了没")),
    ("list_orders", re.compile(r"我的订单|我买过|我所有的|历史订单|一共买了")),
    ("status", re.compile(r"状态|怎么样了|处理到|进度|什么情况")),
]

ACTIONS = {"tracking", "status", "create_return", "list_orders"}

_LLM_PROMPT = """从一个电商客服对话里提取结构化意图。只输出 JSON，不要代码块：
{{"action": "tracking|status|create_return|list_orders|unknown", "order_id": "订单号或空", "reason": "退货原因或空", "reason_text": "一句话理由"}}

- tracking：问物流到哪了
- status：问订单状态
- create_return：要退货/退款
- list_orders：问某人名下所有订单
- unknown：看不出来

【用户的话】
{question}"""


def _rule_extract(question: str) -> dict:
    """
    规则提取。返回 {"action", "order_id", "reason"}，认不出 action 时 action=None。

    两条判断顺序不能换：先看"这是不是在问政策"（那就不是要办退货），
    再按动作词表扫。
    """
    match = _ORDER_ID.search(question)
    order_id = match.group(1) if match else ""

    action = None
    if CREATE_RETURN_HINT.search(question) and not policy_question(question):
        action = "create_return"
    else:
        for name, pattern in ACTION_KEYWORDS:
            if pattern.search(question):
                action = name
                break

    return {"action": action, "order_id": order_id, "reason": question.strip()[:200]}


def _first_person_demand(question: str) -> bool:
    """
    这句话里有没有明确的"我要办"的意思。

    有的话，哪怕同时提到"政策"，也按办退货处理——「我要退货，顺便问下政策」
    是两条需求，写的那条不该被政策词挡掉。挡掉的只是**纯问政策**。
    """
    return bool(re.search(r"我要|我想|帮我|给我|麻烦|申请|退掉|退了", question))


def policy_question(question: str) -> bool:
    """像在问规则、而不是在提要求。**写操作前必须过这一关。**"""
    return bool(POLICY_HINT.search(question)) and not _first_person_demand(question)


def _llm_extract(question: str, llm) -> dict:
    """LLM 兜底提取。解析不出来返回 action=None（调用方据此判定 insufficient）。"""
    response = llm.invoke([("system", "你是订单意图抽取器，只输出 JSON。"),
                           ("human", _LLM_PROMPT.format(question=question))])
    parsed = loads_or_none(response.content)
    if not isinstance(parsed, dict):
        return {"action": None, "order_id": "", "reason": "", "tokens": usage_of(response)}

    action = str(parsed.get("action") or "").strip().lower()
    if action not in ACTIONS:
        action = None
    order_id = str(parsed.get("order_id") or "").strip()
    if not _ORDER_ID.fullmatch(order_id):
        order_id = ""  # 模型给的订单号也要过同一道校验，不能因为"是模型说的"就信
    return {
        "action": action,
        "order_id": order_id,
        "reason": str(parsed.get("reason_text") or parsed.get("reason") or "").strip()[:200],
        "tokens": usage_of(response),
    }


def extract_intent(question: str, llm=None) -> dict:
    """
    规则优先、LLM 兜底。

    规则"够用"的判据：认出了 action，且这个 action 需要的参数齐了
    （tracking/status/create_return 都要订单号）。差一个就交给 LLM——
    但也可能 LLM 也认不出来，那就如实说认不出来，别硬猜一个订单号。
    """
    rule = _rule_extract(question)
    needs_id = rule["action"] in {"tracking", "status", "create_return"}
    if rule["action"] and (not needs_id or rule["order_id"]):
        return {**rule, "source": "rule", "tokens": dict(EMPTY)}

    try:
        llm = llm or get_llm(0)
        llm_result = _llm_extract(question, llm)
    except Exception as exc:  # noqa: BLE001 —— LLM 挂了不该拖垮规则能处理的那些
        logger.warning("LLM 意图提取不可用，退回规则结果：%s", exc)
        return {**rule, "source": "rule", "tokens": dict(EMPTY)}

    # LLM 若给出更完整的结论就用它；否则保留规则结果
    if llm_result["action"]:
        return {**llm_result, "source": "llm"}
    return {**rule, "source": "rule", "tokens": llm_result.get("tokens", dict(EMPTY))}


# ---------------------------------------------------------------------------
# 结果 → 人话
# ---------------------------------------------------------------------------
# 「这单没有物流」分两种，回答要说清是哪一种。DataCo 里没有运单的那批订单，
# 配送状态一律是 Shipping canceled（见 derive_seed.py 第 5 条）——所以判断依据
# 换成订单状态：取消/疑似欺诈的单本来就没发出去，其余是运单还没回传。
_NO_SHIPMENT_STATUS = {"CANCELED", "SUSPECTED_FRAUD"}


def describe_route(logistics: dict) -> str:
    """
    物流的起止地怎么写。**单独一个函数是因为界面也要用同一份说法**
    （`app.py` 的订单 Tab）。

    发货地为空时**不能**拼出「起点 → 目的地」：路线是两个地名之间的关系，
    只有一头就写不出路线，只能写"目的地"。这不是显示层的将就——是数据里
    真的没有起点，硬凑一个就是在回答里塞了一个不存在的事实。

    （公开的 DataCo 数据里只有 `Order City`，那是**目的地**；没有发货地。
     所以这条分支不是边角料，它是**常态**——每一笔订单都走这里。）
    """
    origin = logistics.get("from_city") or ""
    dest = logistics.get("to_city") or ""
    if origin and dest:
        return f"{origin} → {dest}"
    if dest:
        return f"目的地 {dest}"
    return "路线未记录"


def _fmt_tracking(data: dict, order_id: str) -> str:
    order, logistics = data["order"], data["logistics"]
    if not logistics:
        status = describe_status(order["status"])
        reason = "订单已取消，配送没有发生" if order["status"] in _NO_SHIPMENT_STATUS \
            else "运单信息还没回传"
        return f"订单 {order_id} 目前状态是「{status}」，没有物流信息——{reason}。"

    # carrier / tracking_no / from_city 允许为空，而且**不是"数据没灌全"**：
    # 公开数据集里根本没有这三项（承运商和运单号是快递公司的私有数据；连"发货地"
    # 都没有，只有目的地）。seed 如实留空，不编一个假的。
    #
    # 所以这里必须有分支：漏了它，回答里会出现「承运商 None，运单号 None」——
    # 比不显示更糟，它看起来像系统坏了，而不是"我们本来就没有这个信息"。
    carrier = logistics.get("carrier") or ""
    tracking_no = logistics.get("tracking_no") or ""
    if carrier and tracking_no:
        head = f"订单 {order_id} 的物流：{carrier}，运单号 {tracking_no}，"
    elif carrier:
        head = f"订单 {order_id} 的物流：{carrier}，"
    else:
        head = f"订单 {order_id} 的物流："

    # 起止地的写法抽到 describe_route 里了——界面要显示同一件事，两处各写一遍
    # 迟早在某一种组合上分叉（这里三种情况，手抄一遍就是六个分支要同步）。
    route = describe_route(logistics)
    return (
        f"{head}{route}，当前「{logistics['status']}」，最近更新 {logistics['updated_at']}。"
    )


def _fmt_status(order: dict) -> str:
    # 支付时间只在**有值**时才说。旧版这里是
    #   ... if order['paid_at'] else '（未支付）'
    # 换数据之后这个分支变成假话：DataCo 里没有支付时间这一列，`paid_at` 为空
    # 表示的是"数据集没有这个字段"，而不是"这笔订单没付钱"。一句话里的空值
    # 有两种含义，就不能用同一个说法兜——所以空着不说，比说错强。
    paid = f"，支付时间 {order['paid_at']}" if order.get("paid_at") else ""
    return (
        f"订单 {order['order_id']}：状态「{describe_status(order['status'])}」，"
        f"金额 ￥{order['amount']:.2f}，下单时间 {order['created_at']}{paid}。"
    )


def _fmt_return(result: dict) -> str:
    return (
        f"已为订单 {result['order_id']} 提交退货申请，申请单号 {result['return_id']}，"
        f"当前状态「{result['status']}」。{result['note']}"
    )


def _fmt_orders(rows: list, customer_id: str) -> str:
    if not rows:
        return f"客户 {customer_id} 名下没有查询到订单。"
    lines = [f"客户 {customer_id} 名下共 {len(rows)} 笔订单："]
    for row in rows:
        lines.append(
            f"  · {row['order_id']}　{row['status']}　￥{row['amount']:.2f}　{row['created_at']}"
        )
    return "\n".join(lines)


def _ask_before_writing(order_id: str, intent: dict, tokens: dict) -> dict:
    """写操作遇到"像是在问规则"时，先问清楚再动手。"""
    target = f"订单 {order_id}" if order_id else "某一笔订单"
    return {
        "answer": f"这句话像是在咨询退货规则，我不确定是否真要提交申请。"
                  f"如果要为{target}提交退货申请，回一句「确认申请退货」，我再来办；"
                  "只想了解规则的话，可以问「退货政策是什么」。",
        "insufficient": True,
        "data": {"intent": intent, "needs_confirmation": True},
        "tokens": tokens,
    }


def run_order(question: str, chat_history: list = None) -> dict:
    # 数据库可能还没灌数据（clone 下来第一次跑）。这里补一次幂等 seed，
    # 避免"演示第一步就报订单不存在"。
    seed()

    intent = extract_intent(question)
    action, order_id = intent["action"], intent["order_id"]
    tokens = intent.get("tokens") or dict(EMPTY)

    # 写操作的第二道闸（第一道在 _rule_extract）。上面那道只挡规则路径，
    # 模型也有可能把「你们的退货政策是什么」听成"要退货"——**写操作两层都要守**。
    # 挡下来的代价是多问一句；放过去的代价是替用户提了一个他没提的申请。
    #
    # 下面第二个分支把它顺带用在了"认不出动作、但明显在问退货"的情况上：
    # 「订单 52644 能不能退」用一句"没有明确的订单操作"回，等于把球扔回给用户；
    # 直接告诉他"你是想问规则，还是要办"，他下一句就能接上。
    if action == "create_return" and policy_question(question):
        return _ask_before_writing(order_id, intent, tokens)
    if not action:
        if policy_question(question) and "退" in question:
            return _ask_before_writing(order_id, intent, tokens)
        return {
            "answer": "这句话里没有明确的订单操作（查物流 / 查状态 / 申请退货）。"
                      "可以试试直接说订单号，例如「订单 52644 到哪了」。",
            "insufficient": True,
            "data": {"intent": intent},
            "tokens": tokens,
        }

    if action in {"tracking", "status", "create_return"} and not order_id:
        return {
            "answer": "请提供订单号，例如「订单 52644 到哪了」。",
            "insufficient": True,
            "data": {"intent": intent},
            "tokens": tokens,
        }

    try:
        if action == "tracking":
            data = query_order(order_id)
            answer = _fmt_tracking(data, order_id)
            sources = [order_id]
        elif action == "status":
            order = get_order_status(order_id)
            data = {"order": order}
            answer = _fmt_status(order)
            sources = [order_id]
        elif action == "list_orders":
            customer_id = _guess_customer(question)
            rows = list_customer_orders(customer_id)
            data = {"orders": rows, "customer_id": customer_id}
            answer = _fmt_orders(rows, customer_id)
            sources = [r["order_id"] for r in rows]
        else:  # create_return
            result = create_return(order_id, intent.get("reason") or question)
            data = {"return": result}
            answer = _fmt_return(result)
            sources = [order_id]
    except OrderNotFound as exc:
        # 业务结论，不是故障——ok 仍为 True
        return {
            "answer": f"{exc}。请核对订单号，或把订单号发我再查一次。",
            "insufficient": False,
            "data": {"intent": intent},
            "tokens": tokens,
        }
    except ReturnNotAllowed as exc:
        return {
            "answer": f"{exc}。",
            "insufficient": False,
            "data": {"intent": intent},
            "tokens": tokens,
        }

    return {
        "answer": answer,
        "sources": sources,
        "insufficient": False,
        "data": {"action": action, "intent": intent, **data},
        "tokens": tokens,
    }


def _guess_customer(question: str) -> str:
    """
    从话里猜客户号。**这是 demo 的将就**：真系统里客户身份来自登录态，
    绝不从自然语言里猜——那等于让任何人问出别人的订单。
    这里只在同一句话里显式写了 C9995 这种编号时才认，认不出就用演示账号。

    【为什么是 `(?<![A-Za-z0-9])` 而不是 `\\b`】
    中文和编号之间没有空格：「客户C9995的订单」是最自然的写法，而中文汉字在
    Python 的 `re` 里算 word 字符，`户` 和 `C` 之间**没有**词边界，`\\b` 会直接
    认不出来。改成"前面不是字母数字"就两种写法都能认。

    【为什么客户号带 C 前缀】
    DataCo 的订单号和客户号都是纯数字，不加前缀的话「订单 37570」和「客户 37570」
    是同一个字符串，规则层没法区分。前缀是派生时加的展示编码，不是原始字段——
    见 derive_seed.py 第 2 条。
    """
    match = re.search(r"(?<![A-Za-z0-9])(C\d{3,})(?!\d)", question, re.IGNORECASE)
    # 认不出就用演示账号。C9995 是演示订单 52644 的客户：默认值必须是**库里真实
    # 存在的**客户号，否则「我的订单」永远回一句"没有查询到订单"，看起来像坏了。
    return match.group(1).upper() if match else "C9995"
