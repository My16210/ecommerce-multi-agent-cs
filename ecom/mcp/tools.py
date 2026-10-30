"""
MCP 工具层：纯函数 + 一份稳定的 JSON 契约，**不依赖 MCP SDK**。

【为什么所有工具都返回同一个信封，而不是"成功就返回数据、失败就抛异常"】

调用方是另一个 Agent。抛异常在 MCP 里会变成一次协议错误，模型通常只看到
"tool failed" 四个字；而这里想让它看到的是**可读的、能自己纠正的话**——
"订单号应为 5 位以上纯数字"可以直接让它改参数重试，"查不到订单号 12345"
可以让它转告用户。所以所有失败都落在 `{"ok": false, "error": {...}}` 里，
错误码只有五个，够调用方分支，也不至于变成一堆没人用的枚举。

【为什么错误码里没有 "permission_denied"】

因为不注册的工具**根本不在工具列表里**（见 server.py）。调用一个不存在的
工具，MCP 自己会报 unknown tool；能走到工具函数里的，只有"功能开着但这次
调用不合法"的情况：`not_allowed` 表示业务规则不允许（状态不可退、演示只读、
重复申请），`unavailable` 表示依赖暂时坏了（知识库检索失败）。

【为什么不在这里重新实现 SQL / 检索】

工具层只做三件事：**校验输入、调用已有函数、裁剪输出**。业务规则（哪些状态
能退、重复申请怎么算）在 `orchestrator/tools/order_tools.py`；检索和 rerank
在 `knowledge/`。复制一份到这里，两份就会分叉——而分叉的那一天，客服机器人和
MCP 调用方会对同一笔订单给出不同答案。

【为什么输出要裁剪】

`orders` 表里有 `customer_name` / `customer_id`。对话链路需要它们（判断是不是
本人），MCP 的只读查单不需要——按单号查到的订单，返回客户姓名只是顺手把
一份 PII 交给了一个我们不知道会不会转存的客户端。**只给调用方办这件事真正
需要的最小字段。**
"""
import re
import time

from ecom import demo
from ecom.mcp import settings
from ecom.orchestrator.db.engine import Database, get_db
from ecom.orchestrator.tools.order_tools import (
    OrderNotFound,
    ReturnNotAllowed,
    get_order_status,
    get_returns,
    query_order,
)
from ecom.orchestrator.tools.order_tools import (
    create_return as _create_return,
)
from ecom.orchestrator.tools.order_tools import (
    list_customer_orders as _list_customer_orders,
)
from ecom.orchestrator.utils.ids import ORDER_ID_RE
from ecom.orchestrator.utils.logger import setup_logger

logger = setup_logger(__name__)

# 对外暴露的字段清单。**是白名单，不是黑名单**：以后表里加一列（比如手机号），
# 黑名单不会自动拦住它，白名单会。
_ORDER_FIELDS = ("order_id", "status", "amount", "region", "created_at", "paid_at")
_LOGISTICS_FIELDS = ("carrier", "tracking_no", "from_city", "to_city", "status", "updated_at")
_RETURN_FIELDS = ("return_id", "status", "reason", "created_at")

# 客户号：C + 4~8 位数字（DataCo 的 Customer Id 是 C20755 这个形状）。
_CUSTOMER_ID_RE = re.compile(r"C\d{4,8}")

# 检索结果里每条正文最多给多少字。1500 大约是一段 chunk 的长度；再长就不是
# "检索结果"而是"把整篇文档搬给调用方"了，它的上下文会被这一条占满。
_PREVIEW_CHARS = 1500

# 错误码。**五个，别再加**——加一个就要问"它和 not_allowed 有什么区别"。
INVALID_INPUT = "invalid_input"
NOT_FOUND = "not_found"
NOT_ALLOWED = "not_allowed"
UNAVAILABLE = "unavailable"
INTERNAL = "internal"


# ——————————————————————————————————————————————————————————————
# 信封
# ——————————————————————————————————————————————————————————————
def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def _ok(data: dict, started: float) -> dict:
    return {"ok": True, "data": data, "meta": {"elapsed_ms": _elapsed_ms(started)}}


def _err(code: str, message: str, started: float) -> dict:
    return {
        "ok": False,
        "error": {"code": code, "message": message},
        "meta": {"elapsed_ms": _elapsed_ms(started)},
    }


def _db(db: Database | None) -> Database:
    """测试注入用。生产路径永远走进程内那一个（每次仍开短连接）。"""
    return db if db is not None else get_db()


def _clean_order_id(order_id) -> str | None:
    """订单号的形状**照搬对话链路那一份正则**，不在这里写第二份。"""
    value = str(order_id or "").strip()
    return value if ORDER_ID_RE.fullmatch(value) else None


def _clean_int(value, default: int) -> int | None:
    """把 JSON 里的数字转成 int；布尔值不算数字（`True` 会被 int 成 1）。"""
    if value is None:
        return default
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _pick(row: dict, fields: tuple) -> dict:
    return {field: row.get(field) for field in fields}


# ——————————————————————————————————————————————————————————————
# 只读：订单 / 物流
# ——————————————————————————————————————————————————————————————
def get_order(order_id: str, db: Database | None = None) -> dict:
    """
    按订单号查订单与物流。**不返回客户姓名和客户号。**

    参数：
        order_id: 5 位以上纯数字，例如 "52644"

    返回 data：
        {"order": {order_id, status, amount, region, created_at, paid_at},
         "logistics": {carrier, tracking_no, from_city, to_city, status, updated_at} | null}

    已知边界：DataCo 的订单号有 1~4 位的，对话链路和这里都**认不出来**
    （见 `ecom/orchestrator/utils/ids.py`）。这是刻意的，不是漏了补零。
    """
    started = time.perf_counter()
    value = _clean_order_id(order_id)
    if value is None:
        return _err(INVALID_INPUT, "订单号应为 5 位以上纯数字，例如 52644。", started)

    try:
        payload = query_order(value, _db(db))
    except OrderNotFound:
        return _err(NOT_FOUND, f"查不到订单号 {value}。", started)
    except Exception as exc:  # noqa: BLE001 —— 对外只给错误码，细节进日志
        logger.error("MCP get_order 失败：%s: %s", type(exc).__name__, exc)
        return _err(INTERNAL, "订单库暂时不可用，请稍后重试。", started)

    logistics = payload.get("logistics")
    return _ok(
        {
            "order": _pick(payload["order"], _ORDER_FIELDS),
            "logistics": _pick(logistics, _LOGISTICS_FIELDS) if logistics else None,
        },
        started,
    )


def get_order_returns(order_id: str, db: Database | None = None) -> dict:
    """
    查某订单已有的退货申请（只读）。

    **先确认订单存在再查申请**：订单不存在时 `get_returns` 返回空列表，
    和"订单存在但没有申请"长得一模一样。不区分这两者，调用方会把
    "订单号打错了"当成"这笔订单没有退过"。

    返回 data：{"order_id": "...", "count": n, "returns": [...]}
    """
    started = time.perf_counter()
    value = _clean_order_id(order_id)
    if value is None:
        return _err(INVALID_INPUT, "订单号应为 5 位以上纯数字，例如 52644。", started)

    try:
        get_order_status(value, _db(db))
    except OrderNotFound:
        return _err(NOT_FOUND, f"查不到订单号 {value}。", started)
    except Exception as exc:  # noqa: BLE001
        logger.error("MCP get_order_returns 失败：%s: %s", type(exc).__name__, exc)
        return _err(INTERNAL, "订单库暂时不可用，请稍后重试。", started)

    try:
        rows = get_returns(value, _db(db))
    except Exception as exc:  # noqa: BLE001
        logger.error("MCP get_order_returns 读申请失败：%s: %s", type(exc).__name__, exc)
        return _err(INTERNAL, "订单库暂时不可用，请稍后重试。", started)

    returns = [_pick(row, _RETURN_FIELDS) for row in rows]
    return _ok({"order_id": value, "count": len(returns), "returns": returns}, started)


# ——————————————————————————————————————————————————————————————
# 只读：知识库检索（**不生成答案**）
# ——————————————————————————————————————————————————————————————
def search_knowledge(query: str, top_k: int = 3) -> dict:
    """
    在售后政策 / 制度 / 商品 FAQ 里做混合检索 + rerank，返回相关段落。

    **只检索，不生成。** 对话链路里 `run_knowledge` 会再调一次模型把段落写成
    一段话；MCP 的调用方自己就是模型，它要的是原文和出处，不是我们替它写好的
    二手结论。少一次模型调用也少一次成本。

    返回 data：{"query": "...", "count": n,
                "hits": [{chunk_id, source, score, preview}, ...]}

    降级：知识库检索失败返回 `unavailable`，**不返回空列表**——空列表的意思是
    "库里确实没有"，和"检索坏了"是两件事，调用方会据此决定要不要换句话问。
    """
    started = time.perf_counter()
    text = str(query or "").strip()
    if not text:
        return _err(INVALID_INPUT, "query 不能为空。", started)
    if len(text) > settings.MCP_MAX_QUERY_CHARS:
        return _err(
            INVALID_INPUT,
            f"query 最多 {settings.MCP_MAX_QUERY_CHARS} 字，当前 {len(text)} 字。",
            started,
        )

    k = _clean_int(top_k, 3)
    if k is None or not 1 <= k <= settings.MCP_MAX_TOP_K:
        return _err(INVALID_INPUT, f"top_k 应为 1~{settings.MCP_MAX_TOP_K} 的整数。", started)

    try:
        # 延迟 import：让 `ecom.mcp.tools` 在没装知识库依赖/没配 key 的环境里
        # 仍能被 import 和单测；真正调用时才付加载成本。
        from ecom.knowledge.retriever.reranker import rerank
        from ecom.knowledge.retriever.retriever import retrieve
        from ecom.knowledge.vector_store import get_vector_store

        hits = retrieve(text, get_vector_store())
        top = rerank(text, hits, top_k=k)
    except Exception as exc:  # noqa: BLE001 —— 检索链路的失败不往外抛
        logger.error("MCP search_knowledge 失败：%s: %s", type(exc).__name__, exc)
        return _err(UNAVAILABLE, "知识库检索暂时不可用，请稍后重试。", started)

    items = []
    for hit in top:
        try:
            score = round(float(hit.get("score") or 0.0), 6)
        except (TypeError, ValueError):
            score = 0.0
        items.append(
            {
                "chunk_id": hit.get("chunk_id"),
                "source": hit.get("source"),
                "score": score,
                "preview": str(hit.get("text") or "")[:_PREVIEW_CHARS],
            }
        )
    return _ok({"query": text, "count": len(items), "hits": items}, started)


# ——————————————————————————————————————————————————————————————
# 受控：按客户号查订单（默认不注册）
# ——————————————————————————————————————————————————————————————
def _clean_customer_id(customer_id) -> str | None:
    value = str(customer_id or "").strip().upper()
    return value if _CUSTOMER_ID_RE.fullmatch(value) else None


def list_customer_orders(customer_id: str, limit: int = 10, db: Database | None = None) -> dict:
    """
    按客户号查名下订单，最近的在前。**需要 `MCP_ENABLE_PII=1`。**

    这是 v1 里唯一一个"输入是 PII 入口"的工具：拿到客户号就能看到这个人买过
    什么。所以它默认关着，而且只返回订单号 / 状态 / 金额 / 下单时间——
    连区域都不给，因为"某客户 + 某区域"组合起来比单看订单更能定位到人。

    返回 data：{"customer_id": "...", "count": n, "orders": [...]}
    `count` 是**返回条数**，不是名下总数（总数不额外查，免得为一句摘要多打一次库）。
    """
    started = time.perf_counter()
    if not settings.MCP_ENABLE_PII:
        return _err(
            NOT_ALLOWED,
            "按客户号查订单未启用（需要服务端设置 MCP_ENABLE_PII=1）。",
            started,
        )

    value = _clean_customer_id(customer_id)
    if value is None:
        return _err(INVALID_INPUT, "客户号应为 C + 4~8 位数字，例如 C20755。", started)

    n = _clean_int(limit, 10)
    if n is None or not 1 <= n <= settings.MCP_MAX_CUSTOMER_ORDERS:
        return _err(
            INVALID_INPUT,
            f"limit 应为 1~{settings.MCP_MAX_CUSTOMER_ORDERS} 的整数。",
            started,
        )

    try:
        rows = _list_customer_orders(value, _db(db))
    except Exception as exc:  # noqa: BLE001
        logger.error("MCP list_customer_orders 失败：%s: %s", type(exc).__name__, exc)
        return _err(INTERNAL, "订单库暂时不可用，请稍后重试。", started)

    orders = [
        _pick(row, ("order_id", "status", "amount", "created_at")) for row in rows[:n]
    ]
    return _ok({"customer_id": value, "count": len(orders), "orders": orders}, started)


# ——————————————————————————————————————————————————————————————
# 写：提交退货申请（默认不注册，且要求 confirm）
# ——————————————————————————————————————————————————————————————
def create_return(
    order_id: str,
    reason: str,
    confirm: bool = False,
    db: Database | None = None,
) -> dict:
    """
    为订单提交一笔**待审核**的退货申请。需要 `MCP_ENABLE_WRITES=1` 且 `confirm=true`。

    两道闸缺一不可，因为它们防的不是同一件事：
        MCP_ENABLE_WRITES  服务端决定"这个能力开不开放"
        confirm=true       调用方决定"这一次是不是真的要办"

    只有第一道，模型可能把一句"能退吗"直接执行成申请；只有第二道，服务端
    等于把开关交给了外部客户端。**写操作必须两边都点头。**

    这里只会写入「待审核」——通过/拒绝是人走后台 `ecom/orchestrator/admin/`
    做的，工具层没有那个入口。和对话链路是同一条原则：模型不拥有最终决定权。

    返回 data：{"return_id", "order_id", "status", "created_at", "note"}
    失败码：`not_allowed`（未启用 / 演示只读 / 状态不可退 / 重复申请）、
    `not_found`（订单不存在）、`invalid_input`（参数形状不对 / 没确认）。
    """
    started = time.perf_counter()
    if not settings.MCP_ENABLE_WRITES:
        return _err(
            NOT_ALLOWED,
            "退货写操作未启用（需要服务端设置 MCP_ENABLE_WRITES=1）。",
            started,
        )
    # 演示模式是第二道服务端闸：即使误开了 MCP_ENABLE_WRITES，公网演示也不能落库。
    if not demo.writes_allowed():
        return _err(NOT_ALLOWED, "当前是只读演示环境，不能提交退货申请。", started)

    value = _clean_order_id(order_id)
    if value is None:
        return _err(INVALID_INPUT, "订单号应为 5 位以上纯数字，例如 52644。", started)

    if confirm is not True:
        return _err(
            INVALID_INPUT,
            "提交退货申请需要显式传 confirm=true；只传订单号和理由不会落库。",
            started,
        )

    text = str(reason or "").strip()
    if not text:
        return _err(INVALID_INPUT, "reason 不能为空。", started)
    if len(text) > 255:
        return _err(INVALID_INPUT, "reason 最多 255 字。", started)

    try:
        result = _create_return(value, text, _db(db))
    except OrderNotFound:
        return _err(NOT_FOUND, f"查不到订单号 {value}。", started)
    except ReturnNotAllowed as exc:
        return _err(NOT_ALLOWED, f"{exc}。", started)
    except Exception as exc:  # noqa: BLE001
        logger.error("MCP create_return 失败：%s: %s", type(exc).__name__, exc)
        return _err(INTERNAL, "订单库暂时不可用，请稍后重试。", started)

    return _ok(result, started)
