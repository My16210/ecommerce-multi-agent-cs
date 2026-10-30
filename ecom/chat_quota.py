"""
已登录用户的问答配额：每人每分钟请求数 + 每人每天 token 预算。

【和 `ecom/demo.py` 的限流/预算不是同一件事，不要合并】

    demo   面向**没有身份**的公网只读演示。键是客户端 IP，预算是全站一份。
           它回答的是"陌生人能看多久、整站一天最多花多少"。
    这里   面向**已经登录**的 API。键是 principal id，预算是每人一份，
           外加一道全部署的总预算。它回答的是"这个账号 / 这台部署今天
           最多能花多少"。

两者可以同时生效（例如身份层和演示模式都打开），各挡各的。合并成一套的
结果是：公网演示的匿名访客和登录客户的额度被混在同一个桶里，谁都能把
另一个人挤掉。

【为什么限流在进程内，预算在磁盘】

限流只需要挡住单个账号的突发请求，不需要跨副本精确；多副本时每人实际能
打 N 倍，这是已知边界，真要精确得把计数器挪到 Redis。预算不行——它按实际
用量记账，重启和多副本下都必须算数，所以落 sqlite，按 (天, principal) 记账。

【预算读不出来时为什么拒绝，而不是放行】

和 demo 同一条规矩：**成本闸失效时倒向不花钱那一边**。账本读不出来时抛
`LedgerUnavailable`，API 把它翻成 503。一个因为磁盘问题停下来的服务会被
立刻发现；反过来"读不出来就继续花"会在没人知道的情况下把额度烧穿。
"""
import os
import sqlite3
import threading
import time
from contextlib import closing
from datetime import date

from ecom.orchestrator.utils.logger import setup_logger
from ecom.settings import DATA_DIR

logger = setup_logger(__name__)

# 每个已登录账号每分钟最多几次提问。默认 30：远高于一个人正常问答的节奏，
# 又低于"拿脚本刷"的量级。<= 0 表示关闭这一道闸。
CHAT_RATE_LIMIT_PER_MINUTE = int(os.getenv("CHAT_RATE_LIMIT_PER_MINUTE", "30"))

# 每个账号每天最多花多少 token（in+out）。<= 0 表示不限额。
# 默认 20 万，和演示预算同量级：按 qwen-plus 公开价格是几块钱。
CHAT_DAILY_TOKEN_BUDGET = int(os.getenv("CHAT_DAILY_TOKEN_BUDGET", "200000"))

# 整个部署每天最多花多少 token（所有账号加起来）。<= 0 表示不限额。
# 单人预算挡的是"一个账号失控"，挡不住"很多账号一起花"；这一道闸是
# 部署者的总成本上限。默认 100 万 ≈ 单人的 5 倍，正常小规模使用不会碰到，
# 但脚本批量注册账号时会在超支前停下来。
CHAT_GLOBAL_DAILY_TOKEN_BUDGET = int(
    os.getenv("CHAT_GLOBAL_DAILY_TOKEN_BUDGET", "1000000")
)

# 单独一个账本，不和演示预算共用一个文件：两者的键和保留策略不同，
# 混在一起会让"清掉演示预算"顺手清掉登录用户的额度。
CHAT_USAGE_DB_PATH = os.getenv(
    "CHAT_USAGE_DB_PATH", str(DATA_DIR / "orchestrator" / "chat_usage.sqlite")
)

_RATE_WINDOW_SECONDS = 60.0
_rate_hits: dict[str, list[float]] = {}
_rate_lock = threading.Lock()
_budget_lock = threading.Lock()


class LedgerUnavailable(RuntimeError):
    """用量账本读不出来。调用方应当拒绝请求，而不是当成"没用过"。"""


def _principal_key(principal_id: str) -> str:
    return str(principal_id or "").strip()


# ——————————————————————————————————————————————————————————————
# 限流：进程内滑动窗口
# ——————————————————————————————————————————————————————————————
def allow_request(principal_id: str, limit: int | None = None) -> bool:
    """
    这个 principal 在最近 60 秒内还能不能再问一次。

    principal id 为空（身份层没开 / 匿名）时直接放行：这条配额只管
    "有账号的人"，匿名流量由 demo 那套按 IP 的闸负责。
    """
    key = _principal_key(principal_id)
    if not key:
        return True
    limit = CHAT_RATE_LIMIT_PER_MINUTE if limit is None else int(limit)
    if limit <= 0:
        return True

    now = time.monotonic()
    with _rate_lock:
        recent = [t for t in _rate_hits.get(key, []) if now - t < _RATE_WINDOW_SECONDS]
        if len(recent) >= limit:
            _rate_hits[key] = recent
            return False
        recent.append(now)
        _rate_hits[key] = recent
        _prune_rate_hits(now)
    return True


def _prune_rate_hits(now: float) -> None:
    """偶尔清掉整桶都过期的 key，避免账号多了以后字典一直长。"""
    if len(_rate_hits) < 512:
        return
    for key in [
        k for k, hits in _rate_hits.items() if not hits or now - hits[-1] >= _RATE_WINDOW_SECONDS
    ]:
        _rate_hits.pop(key, None)


def reset_rate_limits() -> None:
    """测试用。运行时不调它——清空计数等于把限流短时关掉。"""
    with _rate_lock:
        _rate_hits.clear()


# ——————————————————————————————————————————————————————————————
# 每日 token 预算：磁盘账本，按 (天, principal) 记账
# ——————————————————————————————————————————————————————————————
def _connect() -> sqlite3.Connection:
    path = CHAT_USAGE_DB_PATH
    parent = os.path.dirname(str(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS principal_usage ("
        "day TEXT NOT NULL, principal_id TEXT NOT NULL, tokens INTEGER NOT NULL, "
        "PRIMARY KEY (day, principal_id))"
    )
    return conn


def _today() -> str:
    return date.today().isoformat()


def tokens_used_today(principal_id: str) -> int:
    """今天这个账号用了多少 token。账本读不出来时抛 `LedgerUnavailable`。"""
    key = _principal_key(principal_id)
    if not key:
        return 0
    try:
        with closing(_connect()) as conn:
            row = conn.execute(
                "SELECT tokens FROM principal_usage WHERE day = ? AND principal_id = ?",
                (_today(), key),
            ).fetchone()
        return int(row[0]) if row else 0
    except Exception as exc:  # noqa: BLE001 —— 账本坏了要让调用方看见
        logger.error("读取账号用量账本失败：%s: %s", type(exc).__name__, exc)
        raise LedgerUnavailable("账号用量账本不可用") from exc


def budget_exhausted(principal_id: str) -> bool:
    """
    这个账号今天还能不能继续问。

    没设上限 → False；账本读不出来 → 抛 `LedgerUnavailable`，由 API 转成
    拒绝。**不要在这里把异常吞成 True/False**——那会让"账本坏了"和
    "额度用完了"变成同一句话，排查时看不出区别。
    """
    key = _principal_key(principal_id)
    if not key or CHAT_DAILY_TOKEN_BUDGET <= 0:
        return False
    return tokens_used_today(key) >= CHAT_DAILY_TOKEN_BUDGET


def global_tokens_used_today() -> int:
    """今天所有账号一共用了多少 token。账本读不出来时抛 `LedgerUnavailable`。"""
    try:
        with closing(_connect()) as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM principal_usage WHERE day = ?",
                (_today(),),
            ).fetchone()
        return int(row[0]) if row else 0
    except Exception as exc:  # noqa: BLE001 —— 账本坏了要让调用方看见
        logger.error("读取整体用量账本失败：%s: %s", type(exc).__name__, exc)
        raise LedgerUnavailable("整体用量账本不可用") from exc


def global_budget_exhausted() -> bool:
    """
    这台部署今天还能不能继续接待已登录用户。

    单人预算和总预算各查各的：总预算没设（<= 0）时直接放行，不读账本——
    关掉的功能不该因为磁盘问题把服务拖垮。
    """
    if CHAT_GLOBAL_DAILY_TOKEN_BUDGET <= 0:
        return False
    return global_tokens_used_today() >= CHAT_GLOBAL_DAILY_TOKEN_BUDGET


def add_tokens(principal_id: str, tokens: dict) -> int:
    """
    给这个账号记一笔用量，返回记了多少。`tokens` 是 `{"in": .., "out": ..}`。

    记账失败只打日志、不抛：它发生在一次请求的收尾，抛出去会让用户看到
    "答完了但报错"。少记一笔由预算余量吸收；这和 demo 的记账策略一致。
    """
    key = _principal_key(principal_id)
    if not key:
        return 0
    amount = 0
    if isinstance(tokens, dict):
        try:
            amount = int(tokens.get("in") or 0) + int(tokens.get("out") or 0)
        except (TypeError, ValueError):
            amount = 0
    if amount <= 0:
        return 0
    try:
        with _budget_lock, closing(_connect()) as conn:
            conn.execute(
                "INSERT INTO principal_usage (day, principal_id, tokens) VALUES (?, ?, ?) "
                "ON CONFLICT(day, principal_id) "
                "DO UPDATE SET tokens = tokens + excluded.tokens",
                (_today(), key, amount),
            )
            conn.commit()
    except Exception as exc:  # noqa: BLE001 —— 见 docstring
        logger.error(
            "写入账号用量账本失败（少记 %s token）：%s: %s",
            amount,
            type(exc).__name__,
            exc,
        )
        return 0
    return amount
