"""
公网演示模式的四条闸：邀请码、限流、每日 token 预算、只读策略。

【为什么这些不写在 app.py / api.py 里】
两个入口共用同一套规则，各写一份的结果是**两份会分叉的规则**：界面封了写操作、
API 忘了封，或者界面的限流是 10 次、API 是 12 次。而这两种分叉都不会报错，
只会在某一天被人从没设防的那个入口钻进来。所以规则只有这一处，两个入口
都调它；`workers/order.py` 和 `core/router.py` 也调它——**只读不能只靠界面
藏按钮**，模型完全可能绕到写操作那条分支上。

【这四条闸各自挡住什么，挡不住什么——别把边界说大】

    邀请码   挡住"随手扫到端口的人"。挡不住把码转发出去的人，也挡不住
             认真扫邀请码的人（它没有账号、没有审计、没有失效机制）。
    限流     挡住单个客户端刷屏。挡不住换 IP / 多开会话——所以它只是第一道。
    token 预算 挡住"一天烧掉一大笔钱"。**这是唯一真正兜住成本的闸**，因为
             它按实际用量记账，绕开限流也绕不开它。
    只读     挡住退货申请、数据看板里的代码执行、知识库写入。它是**策略闸**，
             不是权限系统：没有用户、没有角色、没有审计。

一句话：这是**能安全地挂在公网上给人看**的最低配置，不是一套生产级权限系统。
真要上线到"能替真实用户办事"，得先有身份和审计——那不在这一层解决。

【为什么预算账本落在磁盘，而不是进程内变量】
Streamlit 的脚本每交互一次都会重跑，API 又会被丢进线程池/多进程。进程内计数
的结果是"每个进程一份预算"，实际花掉的是 N 倍。所以用量落 sqlite，按天记账，
重启也算数。代价是多一个文件、多一把 sqlite 锁——这点开销和"预算真的生效"
比是值得的。

【预算读不出来时为什么是"拒绝"而不是"放行"】
`budget_exhausted()` 在账本打不开时返回 True（并打日志）。一个演示环境因为
磁盘问题停掉，会被立刻发现；反过来"读不出来就继续花"，会在没人知道的情况下
把预算烧穿。**成本闸失效时倒向不花钱那一边**。
"""
import secrets
import sqlite3
import threading
import time
from contextlib import closing
from datetime import date

from ecom.orchestrator.settings import (
    DEMO_DAILY_TOKEN_BUDGET,
    DEMO_INVITE_CODE,
    DEMO_MODE,
    DEMO_RATE_LIMIT_PER_MINUTE,
    DEMO_USAGE_DB_PATH,
)
from ecom.orchestrator.utils.logger import setup_logger

logger = setup_logger(__name__)

# 演示模式下不参与路由的 worker。数据分析会**执行模型生成的代码**，是公网
# 演示里风险最高的一块；它也不是"客服"这个演示的主线。关掉它，同时把
# `analytics` 从路由提示词和回环的未试集合里一起摘掉——只从计划里删名字不够，
# 因为 aggregate 重路由时会从全量 worker 里挑。
DISABLED_WORKERS = frozenset({"analytics"})

_RATE_WINDOW_SECONDS = 60.0
_rate_hits: dict[str, list[float]] = {}
_rate_lock = threading.Lock()

_BUDGET_LOCK = threading.Lock()


def enabled() -> bool:
    """演示模式开没开。每次现读模块变量，测试可以 monkeypatch。"""
    return bool(DEMO_MODE)


def invite_configured() -> bool:
    """邀请码有没有配。界面用它把"没配"和"配错了"分开说。"""
    return bool(DEMO_INVITE_CODE)


def check_invite(code: str) -> bool:
    """
    邀请码对不对。**演示模式打开但没配邀请码时一律拒绝**（fail closed）。

    比较用 `compare_digest`，理由和 api.py 的 API_KEY 一样：普通 `==` 会在
    第一个不同字节返回，时间差可以被利用。两边都先 encode 成 bytes——
    `compare_digest` 碰上中文 str 会直接抛 TypeError，而中文邀请码是很自然
    会被设出来的东西。
    """
    if not enabled():
        return True
    if not DEMO_INVITE_CODE:
        logger.warning("DEMO_MODE 已打开，但没有配置 DEMO_INVITE_CODE：拒绝所有请求")
        return False
    return secrets.compare_digest(
        str(code or "").encode("utf-8"), DEMO_INVITE_CODE.encode("utf-8")
    )


# ——————————————————————————————————————————————————————————————
# 只读策略
# ——————————————————————————————————————————————————————————————
def writes_allowed() -> bool:
    """能不能落库（目前只有 create_return）。演示模式下一律不能。"""
    return not enabled()


def worker_allowed(name: str) -> bool:
    return not (enabled() and name in DISABLED_WORKERS)


def filter_workers(names) -> list:
    """把演示模式下禁用的 worker 从任意名单里摘掉，保序。"""
    return [name for name in names or [] if worker_allowed(name)]


# ——————————————————————————————————————————————————————————————
# 限流：滑动窗口，进程内
# ——————————————————————————————————————————————————————————————
def allow_request(key: str, limit: int | None = None) -> bool:
    """
    `key` 在最近 60 秒内还能不能再发一次。

    进程内计数是**故意的**：它只需要挡住单个客户端的突发请求，不需要跨副本
    精确。跨副本仍然有效的闸是磁盘上的 token 预算。两者一个有状态、一个无状态，
    别把它们的功能说反了。
    """
    if not enabled():
        return True
    limit = DEMO_RATE_LIMIT_PER_MINUTE if limit is None else int(limit)
    if limit <= 0:
        return True
    key = str(key or "anonymous")
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
    """偶尔清掉整桶都过期的 key，避免被大量不同 IP 把字典撑大。"""
    if len(_rate_hits) < 512:
        return
    for key in [k for k, hits in _rate_hits.items() if not hits or now - hits[-1] >= _RATE_WINDOW_SECONDS]:
        _rate_hits.pop(key, None)


def reset_rate_limits() -> None:
    """测试用。运行时不调它——清空计数等于把限流短时关掉。"""
    with _rate_lock:
        _rate_hits.clear()


# ——————————————————————————————————————————————————————————————
# 每日 token 预算：磁盘账本
# ——————————————————————————————————————————————————————————————
def _connect() -> sqlite3.Connection:
    path = DEMO_USAGE_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5)
    # 账本是**多个进程/线程共写**的：API 的请求线程、Streamlit 的脚本重跑、
    # CLI 的清理命令都可能同时开连接。WAL 让读不挡写、写不挡读，busy_timeout
    # 把短暂的锁冲突变成等待而不是立刻 "database is locked"（那会变成少记账）。
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS usage (day TEXT PRIMARY KEY, tokens INTEGER NOT NULL)"
    )
    return conn


def _today() -> str:
    return date.today().isoformat()


def tokens_used_today() -> int:
    try:
        with closing(_connect()) as conn:
            row = conn.execute(
                "SELECT tokens FROM usage WHERE day = ?", (_today(),)
            ).fetchone()
        return int(row[0]) if row else 0
    except Exception as exc:  # noqa: BLE001 —— 账本坏了要让调用方看见后果
        logger.error("读取演示预算账本失败：%s: %s", type(exc).__name__, exc)
        raise


def budget_remaining() -> int | None:
    """今天还剩多少 token；不限额时返回 None。读不出来时抛异常（调用方转成拒绝）。"""
    if not enabled() or DEMO_DAILY_TOKEN_BUDGET <= 0:
        return None
    return max(DEMO_DAILY_TOKEN_BUDGET - tokens_used_today(), 0)


def budget_exhausted() -> bool:
    """
    今天还能不能继续问。**演示模式没开 / 没设上限 → False；账本读不出来 → True。**

    读不出来时拒绝，理由见模块开头：成本闸失效要倒向不花钱那一边。
    """
    if not enabled() or DEMO_DAILY_TOKEN_BUDGET <= 0:
        return False
    try:
        return tokens_used_today() >= DEMO_DAILY_TOKEN_BUDGET
    except Exception:  # noqa: BLE001 —— 已经在 tokens_used_today 里记过日志
        return True


def add_tokens(tokens: dict) -> int:
    """
    记一笔用量，返回记了多少。`tokens` 是 `{"in": .., "out": ..}`。

    模型或图片识别失败、用量拿不到时传进来的是空 dict / 0——那就不写账。
    记账失败只打日志、不抛：它发生在一次请求的收尾，抛出去会让用户看到
    "答完了但报错"。少记一笔的代价由预算的余量吸收。
    """
    if not enabled():
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
        with _BUDGET_LOCK, closing(_connect()) as conn:
            conn.execute(
                "INSERT INTO usage (day, tokens) VALUES (?, ?) "
                "ON CONFLICT(day) DO UPDATE SET tokens = tokens + excluded.tokens",
                (_today(), amount),
            )
            conn.commit()
    except Exception as exc:  # noqa: BLE001 —— 见 docstring
        logger.error("写入演示预算账本失败（少记 %s token）：%s: %s", amount, type(exc).__name__, exc)
        return 0
    return amount
