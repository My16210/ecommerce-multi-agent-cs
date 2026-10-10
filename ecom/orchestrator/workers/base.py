"""
Worker 契约与调度。

【一个 worker 就是"名字 → 函数"，函数只做一件事：回答问题的一小块】
所有 worker 返回同一种结构（见 state.WorkerResult），supervisor 只认这个契约，
不认具体是知识库还是订单。这样加/换 worker 不用动 supervisor——
这是"职责隔离"在代码上的落点，不是文档里的一句话。

【为什么失败一定要在这里被吃掉】
worker 是"会失败的外部世界"：知识库的 embedding 服务可能挂，数据分析的子进程
可能超时，订单库文件可能被删。这些失败必须被翻译成一条 ok=False 的结果，
**不能让异常冒到 supervisor**——否则一个 worker 挂掉等于整次请求挂掉，
而正确答案是"用其余 worker 的结果先答，并如实说明哪一块没拿到"。

【超时的两层含义，必须说清楚】
- 数据分析走子进程，超时是**真杀**（subprocess timeout），进程连同它的循环一起没了。
- 知识库/订单是进程内调用，这里的超时**只是"不再等它"**——Python 杀不掉一个
  正在跑的线程。被放弃的线程会自己跑完然后被回收。它保护的是 supervisor 的响应
  时间（用户不会被一个卡死的调用拖住），**不是**资源占用。
  真要连进程内 worker 都能强杀，得把它们也挪进子进程或进程池——那是另一个取舍，
  当前规模下不值得。这一条别在简历里说成"所有 worker 都能超时中断"。
"""
import threading
import time
from collections.abc import Callable

from ecom.orchestrator.settings import WORKER_TIMEOUT, WORKER_TIMEOUT_DEFAULT
from ecom.orchestrator.utils.logger import setup_logger
from ecom.orchestrator.utils.tokens import EMPTY

logger = setup_logger(__name__)


def ok_result(
    worker: str,
    answer: str = "",
    data: dict = None,
    sources: list = None,
    insufficient: bool = False,
    tokens: dict = None,
    elapsed_ms: int = 0,
) -> dict:
    return {
        "worker": worker,
        "ok": True,
        "answer": answer,
        "data": data or {},
        "sources": list(sources or []),
        "insufficient": insufficient,
        "error": "",
        "elapsed_ms": elapsed_ms,
        "tokens": dict(tokens or EMPTY),
    }


def error_result(worker: str, error: str, elapsed_ms: int = 0, tokens: dict = None) -> dict:
    """注意 insufficient=False——坏了不是"看过了没有"，两者对 aggregate 的意义不同。"""
    return {
        "worker": worker,
        "ok": False,
        "answer": "",
        "data": {},
        "sources": [],
        "insufficient": False,
        "error": str(error)[:500],
        "elapsed_ms": elapsed_ms,
        "tokens": dict(tokens or EMPTY),
    }


def stamp_round(result: dict, round_no: int) -> dict:
    """
    给结果盖上"这是第几轮跑的"。

    为什么非要有这个字段：`results` 的 reducer 是 `operator.add`，跨轮**只增不减**
    （往一个 add-型通道写 `[]` 等于没写，清不掉）。这是 LangGraph  reducer 的语义，
    不是 bug——它保证了并行写入不丢数据。代价是汇总节点拿到的是**所有轮次**的结果，
    必须自己挑出本轮的。round 就是那把尺子。

    顺带的好处：trace 里每条 worker 记录都带轮次，能一眼看出"这次请求第二轮
    又调了谁"，而这正是排查回环问题时要看的东西。
    """
    return {**result, "round": int(round_no or 0)}


def round_results(results: list, round_no: int) -> list:
    """
    从跨轮累加的结果里挑出某一轮的。

    没有 round 字段的当作"就是这一轮的"——手工构造 state 的场景（测试、脚本）
    不该被迫也把轮次标出来。
    """
    return [r for r in results or [] if r.get("round", round_no) == round_no]


def _registry() -> dict:
    """
    延迟取注册表。

    必须在**函数里** import 三个 worker：knowledge 会 import sibling 仓库
    rag_knowledge_base，如果放在模块顶层，那么"这台机器上没有那个仓库"就会在
    import 本模块时直接炸掉，连不带知识库的部署都起不来。放这里是故意的。
    """
    from ecom.orchestrator.workers.analytics import run_analytics
    from ecom.orchestrator.workers.knowledge import run_knowledge
    from ecom.orchestrator.workers.order import run_order

    return {
        "knowledge": run_knowledge,
        "order": run_order,
        "analytics": run_analytics,
    }


def worker_names() -> list:
    return list(_registry())


def run(
    name: str,
    question: str,
    chat_history: list = None,
    timeout: int = None,
    fn: Callable = None,
) -> dict:
    """
    调用一个 worker，把任何失败（异常/超时/不存在）翻成 ok=False 的结果。

    fn：可注入替身（测试用）。不传就按 name 查注册表。
    """
    fn = fn or _registry().get(name)
    if fn is None:
        return error_result(name, f"没有叫 {name} 的 worker")

    limit = timeout if timeout is not None else WORKER_TIMEOUT.get(name, WORKER_TIMEOUT_DEFAULT)
    box: dict = {}

    def target():
        try:
            box["result"] = fn(question, chat_history or [])
        except BaseException as exc:  # noqa: BLE001 —— worker 抛什么都得接住，见模块开头
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True, name=f"worker-{name}")
    started = time.perf_counter()
    thread.start()
    thread.join(limit)
    elapsed = int((time.perf_counter() - started) * 1000)

    if thread.is_alive():
        logger.warning("worker %s 超过 %ss 仍未返回，放弃等待", name, limit)
        return error_result(name, f"超时（超过 {limit}s 未返回）", elapsed)

    if "error" in box:
        exc = box["error"]
        logger.warning("worker %s 失败：%s: %s", name, type(exc).__name__, exc)
        return error_result(name, f"{type(exc).__name__}: {exc}", elapsed)

    payload = box.get("result") or {}
    merged = ok_result(
        worker=name,
        answer=payload.get("answer", ""),
        data=payload.get("data"),
        sources=payload.get("sources"),
        insufficient=payload.get("insufficient", False),
        tokens=payload.get("tokens"),
        elapsed_ms=elapsed,
    )
    # worker 若能报出更细的自己内部耗时，以其为准；否则用外面的墙钟时间。
    if isinstance(payload.get("elapsed_ms"), int):
        merged["elapsed_ms"] = payload["elapsed_ms"]
    return merged
