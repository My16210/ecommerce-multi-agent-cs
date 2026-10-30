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
import inspect
import queue
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

    **必须在函数里** import 三个 worker：knowledge 那条链会拉起 chromadb /
    jieba / 向量库客户端，analytics 会拉起 matplotlib。放在模块顶层的话，
    只想跑订单那一路的部署（或一个还没灌数据的环境）也会被这些重依赖拖住，
    import supervisor 就失败。延迟到真正要跑某个 worker 时再付这个成本。
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


def _accepts(fn: Callable, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _invoke(
    fn: Callable,
    question: str,
    chat_history: list,
    customer_id: str,
    on_token: Callable[[str], None] | None = None,
):
    """
    按 worker 的签名决定传不传 `customer_id` / `on_token`。

    只有订单 worker 需要 `customer_id`，只有知识库 worker 支持 `on_token`；
    测试里的替身大多是 `lambda q, h:` 两参数，硬塞多余参数会把一大批测试打成
    TypeError。**用签名判断，而不是 try/except TypeError**——后者会把 worker
    内部真正的 TypeError 也吞掉，变成"少传一个参数"的假故障。
    """
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(question, chat_history)
    kwargs = {}
    if "customer_id" in parameters:
        kwargs["customer_id"] = customer_id
    if on_token is not None and "on_token" in parameters:
        kwargs["on_token"] = on_token
    return fn(question, chat_history, **kwargs)


def run(
    name: str,
    question: str,
    chat_history: list = None,
    timeout: int = None,
    fn: Callable = None,
    customer_id: str = "",
    emit: Callable[[str], None] | None = None,
) -> dict:
    """
    调用一个 worker，把任何失败（异常/超时/不存在）翻成 ok=False 的结果。

    fn：可注入替身（测试用）。不传就按 name 查注册表。
    emit：逐 token 转发（只有流式路径会传）。worker 跑在子线程里，
    `ContextVar` 不会跟过去，所以这里用线程安全队列把 token 运回来，
    由**调用线程**交给 LangGraph 的 writer——writer 依赖图运行时的 context，
    在子线程里直接调会拿不到配置。
    """
    fn = fn or _registry().get(name)
    if fn is None:
        return error_result(name, f"没有叫 {name} 的 worker")

    limit = timeout if timeout is not None else WORKER_TIMEOUT.get(name, WORKER_TIMEOUT_DEFAULT)
    box: dict = {}
    tokens: queue.Queue[str] = queue.Queue()
    relay = emit is not None and _accepts(fn, "on_token")

    def target():
        try:
            box["result"] = _invoke(
                fn,
                question,
                chat_history or [],
                customer_id,
                tokens.put if relay else None,
            )
        except BaseException as exc:  # noqa: BLE001 —— worker 抛什么都得接住，见模块开头
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True, name=f"worker-{name}")
    started = time.perf_counter()
    thread.start()
    if relay:
        relay_broken = False

        def forward(text: str) -> None:
            nonlocal relay_broken
            if not text or relay_broken:
                return
            try:
                emit(text)
            except Exception as exc:  # noqa: BLE001 —— 转发失败不该让作答失败
                logger.warning("worker %s 的流式转发失败，继续取结果：%s", name, exc)
                relay_broken = True

        deadline = started + limit
        while thread.is_alive():
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                text = tokens.get(timeout=min(0.05, remaining))
            except queue.Empty:
                continue
            forward(text)
        # 线程刚结束时，队列里可能还有最后几块；超时放弃的线程不再往外发。
        if not thread.is_alive():
            while True:
                try:
                    forward(tokens.get_nowait())
                except queue.Empty:
                    break
    else:
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
