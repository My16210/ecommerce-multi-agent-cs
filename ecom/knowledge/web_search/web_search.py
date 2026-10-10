"""
联网搜索模块
功能：给问答流程的第二路提供实时搜索结果

默认用 ddgs（DuckDuckGo，免 key、免费）。
如果 .env 里配了 TAVILY_API_KEY 就优先走 Tavily（结果对 LLM 更友好）。

无论哪条路失败都返回 [] —— 联网只是增强，不能把主链路拖垮。

**整条这一路有一个硬时间预算**（`WEB_SEARCH_TIMEOUT`，默认 8 秒），见
`_run_with_budget` 里为什么"给客户端传 timeout"不够。
"""
import threading
import time

from ecom.knowledge.settings import TAVILY_API_KEY, WEB_SEARCH_MAX_RESULTS, WEB_SEARCH_TIMEOUT
from ecom.knowledge.utils.logger import setup_logger
from ecom.knowledge.utils.timing import stage

logger = setup_logger(__name__)


def _run_with_budget(fn, budget: float):
    """
    在后台线程里跑 fn，最多等 budget 秒，超了就认输。

    为什么不简单地把 timeout 传给客户端：

      - **ddgs 的 timeout 是"每一批"的超时，不是总超时。** 它把引擎分批发请求，
        每批之间重新计时（源码里是 `wait(futures, timeout=self._timeout)` 循环），
        所以总耗时可以是 timeout 的若干倍。实测默认 5 秒的配置跑出过 5.6 秒，
        就是这么来的。只调那个参数，保证不了"最多 8 秒"。
      - Tavily 客户端没有稳定的超时参数可比。

    这里要的是对**整条联网这一路**的上限，所以自己掐表。

    超时之后不强行杀线程（Python 没有安全的杀线程手段），让它自己随请求超时结束。
    这是刻意的取舍：联网只是增强，宁可丢掉它，也要让答案按时出来——
    与文件开头"不能把主链路拖垮"那句是同一件事。
    """
    box: dict = {}

    def target() -> None:
        try:
            box["result"] = fn()
        except Exception as exc:  # noqa: BLE001 —— 原样带回主线程再决定怎么处理
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True, name="web-search")
    thread.start()
    thread.join(budget)

    if thread.is_alive():
        raise TimeoutError(f"联网搜索超过 {budget:.1f} 秒预算")
    if "error" in box:
        raise box["error"]
    return box["result"]


def _search_by_tavily(query: str, max_results: int) -> list:
    try:
        from tavily import TavilyClient
    except ImportError as exc:
        # tavily 是可选依赖（pyproject.toml 里的 search extra）。
        # 「配了 key 但没装包」是最容易踩的一步，而默认的 ModuleNotFoundError
        # 只说"没有这个模块"，不说怎么办——把它换成能照着做的提示，
        # 否则这行 warning 只会让人以为 Tavily 服务挂了。
        raise RuntimeError(
            "配了 TAVILY_API_KEY 但没装 tavily 包。"
            "请执行：uv sync --extra search（不装也不影响，会自动走 ddgs）"
        ) from exc

    client = TavilyClient(api_key=TAVILY_API_KEY)
    response = client.search(query=query, max_results=max_results)
    return [
        {
            "text": item.get("content", ""),
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "source": "web",
        }
        for item in response.get("results", [])
    ]


def _search_by_ddgs(query: str, max_results: int, timeout: float = None) -> list:
    from ddgs import DDGS

    # timeout 是 ddgs 的**每批**超时，不是总超时——它只是让内层别拖太久，
    # 真正的保证是外层 _run_with_budget 那把表。给个至少 1 秒的下限，
    # 免得预算快用完时传进去一个零点几秒的值，直接把 ddgs 一上来就掐死。
    kwargs = {}
    if timeout is not None:
        kwargs["timeout"] = max(1, int(timeout))

    with DDGS(**kwargs) as client:
        results = list(client.text(query, max_results=max_results))
    return [
        {
            "text": item.get("body", ""),
            "title": item.get("title", ""),
            "url": item.get("href", ""),
            "source": "web",
        }
        for item in results
    ]


def web_search(
    query: str,
    max_results: int = WEB_SEARCH_MAX_RESULTS,
    timeout: float = None,
) -> list:
    """
    联网搜索

    返回：[{"text": ..., "title": ..., "url": ..., "source": "web"}, ...]
    字段结构与知识库检索结果保持一致，方便生成节点统一处理

    timeout：整条这一路的时间预算（秒），默认取 WEB_SEARCH_TIMEOUT。
             Tavily 和 ddgs **共用这一个预算**——Tavily 慢掉了，ddgs 拿到的
             就是剩下的部分，而不是各自重新开始计时。
    """
    budget = WEB_SEARCH_TIMEOUT if timeout is None else timeout

    # 剩余预算 = min(deadline - now, budget)。
    # 那个 min 不是为了"更保守"，是因为**减法本身会溢出**：time.monotonic() 已经
    # 是个很大的数（机器开机以来的秒数），`(t0 + budget) - t0` 在浮点上可能落回
    # 比 budget 大一点点（实测 0.2 会算出 0.2000000000007276）。数值上无所谓，
    # 但"剩余预算超过了总预算"这句话本身是自相矛盾的，测试也不该为它让步。
    deadline = time.monotonic() + budget

    def remaining_budget() -> float:
        return min(deadline - time.monotonic(), budget)

    # 整段算一个阶段：Tavily 失败回退 ddgs 时，两次网络往返（含超时等待）
    # 都该算进这一路——画像里那一项才不会被伪装成"很快"。
    with stage("web_search"):
        if TAVILY_API_KEY:
            try:
                remaining = remaining_budget()
                results = _run_with_budget(
                    lambda: _search_by_tavily(query, max_results), remaining
                )
                logger.info("Tavily 搜索 %r 得到 %d 条", query, len(results))
                return results
            except Exception as exc:
                logger.warning("Tavily 搜索失败或超时，回退到 ddgs：%s", exc)

        remaining = remaining_budget()
        if remaining <= 0:
            logger.warning("联网搜索预算（%.1f 秒）已用尽，本轮跳过联网结果", budget)
            return []

        try:
            results = _run_with_budget(
                lambda: _search_by_ddgs(query, max_results, remaining), remaining
            )
            logger.info("ddgs 搜索 %r 得到 %d 条", query, len(results))
            return results
        except Exception as exc:
            logger.warning("联网搜索失败或超时（%s），本轮跳过联网结果", exc)
            return []


if __name__ == "__main__":
    for result in web_search("LangGraph 是什么", max_results=3):
        print(f"- {result['title']}\n  {result['url']}\n  {result['text'][:80]}...")
