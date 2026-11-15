"""
可观测：一次请求一行 JSONL。

为什么不是"打日志就行"——因为日志是给人扫的，而这里要回答的是**可聚合的问题**：
哪个 worker 失败率最高、p95 延迟多少、一次请求平均烧多少 token。
那些问题需要机器可读、一条记录一行的东西。所以日志（logger.py）和 trace 各管一头。

【为什么记录的是"事实"而不是"结论"】
每条记录写的是 route / 每个 worker 的 ok / elapsed_ms / tokens，而不是
"这次成功了吗"。判定成功是查询方的事——把结论也写进去，口径一变就要重跑历史数据。
"""
import json
import threading
import uuid
from datetime import datetime

from ecom.orchestrator.settings import LOGS_DIR

_lock = threading.Lock()


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def record(entry: dict) -> str:
    """
    追加一条请求记录，返回 request_id。

    多线程下要上锁：api.py 里 `/chat` 是同步函数，FastAPI 会把它丢进线程池，
    于是真有多个请求同时写同一个文件。append 本身在多数平台是原子的，但
    "一次写一行"这件事不该指望文件系统的原子性——锁很便宜。
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    entry = {"ts": datetime.now().isoformat(timespec="seconds"), **entry}
    line = json.dumps(entry, ensure_ascii=False)
    with _lock:
        with open(LOGS_DIR / "requests.jsonl", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    return entry.get("request_id", "")


def summarize_workers(results: list) -> dict:
    """把 worker 结果列表压成一行能看懂的统计，写进 trace。

    `calls` 仍然是**这个 worker 跑了几次**（多轮 / 重试会累加），语义没变。
    每次 worker 内部的模型调用记录放在 `call_records` 里；**只有真的产生过
    记录才加这个键**——没有调用就不出现，不是空数组。这样既不用改旧口径，
    又能区分「没调用」和「调用了但没记上」。
    """
    per_worker = {}
    for r in results:
        name = r.get("worker", "?")
        slot = per_worker.setdefault(name, {"calls": 0, "ok": 0, "ms": 0, "errors": []})
        slot["calls"] += 1
        slot["ok"] += 1 if r.get("ok") else 0
        slot["ms"] += int(r.get("elapsed_ms", 0) or 0)
        if not r.get("ok") and r.get("error"):
            slot["errors"].append(str(r["error"])[:200])
        records = r.get("calls") or []
        if records:
            slot.setdefault("call_records", []).extend(records)
    return per_worker


def collect_calls(results: list, state: dict = None) -> list:
    """把 router / 所有 worker / aggregate 的调用记录摊成一条扁平数组。

    顺序就是实际发生的顺序：先路由（可能多次 attempt / 多轮），再各 worker
    （results 本身按写入顺序跨轮累积），最后汇总。`results` 里的记录已经
    包含失败路径的 unknown 记录，所以这张表是完整的——不会因为某个 worker
    挂了就把那一次调用从成本里抹掉。
    """
    state = state or {}
    calls = list(state.get("router_calls") or [])
    for result in results or []:
        calls.extend(result.get("calls") or [])
    calls.extend(state.get("synth_calls") or [])
    return calls
