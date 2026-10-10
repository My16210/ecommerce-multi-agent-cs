"""
HTTP 接口（FastAPI）。

    uv sync --extra api
    uv run uvicorn api:app --reload --port 8000
    # 或者直接跑：uv run python api.py

【为什么入口是 `def` 不是 `async def`】
图里全是**阻塞**调用（同步的 HTTP 请求、sqlite、subprocess）。写成 `async def`
却继续调阻塞代码，会把事件循环堵死——单线程下第二个请求要等第一个跑完，
而且症状是"并发一上来全卡住"，很难联想到一个 `async` 关键字。写 `def`，
FastAPI 会把它丢进线程池，这才是这类代码该待的地方。要真做高并发，
该动的是把阻塞调用换成异步的，不是给入口加 `async`。

【会话状态为什么是进程内字典】
多轮对话要带 chat_history。这里用进程内字典 + 上限，**重启即丢、多副本不共享**。
对一个演示项目，这比引 Redis 划算；但它是个明确的边界，所以写在这里，
而不是等有人说"部署两个副本怎么上下文丢了"才发现。
"""
from collections import OrderedDict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ecom.orchestrator.core.supervisor import answer
from ecom.orchestrator.db.engine import get_db
from ecom.orchestrator.settings import MAX_QUESTION_CHARS, MAX_ROUNDS
from ecom.orchestrator.utils import trace
from ecom.orchestrator.utils.logger import setup_logger
from ecom.orchestrator.workers.base import worker_names

logger = setup_logger(__name__)

app = FastAPI(title="多 Agent 电商客服", version="0.1.0")

# 会话历史：最多记 200 个会话，每个会话最多 6 轮（12 条）。有上限才不会长跑之后
# 把内存吃满——一个没有上限的进程内缓存，等于一个延迟发生的 OOM。
_SESSIONS: OrderedDict = OrderedDict()
MAX_SESSIONS = 200
MAX_TURNS = 6


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    session_id: str = ""


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    sources: list
    decision: str
    plan: list
    tokens: dict


@app.get("/health")
def health() -> dict:
    """
    只报"组件在不在"，**不报"组件好不好"**。

    真去 ping 一遍模型/向量库会让健康检查本身变成一个慢接口，而 K8s 那种
    探针是按秒调的——一个慢的健康检查会把系统探死。深检查另开一个 /ready 端点。
    """
    db = get_db()
    return {
        "status": "ok",
        "workers": worker_names(),
        "order_db": db.backend,
        "max_rounds": MAX_ROUNDS,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> dict:
    history = _SESSIONS.get(req.session_id, []) if req.session_id else []
    final = answer(req.question, history)

    if req.session_id:
        _remember(req.session_id, req.question, final.get("answer") or "")

    return {
        "request_id": final.get("request_id", ""),
        "answer": final.get("answer", ""),
        "sources": final.get("sources") or [],
        "decision": final.get("decision", ""),
        "plan": final.get("plan") or [],
        "tokens": final.get("tokens") or {},
    }


@app.get("/trace")
def traces(limit: int = 20) -> dict:
    """
    最近若干条请求的 trace。**这是给别人看"系统怎么想的"的那扇窗**：
    一条记录里有路由、每个 worker 的成功与否和耗时、最后的决策。
    """
    rows = _read_traces(limit)
    if not rows:
        raise HTTPException(status_code=404, detail="还没有请求记录")
    return {"count": len(rows), "items": rows}


def _remember(session_id: str, question: str, answer_text: str) -> None:
    history = _SESSIONS.setdefault(session_id, [])
    history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer_text}]
    del history[:-MAX_TURNS * 2]
    _SESSIONS.move_to_end(session_id)
    while len(_SESSIONS) > MAX_SESSIONS:
        _SESSIONS.popitem(last=False)


def _read_traces(limit: int) -> list:
    import json

    path = trace.LOGS_DIR / "requests.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    out = []
    for line in lines[-max(1, min(limit, 200)):]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # 正在写的那一行可能是半行，跳过就好，不是故障
    return out


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
