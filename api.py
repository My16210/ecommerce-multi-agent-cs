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

【会话状态为什么是**磁盘**，以及为什么它不是 checkpointer】
这段以前写的是"进程内字典，重启即丢、多副本不共享"。那是个明确的边界，但它
有个更硬的问题：**它把"多轮对话"变成了一个假承诺**——界面上聊得好好的，容器
一重启上下文就全没了，而这在演示里一定会被撞上。

现在落在 `data/orchestrator/sessions.db`（`ecom/orchestrator/session.py`）。
两个必须说清的点：

1. **存的是扁平消息行，不是图状态。** 所以这里没有 `SqliteSaver`。为什么
   supervisor 不该复用 `ask.py` 那套 checkpointer，`session.py` 的 docstring
   里写了四条理由（最硬的是一条：`chat_history` 在这个图里是**节点输入**，
   不是图状态；而 `results` 那唯一一条 add-型通道已经警告过"写 `[]` 等于没写"）。
2. **它的代价从"丢数据"换成了"留数据"。** 以前重启即丢，所以不需要清理；
   现在数据会一直在磁盘上，而里面是用户说过的话。于是多了一个保留期
   （`SESSION_TTL_DAYS`，默认 30 天）、一个 `purge()`、一个 CLI。
   启动时清一次，长驻期间不自动清——这条限制写在 README 里。

【仍然没解决的那条边界：多副本】
落到本地 sqlite 只解决了"重启丢"，**没解决"两个副本共享"**——多副本各写各的
文件，上下文照样会岔。要真解决得把存储换成 Redis 之类的外部服务，那是另一个
决定（多一个要部署的东西），这次**明确不做**。但至少这个边界现在是从
"sessions.db 是单节点文件"读得出来的，不再是藏在"我们用了 Redis"背后的假设。
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ecom.orchestrator import session
from ecom.orchestrator.core.supervisor import answer
from ecom.orchestrator.db.engine import get_db
from ecom.orchestrator.settings import MAX_QUESTION_CHARS, MAX_ROUNDS
from ecom.orchestrator.utils import trace
from ecom.orchestrator.utils.logger import setup_logger
from ecom.orchestrator.workers.base import worker_names

logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    启动时清一次过期会话。

    **为什么放在启动而不是"读的时候顺手删"**：懒清理会让一个只读操作产生写
    副作用，出问题（"我的会话怎么没了"）时极难定位。代价是长驻进程不会自我
    清理，所以要外挂一行 cron——这条写进 README 的已知限制。
    """
    result = session.purge()
    logger.info(
        "会话库就绪：%s（清理过期 %s 个会话 / %s 条消息）",
        session.SESSION_DB_PATH,
        result["sessions"],
        result["messages"],
    )
    yield


app = FastAPI(title="多 Agent 电商客服", version="0.1.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    # 空串 = 新开一个会话。服务端生成 id 并通过 ChatResponse 还给客户端，
    # 客户端下一次带上它就能接上上下文。
    session_id: str = ""


class ChatResponse(BaseModel):
    request_id: str
    # **必须回给客户端**：这是接上下一轮的唯一凭据。不给的话，落盘会话
    # 就只有"服务端知道"，多轮在 HTTP 上根本用不起来。
    session_id: str
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
    # 先确保会话存在再读历史——顺序反过来的话，新会话的第一问会因为
    # "查不到这个 session_id"而拿不到空历史（这一版不会报错，只是白查一次）。
    session_id = session.ensure_session(req.session_id)
    history = session.history(session_id)

    final = answer(req.question, history)

    # **先答完再落盘**。反过来的话，模型调用失败也会留下一条"问了没答"的
    # 记录。`append_turn` 内部一问一答是同一个提交，所以这里要么成对写入，
    # 要么一行都不写。
    session.append_turn(session_id, req.question, final.get("answer") or "")

    return {
        "request_id": final.get("request_id", ""),
        "session_id": session_id,
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
