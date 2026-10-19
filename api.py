"""
HTTP 接口（FastAPI）。

    uv sync --extra api
    uv run uvicorn api:app --reload --port 8000
    # 或者直接跑：uv run python api.py

打开 http://127.0.0.1:8000/docs 有一份按 OpenAPI 生成的可交互文档——它和
这个文件里的类型标注是**同一份事实的两个视图**，所以改接口时文档会跟着动，
不需要另外维护一份。

【三个入口，一份实现】
    界面     app.py          Streamlit，给人看的
    命令行   main.py         CLI，排查用的
    接口     api.py          这个文件
三者调的是同一个 `supervisor.answer()`。所以"改一处逻辑要改三处"这件事
在这里不成立——这一条比任何一处接口设计都重要。

【为什么入口是 `def` 不是 `async def`】
图里全是**阻塞**调用（同步的 HTTP 请求、sqlite、subprocess）。写成 `async def`
却继续调阻塞代码，会把事件循环堵死——单线程下第二个请求要等第一个跑完，
而且症状是"并发一上来全卡住"，很难联想到一个 `async` 关键字。写 `def`，
FastAPI 会把它丢进线程池，这才是这类代码该待的地方。要真做高并发，
该动的是把阻塞调用换成异步的，不是给入口加 `async`。

（顺带：正因为会被丢进线程池，`/chat` 是**真的会并发**执行——这是会话存储
那边需要一把锁的原因，见 ecom/orchestrator/session.py。）

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

【鉴权：这里**默认不强制**，后台那边默认拒绝启动】
两端的不对称是刻意的，不是忘了统一：
    api.py    读为主，面向"clone 下来就能调"。没配 API_KEY 就放行，但**打一条
              WARNING**，把"这个服务是裸的"明说出来，而不是静静地不管。
    admin.py  能对**花钱的操作**（批准退货）下结论。密码没配就拒绝启动。
一句话：**读可以默认开放，写必须默认关闭。**

配了 `API_KEY` 之后，`/chat` 和 `/trace` 要求请求头 `X-API-Key`。
`/health` 和 `/docs` **永远放行**——健康检查需要凭证会让探针跟着一起需要
凭证，而它唯一的用途就是"在什么都还没配好的时候也能被调到"。

【错误体为什么长一个样，且总带 request_id】
以前 `answer()` 抛异常就是 FastAPI 默认的 500，**没有任何结构化错误体**；
客户端拿到的是一个纯文本 "Internal Server Error"，没法区分"我请求写错了"
和"你服务挂了"。现在三类失败（HTTPException / 请求校验失败 / 未捕获异常）
都返回同一个 `ErrorResponse`。

**不回异常原文**：里面可能有内部路径、SQL、甚至配置片段。回一句人话 + 一个
request_id，出问题时拿这个 id 去 `logs/` 里对——错误体的用途是"让客户端知道
该怎么办"，不是"让人隔空读源码"。
"""
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from ecom.orchestrator import memory, session
from ecom.orchestrator.core.supervisor import answer
from ecom.orchestrator.db.engine import get_db
from ecom.orchestrator.settings import API_KEY, MAX_QUESTION_CHARS, MAX_ROUNDS
from ecom.orchestrator.utils import trace
from ecom.orchestrator.utils.logger import setup_logger
from ecom.orchestrator.workers.base import worker_names

logger = setup_logger(__name__)

TAGS = [
    {"name": "health", "description": "存活与组件概览。**永远不需要凭证**——探针和监控走这里。"},
    {"name": "chat", "description": "一次提问：路由 → worker → 汇总。多轮由 `session_id` 串起来。"},
    {"name": "trace", "description": "回看历史请求的决策链（路由、每个 worker 的成功与否和耗时）。"},
]


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    启动时清一次过期会话。

    **为什么放在启动而不是"读的时候顺手删"**：懒清理会让一个只读操作产生写
    副作用，出问题（"我的会话怎么没了"）时极难定位。代价是长驻进程不会自我
    清理，所以要外挂一行 cron——这条写进 README 的已知限制。
    """
    result = session.purge()
    # 记忆也一起报：它是这段对话的摘要，**也是用户说过的话**。只报消息数的话，
    # "记忆有没有被删掉"这件事在日志里看不见——而它恰恰是最该被看见的那部分。
    logger.info(
        "会话库就绪：%s（清理过期 %s 个会话 / %s 条消息 / %s 条记忆）",
        session.SESSION_DB_PATH,
        result["sessions"],
        result["messages"],
        result["memories"],
    )
    yield


app = FastAPI(
    title="多 Agent 电商客服",
    version="0.1.0",
    lifespan=lifespan,
    openapi_tags=TAGS,
    description=(
        "一个 Supervisor 带多个 worker 的电商客服服务：路由决定交给谁"
        "（知识库 / 订单 / 数据分析），汇总把结论揉成一段话。\n\n"
        "**这是三个入口之一**——`app.py`（界面）、`main.py`（命令行）、"
        "`api.py`（这里）调的是同一个 `supervisor.answer()`，不是三份实现。\n\n"
        "**多轮对话**：第一次请求不带 `session_id`，服务端会生成一个并放在响应里；"
        "下一次带上它就能接上上下文。会话落在磁盘上，默认 30 天没动静就清掉。\n\n"
        "⚠️ `session_id` **是凭证不是账号**——拿到它就能读到这个会话的全部历史。"
        "别把它当身份用。"
    ),
)


# ═══════════════════════════════════════════════════════════════════════════
# 数据模型
#
# 这些类不只是给 FastAPI 做校验用的：它们同时就是 /docs 上那份文档的**来源**。
# 所以字段上的 description 是写给"照着文档接接口的人"看的，不是写给读代码的人。
# ═══════════════════════════════════════════════════════════════════════════
class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=MAX_QUESTION_CHARS,
        description=f"用户这一句。上限 {MAX_QUESTION_CHARS} 字符——路由的 prompt "
                    "会把这句话整个放进去，不设上限等于允许一句话把上下文撑爆。",
        examples=["订单 52644 到哪了"],
    )
    session_id: str = Field(
        default="",
        description="上一轮响应里返回的 `session_id`。**留空表示新开一个会话**，"
                    "服务端会生成一个并通过响应返回。",
    )


class ChatResponse(BaseModel):
    request_id: str = Field(description="本次请求的 id。troubleshooting 时拿它对 traces。")
    session_id: str = Field(
        description="**接上下一轮的唯一凭据**，客户端必须存下来。"
                    "它同时是凭证——拿到它就能读到这个会话的全部历史。"
    )
    answer: str = Field(description="汇总后的答复。")
    sources: list = Field(description="结论里引用到的来源（订单库 / 知识库文档名等）。")
    decision: str = Field(description="汇总的决策：答 / 重试 / 放弃。")
    plan: list = Field(description="这一轮路由给谁了（worker 名字）。")
    tokens: dict = Field(description="token 用量，`{\"in\": .., \"out\": ..}`。")


class HealthResponse(BaseModel):
    status: str = Field(description="恒定 `ok`——这里只报「组件在不在」，不报「组件好不好」。")
    workers: list[str] = Field(description="已注册的 worker 名字。")
    order_db: str = Field(description="订单库后端：`sqlite` 或 `mysql`。")
    max_rounds: int = Field(description="一次提问最多跑几轮（含首轮）。")


class TraceItem(BaseModel):
    """
    一条 trace。

    **刻意不把字段列全**：trace 是编排层写的，字段会随那层演进而增减，在这里
    抄一份字段清单就是第二个真相来源，而且它只会往一个方向漂——漂了没有任何
    东西会报警。所以这里只声明"我们保证有的"那几个，其余原样透传。
    """

    model_config = ConfigDict(extra="allow")

    request_id: str = Field(default="", description="对应的请求 id。")
    question: str = Field(default="", description="当时问的那句话。")


class TraceResponse(BaseModel):
    count: int = Field(description="返回了几条。")
    items: list[TraceItem] = Field(description="最近的请求，新的在后。")


class ErrorResponse(BaseModel):
    """
    所有失败路径的统一形状。

    **故意没有 `detail`**：那是 FastAPI 默认的字段，形状随异常类型变化
    （校验错误是一个数组，HTTPException 是一个字符串）。客户端为了兼容它
    得写两种分支。统一成一个字符串，代价是丢掉了校验错误的结构——换来的是
    "错误只有一种形状"。
    """

    error: str = Field(description="一句能读的话，**不是异常的原文**——原文可能带内部路径。")
    request_id: str = Field(description="拿这个 id 去服务端日志里对。")


# ═══════════════════════════════════════════════════════════════════════════
# 鉴权
# ═══════════════════════════════════════════════════════════════════════════
if not API_KEY:
    # **明确地说出来，不要静默。** 一个"以为配了其实没配"的服务，比一个
    # 明知没有鉴权的服务危险得多——前者会让人以为它是安全的。
    logger.warning(
        "没有配置 API_KEY：/chat 和 /trace 现在**任何人**都能调用。"
        "要开鉴权就在 .env 里设置 API_KEY，然后重新启动。"
    )
elif not API_KEY.isascii():
    # **这条不是洁癖，是"配了等于没配"。** HTTP 头字段的值只能是 ASCII 字节
    # （RFC 9110 的 field-value），httpx 在客户端就会拒收一个中文值：
    #     UnicodeEncodeError: 'ascii' codec can't encode characters …
    # 也就是说这个 key **任何客户端都发不出来**，服务端这边永远收不到它的头，
    # 结果是每一个请求都 401——而现象（"明明配对了还是 401"）会让人去查
    # compare_digest、去查中间件，很难想到问题出在"这个字符串没法进 HTTP 头"。
    logger.warning(
        "API_KEY 含非 ASCII 字符：HTTP 头只能传 ASCII，**客户端根本发不出这个值**，"
        "结果是所有请求都会 401。请改成一个 ASCII 的 key。"
    )


def require_api_key(x_api_key: str = Header(default="", description="配了 API_KEY 时才需要。")) -> None:
    """
    **没配 API_KEY 就放行**（并在 import 时告警），配了就必须对。

    比较用 `secrets.compare_digest`：普通的 `==` 会在第一个不同的字节上
    就返回，逐字节的时间差**在统计上是可以被利用的**（timing attack）。
    这里先 `.encode()` 成 bytes——`compare_digest` 碰上非 ASCII 的 str
    会直接抛 `TypeError`，而中文密码是很自然会被设出来的东西。
    """
    if not API_KEY:
        return
    if not secrets.compare_digest(x_api_key.encode("utf-8"), API_KEY.encode("utf-8")):
        raise HTTPException(
            status_code=401,
            detail="X-API-Key 不正确或缺失",
            headers={"WWW-Authenticate": "X-API-Key"},
        )


# ═══════════════════════════════════════════════════════════════════════════
# 端点
# ═══════════════════════════════════════════════════════════════════════════
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="存活与组件概览",
    description="**永远不需要凭证。** 只报「组件在不在」，不报「组件好不好」："
                "真去 ping 一遍模型/向量库会让健康检查本身变成一个慢接口，"
                "而 K8s 那种探针是按秒调的——一个慢的健康检查会把系统探死。"
                "深检查请 exec 进去跑一次 `python main.py`。",
)
def health() -> dict:
    db = get_db()
    return {
        "status": "ok",
        "workers": worker_names(),
        "order_db": db.backend,
        "max_rounds": MAX_ROUNDS,
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["chat"],
    summary="问一句",
    description="路由决定交给谁（知识库 / 订单 / 数据分析），worker 跑完由汇总"
                "揉成一段话。首轮不带 `session_id`，之后带上响应里返回的那个即可多轮。",
    dependencies=[Depends(require_api_key)],
    responses={401: {"model": ErrorResponse, "description": "配了 API_KEY 但没带或带错"}},
)
def chat(req: ChatRequest) -> dict:
    # 先确保会话存在再读历史——顺序反过来的话，新会话的第一问会因为
    # "查不到这个 session_id"而拿不到空历史（这一版不会报错，只是白查一次）。
    session_id = session.ensure_session(req.session_id)
    history = session.history(session_id)

    # 记忆拼在真实历史**前面**，而且**每轮现拼、不落盘**——把 `augmented_history`
    # 的返回值直接交给 `answer()`，**不要存回去**（存回去会跨轮重复累积）。
    # 会话还没跑过 `distill`（库是空的）时它原样返回，**一次 embedding 都不调**，
    # 所以没有记忆的会话在这条路上是零额外开销。
    final = answer(req.question, memory.augmented_history(session_id, req.question, history))

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


@app.get(
    "/trace",
    response_model=TraceResponse,
    tags=["trace"],
    summary="最近若干条请求的决策链",
    description="**这是给别人看「系统怎么想的」那扇窗**：一条记录里有路由、"
                "每个 worker 的成功与否和耗时、最后的决策。界面会把这一层"
                "藏起来（超时变成转圈、异常变成红框），要查问题就看这里。",
    dependencies=[Depends(require_api_key)],
    responses={
        401: {"model": ErrorResponse, "description": "配了 API_KEY 但没带或带错"},
        404: {"model": ErrorResponse, "description": "还没有任何请求记录"},
    },
)
def traces(limit: int = 20) -> dict:
    rows = _read_traces(limit)
    if not rows:
        raise HTTPException(status_code=404, detail="还没有请求记录")
    return {"count": len(rows), "items": rows}


# ═══════════════════════════════════════════════════════════════════════════
# 错误：三种失败路径收敛到同一个形状
# ═══════════════════════════════════════════════════════════════════════════
def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(error=message, request_id=trace.new_request_id()).model_dump(),
    )


@app.exception_handler(StarletteHTTPException)
async def _on_http_error(_, exc: StarletteHTTPException) -> JSONResponse:
    """`HTTPException`（我们主动抛的 404/401）。"""
    return _error(exc.status_code, str(exc.detail))


@app.exception_handler(RequestValidationError)
async def _on_bad_request(_, exc: RequestValidationError) -> JSONResponse:
    """请求体/参数不合法。字段名带上，**不带原始输入**——那可能是用户的问题
    原文，回显进日志和错误体没什么好处。"""
    missing = "、".join(".".join(str(p) for p in e["loc"][1:]) for e in exc.errors()[:5])
    return _error(422, f"请求格式不对，检查这些字段：{missing or '（见 /docs）'}")


@app.exception_handler(Exception)
async def _on_unhandled(_, exc: Exception) -> JSONResponse:
    """
    兜底。

    **异常原文只进日志，不进响应体。** 图里会抛的东西五花八门（模型超时、
    sqlite 锁、子进程崩），原文里可能带内部路径甚至配置片段。给客户端一句
    人话加一个 request_id，排查靠那个 id 去 logs/ 里对。
    """
    request_id = trace.new_request_id()
    logger.exception("未处理的异常 request_id=%s", request_id)
    return _error(500, "服务内部错误。把这个 request_id 提供给维护者，日志里有完整原因。")


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
