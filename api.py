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
三者调的是同一套编排逻辑（`supervisor.answer()` / `answer_stream()`，两者
共用同一份收尾与记账）。所以"改一处逻辑要改三处"这件事在这里不成立——
这一条比任何一处接口设计都重要。

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
import inspect
import json
import secrets
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from ecom import demo
from ecom.auth import settings as auth_settings
from ecom.auth import store as auth_store
from ecom.orchestrator import memory, session
from ecom.orchestrator.core.supervisor import answer, answer_stream
from ecom.orchestrator.db.engine import get_db
from ecom.orchestrator.settings import API_KEY, MAX_QUESTION_CHARS, MAX_ROUNDS
from ecom.orchestrator.utils import trace
from ecom.orchestrator.utils.logger import setup_logger
from ecom.orchestrator.workers.base import worker_names
from ecom.settings import ROOT

logger = setup_logger(__name__)

TAGS = [
    {"name": "health", "description": "存活与组件概览。**永远不需要凭证**——探针和监控走这里。"},
    {"name": "auth", "description": "用用户名 + 密码换短期 bearer token（需要 `AUTH_ENABLED=1`）。"},
    {"name": "chat", "description": "一次提问：路由 → worker → 汇总。多轮由 `session_id` 串起来。`/chat` 一次返回完整 JSON，`/chat/stream` 走 SSE。"},
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
    # 登录会话也要清一次：它们是短期凭证，过期后留着只会让 auth.db 变大。
    auth_store.purge_sessions()
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
        "`api.py`（这里）调的是同一套编排逻辑（`/chat` → `answer()`，"
        "`/chat/stream` → `answer_stream()`），不是三份实现。\n\n"
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


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64, examples=["alice"])
    password: str = Field(
        min_length=1,
        max_length=256,
        description="明文只在这条请求里出现；服务端存的是 scrypt 哈希。",
    )


class LoginResponse(BaseModel):
    token: str = Field(description="短期 bearer token，默认 12 小时；**只返回这一次**。")
    expires_at: str = Field(description="过期时间（本地时间字符串）。")
    name: str = Field(description="principal 名字。")
    role: str = Field(description="customer / agent / admin。")


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
    workers: list[str] = Field(description="已注册的 worker 名字（演示模式下不含被停用的）。")
    order_db: str = Field(description="订单库后端：`sqlite` 或 `mysql`。")
    max_rounds: int = Field(description="一次提问最多跑几轮（含首轮）。")
    demo: bool = Field(default=False, description="是否运行在只读演示模式。")
    auth: bool = Field(default=False, description="是否启用了 bearer token 身份层。")


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


# 鉴权没开时的占位 principal。`id=""` 表示"不做会话归属检查"（老行为）。
ANONYMOUS_PRINCIPAL = {"id": "", "name": "anonymous", "role": "admin"}


def _bearer_token(authorization: str) -> str:
    scheme, _, token = str(authorization or "").strip().partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


def require_access(
    authorization: str = Header(default="", description="`Bearer <token>`；开了 AUTH_ENABLED 才需要。"),
    x_api_key: str = Header(default="", description="兼容旧的 X-API-Key 用法。"),
) -> dict:
    """
    统一的访问入口：**开了身份层就认 token，没开就走旧的 API_KEY 语义**。

    两种模式不混用：开了 AUTH_ENABLED 之后，`X-API-Key` 只作为 bearer 的
    兼容别名，旧的共享 key 不再有特殊地位——否则"一个共享 key = 所有人"
    会从后门回来。
    """
    if auth_settings.AUTH_ENABLED:
        token = _bearer_token(authorization) or str(x_api_key or "").strip()
        principal = auth_store.authenticate(token)
        if principal is None:
            raise HTTPException(
                status_code=401,
                detail="需要有效的 bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return principal

    require_api_key(x_api_key)
    return dict(ANONYMOUS_PRINCIPAL)


def require_admin(principal: Annotated[dict, Depends(require_access)]) -> dict:
    """admin 才能看 trace。没开身份层时沿用旧语义：API_KEY 对了就行。"""
    if auth_settings.AUTH_ENABLED and not auth_store.has_role(principal, "admin"):
        raise HTTPException(status_code=403, detail="需要 admin 角色")
    return principal


def _session_for(requested: str, principal: dict) -> str:
    """开会话并做归属检查；不属于当前 principal 时按 404 处理（不暴露存在性）。"""
    try:
        return session.ensure_session(requested, owner_id=principal.get("id", ""))
    except session.SessionAccessError:
        raise HTTPException(status_code=404, detail="会话不存在或已过期") from None


def _call_answer(question: str, history: list, customer_id: str) -> dict:
    """测试会把 `answer` 换成两参数替身；有 customer_id 形参时才传。"""
    if "customer_id" in inspect.signature(answer).parameters:
        return answer(question, history, customer_id=customer_id)
    return answer(question, history)


def _call_answer_stream(question: str, history: list, customer_id: str):
    if "customer_id" in inspect.signature(answer_stream).parameters:
        return answer_stream(question, history, customer_id=customer_id)
    return answer_stream(question, history)


def require_demo_access(
    x_invite_code: str = Header(
        default="", alias="X-Invite-Code", description="演示模式打开时才需要。"
    ),
) -> None:
    """
    演示模式的邀请码。**没开演示模式就放行**（和 API_KEY 的"不配就放行"同向，
    但语义不同：这里是"没启用这个功能"，不是"选择不鉴权"）。

    它和 API_KEY 是两道独立的门，可以同时开：API_KEY 管"谁能调接口"，
    邀请码管"公网演示给谁看"。只开一个也能工作，但公网演示建议两个都开。
    """
    if not demo.enabled():
        return
    if not demo.check_invite(x_invite_code):
        raise HTTPException(status_code=403, detail="演示环境需要正确的 X-Invite-Code")


def _demo_precheck(request: Request) -> None:
    """
    每次问答前的两道闸：限流、每日预算。**都在真正调模型之前**——拦在调用
    之后等于钱已经花了。演示模式没开时它是空操作。
    """
    if not demo.enabled():
        return
    client = request.client.host if request.client else "unknown"
    if not demo.allow_request(f"ip:{client}"):
        raise HTTPException(
            status_code=429,
            detail="请求过于频繁，请稍后再试",
            headers={"Retry-After": "60"},
        )
    if demo.budget_exhausted():
        raise HTTPException(status_code=429, detail="今天的演示额度已经用完")


# 登录限流：进程内滑动窗口。它挡的是"对着一台实例猜密码"，不是分布式撞库；
# 真要跨副本，得把这本账挪到共享存储（和 demo 的预算账本同一个道理）。
_LOGIN_MAX_ATTEMPTS = 10
_LOGIN_WINDOW_SECONDS = 60.0
_login_hits: dict[str, list[float]] = {}
_login_lock = threading.Lock()


def _login_precheck(request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _login_lock:
        recent = [
            stamp
            for stamp in _login_hits.get(client, [])
            if now - stamp < _LOGIN_WINDOW_SECONDS
        ]
        if len(recent) >= _LOGIN_MAX_ATTEMPTS:
            _login_hits[client] = recent
            raise HTTPException(
                status_code=429,
                detail="登录尝试过于频繁，请稍后再试",
                headers={"Retry-After": "60"},
            )
        recent.append(now)
        _login_hits[client] = recent
        if len(_login_hits) > 2048:
            for key, hits in list(_login_hits.items()):
                if not hits or now - hits[-1] >= _LOGIN_WINDOW_SECONDS:
                    _login_hits.pop(key, None)


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
        # 演示模式下不报告被停用的单元：健康检查也在公网上，报一个"存在但
        # 不可用"的 worker 会让人以为只是这一次没路由到它。
        "workers": demo.filter_workers(worker_names()),
        "order_db": db.backend,
        "max_rounds": MAX_ROUNDS,
        "demo": demo.enabled(),
        "auth": auth_settings.AUTH_ENABLED,
    }


@app.post(
    "/auth/login",
    response_model=LoginResponse,
    tags=["auth"],
    summary="用户名 + 密码换短期 token",
    description="需要服务端 `AUTH_ENABLED=1` 且该 principal 已用 "
                "`python -m ecom.auth.cli set-password` 设过密码。返回的 token "
                "默认 12 小时有效；和长期 API token 走同一套 bearer 校验。",
    responses={
        401: {"model": ErrorResponse, "description": "用户名或密码不正确"},
        404: {"model": ErrorResponse, "description": "身份层未启用"},
        429: {"model": ErrorResponse, "description": "登录尝试过于频繁"},
    },
)
def login(req: LoginRequest, request: Request) -> dict:
    if not auth_settings.AUTH_ENABLED:
        # 404 而不是 403：没开身份层时，这个端点"不存在"比"存在但不给用"更准确。
        raise HTTPException(status_code=404, detail="身份层未启用（AUTH_ENABLED=0）")
    _login_precheck(request)
    principal = auth_store.verify_password(req.username, req.password)
    if principal is None:
        # 不区分"用户不存在"和"密码错"：区分了就是一个用户名枚举接口。
        raise HTTPException(status_code=401, detail="用户名或密码不正确")
    issued = auth_store.issue_session(principal)
    return {
        "token": issued["token"],
        "expires_at": issued["expires_at"],
        "name": principal["name"],
        "role": principal["role"],
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["chat"],
    summary="问一句",
    description="路由决定交给谁（知识库 / 订单 / 数据分析），worker 跑完由汇总"
                "揉成一段话。首轮不带 `session_id`，之后带上响应里返回的那个即可多轮。",
    dependencies=[Depends(require_demo_access)],
    responses={
        401: {"model": ErrorResponse, "description": "需要 bearer token（开了 AUTH_ENABLED 时）"},
        403: {"model": ErrorResponse, "description": "演示模式但邀请码不对"},
        429: {"model": ErrorResponse, "description": "请求太频繁或今日演示额度已用完"},
    },
)
def chat(
    req: ChatRequest, request: Request, principal: Annotated[dict, Depends(require_access)]
) -> dict:
    _demo_precheck(request)

    # 先确保会话存在再读历史——顺序反过来的话，新会话的第一问会因为
    # "查不到这个 session_id"而拿不到空历史（这一版不会报错，只是白查一次）。
    session_id = _session_for(req.session_id, principal)
    history = session.history(session_id)

    # 记忆拼在真实历史**前面**，而且**每轮现拼、不落盘**——把 `augmented_history`
    # 的返回值直接交给 `answer()`，**不要存回去**（存回去会跨轮重复累积）。
    # 会话还没跑过 `distill`（库是空的）时它原样返回，**一次 embedding 都不调**，
    # 所以没有记忆的会话在这条路上是零额外开销。
    final = _call_answer(
        req.question,
        memory.augmented_history(session_id, req.question, history),
        principal.get("customer_id") or "",
    )
    demo.add_tokens(final.get("tokens") or {})

    # **先答完再落盘**。反过来的话，模型调用失败也会留下一条"问了没答"的
    # 记录。`append_turn` 内部一问一答是同一个提交，所以这里要么成对写入，
    # 要么一行都不写。
    session.append_turn(
        session_id, req.question, final.get("answer") or "", owner_id=principal.get("id", "")
    )

    return {
        "request_id": final.get("request_id", ""),
        "session_id": session_id,
        "answer": final.get("answer", ""),
        "sources": final.get("sources") or [],
        "decision": final.get("decision", ""),
        "plan": final.get("plan") or [],
        "tokens": final.get("tokens") or {},
    }


# ═══════════════════════════════════════════════════════════════════════════
# 流式：SSE
#
# 事件只有四种，客户端按 `type` 分派即可：
#
#     {"type":"start", "request_id":"...", "session_id":"..."}
#     {"type":"delta", "text":"..."}
#     {"type":"final", "request_id":"...", "session_id":"...", "answer":"...",
#      "sources":[...], "decision":"...", "plan":[...], "tokens":{...},
#      "elapsed_ms":123}
#     {"type":"error", "error":"...", "request_id":"..."}
#
# **`final.answer` 是权威版本。** 汇总流到一半失败时服务端会退回拼接，
# 前面收到的 delta 拼起来可能和它不一致——流式天生带这个窗口，不是 bug。
#
# 为什么另开一个端点，不改 `/chat`：非流式那个是"一步拿到完整 JSON"的契约，
# 几十个调用方（包括测试）按它写的。让同一个路径按请求头切换返回 body 类型，
# 是那种"curl 一下看起来正常、客户端里悄悄坏掉"的改法。
# ═══════════════════════════════════════════════════════════════════════════
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    # nginx / Caddy 默认会缓冲响应，缓冲之后 token 就不是"逐字到达"了——
    # 现象是"流式写完了、浏览器一次性显示"。这个头让代理别攒。
    "X-Accel-Buffering": "no",
}


def _sse(payload: dict) -> str:
    """一条 SSE 帧。`ensure_ascii=False`：中文按原样发，省字节也好读日志。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post(
    "/chat/stream",
    tags=["chat"],
    summary="问一句（流式，SSE）",
    description="和 `/chat` 是同一套流程、同一份会话存储，区别只在于答复"
                "边生成边以 `text/event-stream` 推回来。事件类型见接口说明："
                "`start` / `delta` / `final` / `error`。\n\n"
                "**`final.answer` 是权威版本**：汇总中途失败时会退回拼接，"
                "前面 `delta` 拼起来可能和它不一致。\n\n"
                "只有在**多个单元给出结论、需要汇总**时才是真逐 token；"
                "只有一个结论时会作为一个 `delta` 整段到达——不把已完成的字符串"
                "切成小块假装在流。",
    dependencies=[Depends(require_demo_access)],
    responses={
        401: {"model": ErrorResponse, "description": "需要 bearer token（开了 AUTH_ENABLED 时）"},
        403: {"model": ErrorResponse, "description": "演示模式但邀请码不对"},
        429: {"model": ErrorResponse, "description": "请求太频繁或今日演示额度已用完"},
    },
)
def chat_stream(
    req: ChatRequest, request: Request, principal: Annotated[dict, Depends(require_access)]
) -> StreamingResponse:
    """
    同步生成器 + `StreamingResponse`。

    `answer_stream()` 是**阻塞**生成器（模型调用、sqlite 都是同步的），
    StreamingResponse 会把它放进线程池迭代——和 `/chat` 写 `def` 是同一个理由，
    不给事件循环塞阻塞代码。
    """
    # 闸必须在**开始发 SSE 之前**：一旦返回了 StreamingResponse，HTTP 状态码
    # 就已经是 200 了，再把 429 塞进流里，客户端会以为"答到一半失败了"。
    _demo_precheck(request)

    session_id = _session_for(req.session_id, principal)
    history = session.history(session_id)
    # 和 /chat 一样：记忆每轮现拼，不落盘。
    augmented = memory.augmented_history(session_id, req.question, history)

    def events():
        request_id = ""
        final = None
        elapsed_ms = None
        try:
            for event in _call_answer_stream(
                req.question, augmented, principal.get("customer_id") or ""
            ):
                kind = event.get("type")
                if kind == "start":
                    request_id = event.get("request_id") or ""
                    yield _sse(
                        {
                            "type": "start",
                            "request_id": request_id,
                            "session_id": session_id,
                        }
                    )
                elif kind == "delta":
                    yield _sse({"type": "delta", "text": event.get("text") or ""})
                elif kind == "final":
                    final = event.get("state") or {}
                    request_id = final.get("request_id") or request_id
                    elapsed_ms = event.get("elapsed_ms")
        except Exception:
            # **异常原文只进日志。** 错误体里给一句人话 + request_id，
            # 和 /chat 那条兜底同一个口径。
            logger.exception("流式请求失败 request_id=%s", request_id)
            yield _sse(
                {
                    "type": "error",
                    "error": "服务内部错误。把这个 request_id 提供给维护者，日志里有完整原因。",
                    "request_id": request_id or trace.new_request_id(),
                }
            )
            return

        if final is None:
            yield _sse(
                {
                    "type": "error",
                    "error": "这次请求没有正常结束。",
                    "request_id": request_id or trace.new_request_id(),
                }
            )
            return

        # **拿到完整答复之后再落盘。** 中途断开的请求不写会话——半句话接进
        # 下一轮，比没答更坏。和 /chat 的"先答完再落盘"是同一条规矩。
        session.append_turn(
            session_id, req.question, final.get("answer") or "", owner_id=principal.get("id", "")
        )
        demo.add_tokens(final.get("tokens") or {})

        yield _sse(
            {
                "type": "final",
                "request_id": request_id,
                "session_id": session_id,
                "answer": final.get("answer") or "",
                "sources": final.get("sources") or [],
                "decision": final.get("decision") or "",
                "plan": final.get("plan") or [],
                "tokens": final.get("tokens") or {},
                "elapsed_ms": elapsed_ms,
            }
        )

    return StreamingResponse(events(), media_type="text/event-stream", headers=SSE_HEADERS)


@app.get(
    "/trace",
    response_model=TraceResponse,
    tags=["trace"],
    summary="最近若干条请求的决策链",
    description="**这是给别人看「系统怎么想的」那扇窗**：一条记录里有路由、"
                "每个 worker 的成功与否和耗时、最后的决策。界面会把这一层"
                "藏起来（超时变成转圈、异常变成红框），要查问题就看这里。",
    dependencies=[Depends(require_admin)],
    responses={
        401: {"model": ErrorResponse, "description": "需要 bearer token（开了 AUTH_ENABLED 时）"},
        403: {"model": ErrorResponse, "description": "需要 admin 角色"},
        404: {"model": ErrorResponse, "description": "还没有任何请求记录"},
    },
)
def traces(limit: int = 20) -> dict:
    # 公网演示不开放 trace：里面有用户问过的话、路由理由和内部耗时。
    # **返回 404 而不是 403**——403 等于告诉对方"这里有东西，只是不给你"。
    if demo.enabled():
        raise HTTPException(status_code=404, detail="演示环境不提供请求记录")
    rows = _read_traces(limit)
    if not rows:
        raise HTTPException(status_code=404, detail="还没有请求记录")
    return {"count": len(rows), "items": rows}


# ═══════════════════════════════════════════════════════════════════════════
# 错误：三种失败路径收敛到同一个形状
# ═══════════════════════════════════════════════════════════════════════════
def _error(status: int, message: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(error=message, request_id=trace.new_request_id()).model_dump(),
        # 401 的 `WWW-Authenticate` 和 429 的 `Retry-After` 都靠这里带出去：
        # 统一错误体不能顺手把协议要求的头也统一掉。
        headers=headers,
    )


@app.exception_handler(StarletteHTTPException)
async def _on_http_error(_, exc: StarletteHTTPException) -> JSONResponse:
    """`HTTPException`（我们主动抛的 401/403/404/429）。"""
    return _error(exc.status_code, str(exc.detail), headers=dict(exc.headers or {}))


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


# ═══════════════════════════════════════════════════════════════════════════
# React 前端（可选）：构建好了就挂到 /ui
#
# 【为什么是"有就挂、没有就跳过"，而不是构建失败就拒绝启动】
# 这个仓库的第一入口始终是 `main.py`，API 和界面都是可选层。前端没构建
# （干净的 clone、只装了核心依赖的环境）不该让整个 API 起不来——那会把
# "可选"变成"必需"。反过来，dist 存在却不挂，则是"明明构建了却打不开"，
# 所以检查放在启动时，并且**只在没有的时候打一条 info**，不吵。
#
# 【为什么挂在 /ui 而不是根路径】
# `/docs`、`/health`、`/chat` 都在根路径上，根路径留给接口。前端是同一份
# 服务的**第二个视图**，不是它的替代品。vite.config.ts 的 `base` 必须和
# 这里一致（`/ui/`），否则页面能打开但资源 404。
# ═══════════════════════════════════════════════════════════════════════════
FRONTEND_DIST = Path(ROOT) / "frontend" / "dist"


def mount_frontend(target_app: FastAPI, dist: Path = FRONTEND_DIST) -> bool:
    """把构建好的静态站点挂到 `/ui`。返回是否挂上。"""
    if not (Path(dist) / "index.html").is_file():
        return False
    target_app.mount("/ui", StaticFiles(directory=str(dist), html=True), name="ui")
    return True


if not mount_frontend(app):
    logger.info("frontend/dist 不存在：/ui 暂不提供 React 界面（cd frontend && npm run build）")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
