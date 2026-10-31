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

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from ecom import chat_quota, demo
from ecom.auth import settings as auth_settings
from ecom.auth import store as auth_store
from ecom.orchestrator import memory, session, tickets
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
    {"name": "sessions", "description": "按登录身份回看自己的会话（只读 + 删自己的）。需要 `AUTH_ENABLED=1`。"},
    {"name": "tickets", "description": "转人工留下的工单：用户看状态，agent 看队列、认领、回复、关闭。"
                                   "需要 `AUTH_ENABLED=1`，且演示环境不开放。"},
    {"name": "trace", "description": "回看历史请求的决策链（路由、每个 worker 的成功与否和耗时）。"},
]


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    启动时先做配置安全检查，再清一次过期会话。

    **为什么放在启动而不是"读的时候顺手删"**：懒清理会让一个只读操作产生写
    副作用，出问题（"我的会话怎么没了"）时极难定位。代价是长驻进程不会自我
    清理，所以要外挂一行 cron——这条写进 README 的已知限制。
    """
    problem = auth_settings.production_problem(demo.enabled())
    if problem:
        # 拒绝启动而不是打 warning：一个"看起来正常、实际裸奔"的服务，
        # 比一个起不来、必须去改配置的服务危险得多。
        raise RuntimeError(problem)
    result = session.purge()
    # 登录会话也要清一次：它们是短期凭证，过期后留着只会让 auth.db 变大。
    auth_store.purge_sessions()
    # 审计按保留期清理。它包含退货审批等解释性记录，不能无限增长；
    # 默认 90 天，`AUDIT_RETENTION_DAYS<=0` 可关闭。
    purged_audit = auth_store.purge_audit()
    # 记忆也一起报：它是这段对话的摘要，**也是用户说过的话**。只报消息数的话，
    # "记忆有没有被删掉"这件事在日志里看不见——而它恰恰是最该被看见的那部分。
    logger.info(
        "会话库就绪：%s（清理过期 %s 个会话 / %s 条消息 / %s 条记忆 / %s 张工单 / %s 条审计）",
        session.SESSION_DB_PATH,
        result["sessions"],
        result["messages"],
        result["memories"],
        result.get("tickets", 0),
        purged_audit,
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


class LogoutResponse(BaseModel):
    revoked: bool = Field(
        description="是否真的吊销了一枚登录 token。"
                    "传长期 API token 时为 false——它不属于登出能结束的凭证。"
    )


class SessionSummary(BaseModel):
    """侧边栏要的一行：够显示标题、时间和条数，**不带正文**。"""

    session_id: str = Field(description="接上下一轮用的会话号。")
    title: str = Field(description="第一条用户提问截断后的标题；空会话是「未命名会话」。")
    created_at: str = Field(description="创建时间（本地时间字符串）。")
    updated_at: str = Field(description="最后一次写入的时间。")
    messages: int = Field(description="消息条数（一问一答算 2 条）。")


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary] = Field(description="最近更新的在前。只含当前登录账号的会话。")


class SessionMessage(BaseModel):
    id: int = Field(description="消息在库里的自增 id，按它排序。")
    role: str = Field(description="`user` / `assistant` / `agent`（人工客服）。")
    content: str = Field(description="消息原文。")
    created_at: str = Field(description="写入时间（本地时间字符串）。")
    author: str = Field(default="", description="`agent` 消息是谁回的；其余为空。")


class SessionDetail(BaseModel):
    session_id: str = Field(description="会话号。")
    title: str = Field(description="第一条用户提问截断后的标题。")
    created_at: str = Field(description="创建时间。")
    updated_at: str = Field(description="最后一次写入的时间。")
    messages: list[SessionMessage] = Field(description="按时间正序的最近 200 条消息。")


class HandoffInfo(BaseModel):
    """这一轮转人工之后落下的工单。`None` 表示没转 / 这个环境没有人工客服。"""

    ticket_id: str = Field(description="工单号，用户可以把它报给客服。")
    status: str = Field(description="`open`（排队中）或 `claimed`（已有人接手）。")
    queue_position: int = Field(description="前面还有几张没被认领的单；已接手是 0。")


class TicketInfo(BaseModel):
    ticket_id: str = Field(description="工单号。")
    status: str = Field(description="`open` / `claimed` / `closed`。")
    reason: str = Field(default="", description="转人工时记录的原因（机器给的决策理由）。")
    created_at: str = Field(description="开单时间。")
    updated_at: str = Field(description="最后一次变化的时间（认领 / 回复 / 关闭）。")
    claimed_by_name: str | None = Field(default=None, description="认领人名字；没人接手是 `null`。")
    queue_position: int = Field(default=0, description="排队位置；已接手或已关闭是 0。")


class TicketStatusResponse(BaseModel):
    ticket: TicketInfo | None = Field(
        default=None,
        description="这个会话最近的一张工单（包括已关闭的）；从没转过人工是 `null`。",
    )


class TicketSummary(TicketInfo):
    session_id: str = Field(description="工单对应的会话号。")
    title: str = Field(description="会话标题（第一条用户提问）。")


class TicketListResponse(BaseModel):
    tickets: list[TicketSummary] = Field(description="排队中的在前（先来先服务）；已关闭的按最近关闭在前。")


class TicketMessage(BaseModel):
    id: int = Field(description="消息在会话库里的自增 id。")
    role: str = Field(description="`user` / `assistant` / `agent`（人工客服）。")
    content: str = Field(description="消息原文。")
    created_at: str = Field(description="写入时间。")
    author: str = Field(default="", description="`agent` 消息是谁回的；其余为空。")


class TicketDetail(TicketSummary):
    transcript: list[TicketMessage] = Field(
        description="这个会话按时间正序的最近 200 条消息，**含用户和机器人的对话**——"
                    "客服要接手就得先看到前因。"
    )


class ReplyRequest(BaseModel):
    content: str = Field(
        min_length=1,
        max_length=tickets.MAX_REPLY_CHARS,
        description="客服回复的正文。写进用户会话，用户下次打开或轮询时能看到。",
        examples=["你好，我是人工客服小李，订单 52644 已签收，我来帮你确认退款进度。"],
    )


class TicketTransferRequest(BaseModel):
    assignee: str | None = Field(
        default=None,
        max_length=128,
        description="目标客服的用户名或 principal id；传 `null` 表示放回队列。",
        examples=["carol", None],
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
    handoff: HandoffInfo | None = Field(
        default=None,
        description="这一轮转人工落的工单。**`decision=human` 但没有工单**说明当前环境"
                    "没开通人工客服（未登录 / 演示模式），`answer` 里会说清。",
    )


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


def require_agent(principal: Annotated[dict, Depends(require_access)]) -> dict:
    """
    agent 或 admin 才能看队列、认领、回复。

    角色只在**开了身份层**时才检查：没开时 `require_access` 返回的匿名
    principal 是 admin（历史遗留的"API_KEY 对了就行"语义），在这里翻成 403
    会让人以为角色配错了。真正的"这个能力在这个环境不存在"由端点里的
    `_require_tickets_enabled()` 统一给 404。
    """
    if auth_settings.AUTH_ENABLED and not auth_store.has_role(principal, "agent"):
        raise HTTPException(status_code=403, detail="需要 agent 或 admin 角色")
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


def _call_answer_stream(question: str, history: list, customer_id: str,
                        on_usage=None):
    """
    兼容旧的、只接受两个参数的 `answer_stream` 替身（测试里常见）。

    真实实现多了 `customer_id` 和 `on_usage`；按签名逐个传，而不是硬编码
    关键字，避免替身函数在收到陌生参数时直接炸掉。
    """
    params = inspect.signature(answer_stream).parameters
    kwargs = {}
    if "customer_id" in params:
        kwargs["customer_id"] = customer_id
    if "on_usage" in params and on_usage is not None:
        kwargs["on_usage"] = on_usage
    return answer_stream(question, history, **kwargs)


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


def _client_ip(request: Request) -> str:
    """
    取限流用的客户端 IP。

    默认只信 TCP 对端（`request.client.host`）。`X-Forwarded-For` 是客户端
    可以随便写的，默认采信等于把限流键交给攻击者——每个请求换一个假 IP 就能
    绕过。只有部署在**自己控制的反向代理**后面、并显式设置
    `TRUST_PROXY_HEADERS=1` 时才读它。

    `X-Forwarded-For` 取最后一跳：代理会把自己看到的对端追加在末尾，最左边
    才是客户端自报的（可伪造）。前提是只有一层代理；多层时请在代理上覆盖
    `X-Real-IP`，而不是让这个头一路透传。
    """
    if auth_settings.TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for", "")
        hops = [hop.strip() for hop in forwarded.split(",") if hop.strip()]
        if hops:
            return hops[-1]
        real = request.headers.get("x-real-ip", "").strip()
        if real:
            return real
    return request.client.host if request.client else "unknown"


def _demo_precheck(request: Request) -> None:
    """
    每次问答前的两道闸：限流、每日预算。**都在真正调模型之前**——拦在调用
    之后等于钱已经花了。演示模式没开时它是空操作。
    """
    if not demo.enabled():
        return
    client = _client_ip(request)
    if not demo.allow_request(f"ip:{client}"):
        raise HTTPException(
            status_code=429,
            detail="请求过于频繁，请稍后再试",
            headers={"Retry-After": "60"},
        )
    if demo.budget_exhausted():
        raise HTTPException(status_code=429, detail="今天的演示额度已经用完")


def _chat_rate_limit(principal: dict) -> None:
    """
    已登录账号的每人限流。**人工接手的轮次也保留**：那一轮虽然不花 token，
    但一封封往客服台里灌消息同样是一种滥用。匿名请求（`id` 为空）由 demo
    闸按 IP 管，这里直接放行。
    """
    principal_id = str(principal.get("id") or "").strip()
    if not principal_id:
        return
    if not chat_quota.allow_request(principal_id):
        raise HTTPException(
            status_code=429,
            detail="请求过于频繁，请稍后再试",
            headers={"Retry-After": "60"},
        )


def _chat_budget_check(principal: dict) -> None:
    """
    已登录账号的每日 token 预算 + 全站每日总预算。

    **只在真的会调模型之前调用**。人工已经接手的会话不进模型，这一轮没有
    成本，不能因为"今日额度用完"把用户挡在人工客服门外——那样用户明明有
    人接待，却连一句话都发不出去。顺序上的细节见 `chat()`。

    匿名请求（`id` 为空）直接放行——它们由 demo 闸按 IP 管。
    """
    principal_id = str(principal.get("id") or "").strip()
    if not principal_id:
        return
    try:
        personal_exhausted = chat_quota.budget_exhausted(principal_id)
        global_exhausted = chat_quota.global_budget_exhausted()
    except chat_quota.LedgerUnavailable:
        # fail closed：读不出用量就不放行，见 ecom/chat_quota.py 的模块说明。
        raise HTTPException(
            status_code=503, detail="用量账本暂时不可用，请稍后再试"
        ) from None
    if personal_exhausted:
        raise HTTPException(status_code=429, detail="今天的个人 token 额度已经用完")
    if global_exhausted:
        raise HTTPException(status_code=429, detail="今天的全站 token 额度已经用完")


def _record_chat_usage(principal: dict, final: dict) -> None:
    """记账失败不影响这次回答（`add_tokens` 内部只记日志）。"""
    chat_quota.add_tokens(
        str(principal.get("id") or "").strip(), final.get("tokens") or {}
    )


# 登录限流：进程内滑动窗口。它挡的是"对着一台实例猜密码"，不是分布式撞库；
# 真要跨副本，得把这本账挪到共享存储（和 demo 的预算账本同一个道理）。
_LOGIN_MAX_ATTEMPTS = 10
# 账号维度放宽一倍：只按 IP 挡不住换 IP 打同一个账号，但账号维度太紧又会
# 变成"知道用户名就能把别人锁在门外"的拒绝服务。60 秒窗口让这个代价可控。
_LOGIN_ACCOUNT_MAX_ATTEMPTS = 20
_LOGIN_WINDOW_SECONDS = 60.0
_login_hits: dict[str, list[float]] = {}
_login_lock = threading.Lock()


def _login_precheck(request: Request, username: str) -> None:
    """
    按 IP 和账号两个桶同时限流，两个都过才放行。

    只按 IP：攻击者换一批出口 IP，每个 IP 试几次。
    只按账号：同一个 IP 换着用户名扫，仍然能撞库。
    """
    buckets = [(f"ip:{_client_ip(request)}", _LOGIN_MAX_ATTEMPTS)]
    account = str(username or "").strip().lower()
    if account:
        buckets.append((f"user:{account}", _LOGIN_ACCOUNT_MAX_ATTEMPTS))

    now = time.monotonic()
    with _login_lock:
        # 先全部剪枝再判断：否则某一次被拒会让另一个桶的旧记录一直留着。
        for key, _limit in buckets:
            _login_hits[key] = [
                stamp
                for stamp in _login_hits.get(key, [])
                if now - stamp < _LOGIN_WINDOW_SECONDS
            ]
        for key, limit in buckets:
            if len(_login_hits[key]) >= limit:
                raise HTTPException(
                    status_code=429,
                    detail="登录尝试过于频繁，请稍后再试",
                    headers={"Retry-After": "60"},
                )
        for key, _limit in buckets:
            _login_hits[key].append(now)
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
def login(req: LoginRequest, request: Request, response: Response) -> dict:
    if not auth_settings.AUTH_ENABLED:
        # 404 而不是 403：没开身份层时，这个端点"不存在"比"存在但不给用"更准确。
        raise HTTPException(status_code=404, detail="身份层未启用（AUTH_ENABLED=0）")
    _login_precheck(request, req.username)
    principal = auth_store.verify_password(req.username, req.password)
    if principal is None:
        # 不区分"用户不存在"和"密码错"：区分了就是一个用户名枚举接口。
        raise HTTPException(status_code=401, detail="用户名或密码不正确")
    issued = auth_store.issue_session(principal)
    # 登录响应里有 bearer token，不能进浏览器/代理的缓存。
    response.headers["Cache-Control"] = "no-store"
    return {
        "token": issued["token"],
        "expires_at": issued["expires_at"],
        "name": principal["name"],
        "role": principal["role"],
    }


@app.post(
    "/auth/logout",
    response_model=LogoutResponse,
    tags=["auth"],
    summary="吊销当前登录 token",
    description="让这枚 token **立刻失效**，而不是等它 12 小时后自然过期。"
                "只结束登录会话；长期 API token 不在这里处理（用 CLI 的 "
                "`rotate` 轮换）。没开 `AUTH_ENABLED` 时返回 404，和 `/auth/login` 一致。",
    responses={
        401: {"model": ErrorResponse, "description": "token 缺失或无效"},
        404: {"model": ErrorResponse, "description": "身份层未启用"},
    },
)
def logout(
    authorization: str = Header(default="", description="`Bearer <token>`"),
    x_api_key: str = Header(default="", description="兼容旧的 X-API-Key 用法。"),
) -> dict:
    if not auth_settings.AUTH_ENABLED:
        raise HTTPException(status_code=404, detail="身份层未启用（AUTH_ENABLED=0）")

    token = _bearer_token(authorization) or str(x_api_key or "").strip()
    principal = auth_store.authenticate(token)
    if principal is None:
        # 已经失效的 token 再来登出也按 401：不能拿它探测"这枚 token 存在过"。
        raise HTTPException(
            status_code=401,
            detail="需要有效的 bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    revoked = auth_store.revoke_token(token)
    auth_store.audit(
        principal.get("name", ""),
        "auth.logout",
        principal.get("name", ""),
        "revoked" if revoked else "long-lived-token",
    )
    return {"revoked": revoked}


def _history_owner(principal: dict) -> str:
    """
    历史接口的前置：必须开了身份层，且确实有一个 principal id。

    没开身份层时返回 **404**，和 `/auth/login` 一致——"这个能力不存在"比
    "存在但不给你"更准确。不能放行：匿名模式下所有人的 `owner_id` 都是空串，
    一旦列出来就是把所有人的匿名会话混在一起。
    """
    if not auth_settings.AUTH_ENABLED:
        raise HTTPException(status_code=404, detail="身份层未启用（AUTH_ENABLED=0）")
    owner = str(principal.get("id") or "").strip()
    if not owner:
        raise HTTPException(status_code=404, detail="当前凭证没有账号身份")
    return owner


def _session_title(text: str) -> str:
    """标题只取第一条提问的前 60 个字符，折叠空白；空会话给一个稳定的名字。"""
    clean = " ".join(str(text or "").split())
    if not clean:
        return "未命名会话"
    return clean[:60] + ("…" if len(clean) > 60 else "")


@app.get(
    "/sessions",
    response_model=SessionListResponse,
    tags=["sessions"],
    summary="当前账号的会话列表",
    description="登录后回看自己的历史会话，最近更新的在前。**只返回自己的**；"
                "没开 `AUTH_ENABLED` 时这个端点不存在（404）。\n\n"
                "标题取第一条用户提问，正文不在这里——点进某个会话再拉消息明细，"
                "避免侧边栏一次把整库正文都搬出来。",
    responses={
        401: {"model": ErrorResponse, "description": "需要 bearer token"},
        404: {"model": ErrorResponse, "description": "身份层未启用"},
    },
)
def list_my_sessions(
    principal: Annotated[dict, Depends(require_access)], limit: int = 20
) -> dict:
    owner = _history_owner(principal)
    limit = max(1, min(int(limit), 100))
    rows = session.list_sessions(limit=limit, owner_id=owner)
    return {
        "sessions": [
            {
                "session_id": row["session_id"],
                "title": _session_title(row.get("title")),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "messages": row["messages"],
            }
            for row in rows
            # 空会话（建了但一轮都没答完）不进列表：点进去只有空白，
            # 而且它通常是某次失败请求留下的，不是用户想回看的东西。
            if row["messages"] > 0
        ]
    }


@app.get(
    "/sessions/{session_id}",
    response_model=SessionDetail,
    tags=["sessions"],
    summary="一个会话的消息明细",
    description="按时间正序返回最近 200 条消息。**只有会话所有者能读**；"
                "不存在和不属于当前账号都按 404 处理，不暴露「存在但没权限」。",
    responses={
        401: {"model": ErrorResponse, "description": "需要 bearer token"},
        404: {"model": ErrorResponse, "description": "身份层未启用，或会话不存在 / 不属于当前账号"},
    },
)
def my_session(
    session_id: str,
    principal: Annotated[dict, Depends(require_access)],
) -> dict:
    owner = _history_owner(principal)
    meta = session.get_session(session_id)
    # 严格比对 owner_id，**不认领**：读历史不应产生"把别人的旧会话收归己有"
    # 的副作用。`_session_for()` 那套认领逻辑只属于写入路径。
    if meta is None or (meta.get("owner_id") or "") != owner:
        raise HTTPException(status_code=404, detail="会话不存在或不属于当前账号")
    rows = session.transcript(session_id)
    title = _session_title(session.first_question(session_id))
    return {
        "session_id": session_id,
        "title": title,
        "created_at": meta["created_at"],
        "updated_at": meta["updated_at"],
        "messages": rows,
    }


@app.delete(
    "/sessions/{session_id}",
    tags=["sessions"],
    summary="删除自己的一个会话",
    description="连同消息和记忆摘要一起删。**只删自己的**；演示环境不提供删除。"
                "删除后会立即从服务端列表消失，不只是前端本地移除。",
    responses={
        401: {"model": ErrorResponse, "description": "需要 bearer token"},
        404: {"model": ErrorResponse, "description": "身份层未启用 / 演示环境 / 会话不属于当前账号"},
    },
)
def delete_my_session(
    session_id: str,
    principal: Annotated[dict, Depends(require_access)],
) -> dict:
    owner = _history_owner(principal)
    if demo.enabled():
        # 公网演示是只读的：连"删自己的"也不开，避免演示数据被一个访客清空。
        raise HTTPException(status_code=404, detail="演示环境不提供删除")
    removed = session.delete_session_owned(session_id, owner)
    if removed is None:
        raise HTTPException(status_code=404, detail="会话不存在或不属于当前账号")
    return {"deleted": removed}


# ═══════════════════════════════════════════════════════════════════════════
# 人工工单
#
# 【为什么工单在这个文件里，而不是在编排层】
# 编排层能判断"该转人工"，但它不知道**是谁**在等：身份、角色、队列都在
# API 这一层。所以分工是：图给出 `decision=human`，这里落工单、补工单号。
#
# 【为什么没登录就没有工单】
# 匿名模式的会话 `owner_id` 是空串，"谁在排队"没有答案；而客户端的
# session_id 是凭证，没法和人对应。所以转人工在没登录时只给一句说明——
# 不伪造一个"已转接"的承诺。
# ═══════════════════════════════════════════════════════════════════════════
NO_HANDOFF_ANSWER = (
    "当前环境没有开通人工客服（需要登录账号）。"
    "你可以换一种问法，我会尽量用现有信息回答。"
)


def _require_tickets_enabled() -> None:
    """演示环境 / 没开身份层时，工单这套能力**不存在**（404，不是 403）。"""
    if demo.enabled():
        raise HTTPException(status_code=404, detail="演示环境不提供人工客服")
    if not auth_settings.AUTH_ENABLED:
        raise HTTPException(
            status_code=404, detail="身份层未启用（AUTH_ENABLED=0），没有人可以接待"
        )


def _ticket_or_error(fn, *args, **kwargs):
    """把工单模块的异常翻成 HTTP：不存在 404、状态冲突 409、参数不对 422。"""
    try:
        return fn(*args, **kwargs)
    except tickets.TicketNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except tickets.TicketConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except tickets.TicketError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


def _ticket_info(ticket: dict) -> dict:
    return {
        "ticket_id": ticket["ticket_id"],
        "status": ticket["status"],
        "reason": ticket.get("reason") or "",
        "created_at": ticket["created_at"],
        "updated_at": ticket["updated_at"],
        "claimed_by_name": ticket.get("claimed_by_name"),
        "queue_position": ticket.get("queue_position") or 0,
    }


def _ticket_summary(ticket: dict) -> dict:
    return {
        **_ticket_info(ticket),
        "session_id": ticket["session_id"],
        "title": _session_title(ticket.get("title") or session.first_question(ticket["session_id"])),
    }


def _attach_handoff(final: dict, session_id: str, principal: dict) -> dict | None:
    """
    需要人工时落一张工单，并把工单号写进答复。

    **在落盘之前调用**：`answer` 会跟着 `append_turn` 写进会话，用户下次
    重新打开历史时还能看到工单号。
    """
    if final.get("decision") != "human":
        return None

    owner = str(principal.get("id") or "").strip()
    ticket = None
    if not demo.enabled() and auth_settings.AUTH_ENABLED and owner:
        try:
            ticket = tickets.open_ticket(
                session_id, owner, final.get("decision_reason") or ""
            )
        except Exception:  # noqa: BLE001 —— 工单开不出来不该让这轮答复变成 500
            logger.exception("创建人工工单失败 session=%s", session_id)

    if ticket is None:
        final["answer"] = NO_HANDOFF_ANSWER
        return None

    position = ticket.get("queue_position") or 0
    if ticket.get("status") == "claimed":
        final["answer"] = (
            f"已为你转接人工客服，工单号 {ticket['ticket_id']}，"
            f"客服 {ticket.get('claimed_by_name') or ''} 已经接手。"
        )
    else:
        final["answer"] = (
            f"已为你转接人工客服，工单号 {ticket['ticket_id']}，"
            f"当前排在第 {position + 1} 位。人工回复会出现在这个对话里。"
        )
    return {
        "ticket_id": ticket["ticket_id"],
        "status": ticket["status"],
        "queue_position": position,
    }


def _human_takeover_ticket(session_id: str, principal: dict) -> dict | None:
    """
    这个会话是不是已经由人工接手了（`status='claimed'`）。

    排队中（`open`）仍然让机器人答：客服还没接，用户至少能拿到点什么。
    **一旦有人认领，机器人就闭嘴**——否则用户会同时收到机器人和客服两套说法，
    而客服看到的上下文里还夹着机器人自说自话的答复。

    读工单失败时返回 None（继续让机器人答）：这不是成本闸，数据库抖一下
    不该把用户晾在那里；真正的成本闸在配额那两层。
    """
    if demo.enabled() or not auth_settings.AUTH_ENABLED:
        return None
    if not str(principal.get("id") or "").strip():
        return None
    try:
        ticket = tickets.ticket_for_session(session_id)
    except Exception:  # noqa: BLE001 —— 见 docstring
        logger.exception("读取会话工单失败 session=%s", session_id)
        return None
    if ticket is not None and ticket.get("status") == "claimed":
        return ticket
    return None


def _human_mode_answer(ticket: dict) -> str:
    """人工接手后代替模型的那句话。说清三件事：谁在接、单号、消息去哪了。"""
    who = str(ticket.get("claimed_by_name") or "").strip()
    head = f"人工客服 {who} 已经接手" if who else "人工客服已经接手"
    return (
        f"{head}（工单 {ticket['ticket_id']}）。你的这条消息已经记下并转给客服，"
        "回复会出现在这个对话里；客服结束工单之前，机器人不会代为回答。"
    )


@app.post(
    "/sessions/{session_id}/handoff",
    response_model=TicketStatusResponse,
    tags=["tickets"],
    summary="给自己的会话开一张人工工单",
    description="幂等：同一会话已有未关闭的工单时，原样返回那一张。"
                "需要 `AUTH_ENABLED=1`、会话属于当前账号，且不在演示模式。",
    responses={
        401: {"model": ErrorResponse, "description": "需要 bearer token"},
        404: {"model": ErrorResponse, "description": "身份层未启用 / 演示环境 / 会话不属于当前账号"},
    },
)
def open_handoff(
    session_id: str, principal: Annotated[dict, Depends(require_access)]
) -> dict:
    _require_tickets_enabled()
    owner = _history_owner(principal)
    meta = session.get_session(session_id)
    if meta is None or (meta.get("owner_id") or "") != owner:
        raise HTTPException(status_code=404, detail="会话不存在或不属于当前账号")
    ticket = tickets.open_ticket(session_id, owner, "用户主动要求人工客服")
    if ticket is None:
        raise HTTPException(status_code=404, detail="会话不存在或不属于当前账号")
    return {"ticket": _ticket_info(ticket)}


@app.get(
    "/sessions/{session_id}/ticket",
    response_model=TicketStatusResponse,
    tags=["tickets"],
    summary="我的会话当前的人工工单状态",
    description="返回这个会话最近的一张工单（包括已关闭的）；从没转过人工时 `ticket` 是 `null`。"
                "前端靠它轮询人工回复，以及显示「已结束」而不是像什么都没发生过。",
    responses={
        401: {"model": ErrorResponse, "description": "需要 bearer token"},
        404: {"model": ErrorResponse, "description": "身份层未启用 / 演示环境 / 会话不属于当前账号"},
    },
)
def my_ticket(
    session_id: str, principal: Annotated[dict, Depends(require_access)]
) -> dict:
    _require_tickets_enabled()
    owner = _history_owner(principal)
    meta = session.get_session(session_id)
    if meta is None or (meta.get("owner_id") or "") != owner:
        raise HTTPException(status_code=404, detail="会话不存在或不属于当前账号")
    ticket = tickets.ticket_for_session(session_id)
    return {"ticket": _ticket_info(ticket) if ticket else None}


@app.get(
    "/tickets",
    response_model=TicketListResponse,
    tags=["tickets"],
    summary="人工排队列表（agent / admin）",
    description="`status=open`（默认）看等待中的，`claimed` 看进行中的，`closed` 看已关闭的，"
                "`all` 看全部。排队中的按先来后到，已关闭的按最近关闭在前。",
    responses={
        401: {"model": ErrorResponse, "description": "需要 bearer token"},
        403: {"model": ErrorResponse, "description": "需要 agent 或 admin 角色"},
        404: {"model": ErrorResponse, "description": "身份层未启用 / 演示环境"},
    },
)
def ticket_queue(
    principal: Annotated[dict, Depends(require_agent)],
    status: str = "open",
    limit: int = 50,
) -> dict:
    _require_tickets_enabled()
    is_admin = auth_store.has_role(principal, "admin")
    rows = _ticket_or_error(
        tickets.list_tickets,
        status=status,
        limit=limit,
        viewer_id=str(principal.get("id") or ""),
        is_admin=is_admin,
    )
    return {"tickets": [_ticket_summary(row) for row in rows]}


@app.get(
    "/tickets/{ticket_id}",
    response_model=TicketDetail,
    tags=["tickets"],
    summary="工单详情 + 用户会话的完整上下文（agent / admin）",
    description="客服接手前要看到前因：返回工单状态和这个会话最近 200 条消息。"
                "**每次查看都写一条审计**——这是在读用户说过的话。",
    responses={
        401: {"model": ErrorResponse, "description": "需要 bearer token"},
        403: {"model": ErrorResponse, "description": "需要 agent 或 admin 角色"},
        404: {"model": ErrorResponse, "description": "工单不存在 / 身份层未启用 / 演示环境"},
    },
)
def ticket_detail(
    ticket_id: str, principal: Annotated[dict, Depends(require_agent)]
) -> dict:
    _require_tickets_enabled()
    ticket = _ticket_or_error(tickets.get_ticket, ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"没有这个工单：{ticket_id}")
    is_admin = auth_store.has_role(principal, "admin")
    if not tickets.can_view(ticket, str(principal.get("id") or ""), is_admin):
        # 列表已经按同一口径过滤；这是"绕过列表直接拿 ticket_id 查"的那条路。
        raise HTTPException(status_code=403, detail="只能查看排队中的或自己处理的工单")
    auth_store.audit(
        principal.get("name", ""), "ticket.view", ticket_id, f"session={ticket['session_id']}"
    )
    return {
        **_ticket_summary(ticket),
        "transcript": session.transcript(ticket["session_id"], 200),
    }


@app.post(
    "/tickets/{ticket_id}/claim",
    response_model=TicketStatusResponse,
    tags=["tickets"],
    summary="认领工单（agent / admin）",
    description="只有 `open` 的工单能被认领；被别人认领会返回 409。"
                "重复认领自己那张是幂等的。",
    responses={
        401: {"model": ErrorResponse, "description": "需要 bearer token"},
        403: {"model": ErrorResponse, "description": "需要 agent 或 admin 角色"},
        404: {"model": ErrorResponse, "description": "工单不存在"},
        409: {"model": ErrorResponse, "description": "工单已关闭 / 已被别人认领"},
    },
)
def claim_ticket(
    ticket_id: str, principal: Annotated[dict, Depends(require_agent)]
) -> dict:
    _require_tickets_enabled()
    ticket = _ticket_or_error(
        tickets.claim_ticket, ticket_id, principal.get("id", ""), principal.get("name", "")
    )
    auth_store.audit(principal.get("name", ""), "ticket.claim", ticket_id, "")
    return {"ticket": _ticket_info(ticket)}


@app.post(
    "/tickets/{ticket_id}/reply",
    tags=["tickets"],
    summary="以人工客服身份回复（agent / admin）",
    description="**先认领才能回**。回复写进用户的会话（`role=agent`），用户下次打开"
                "或轮询工单状态时能看到；模型下一轮也会把它当作上一轮说过的话。",
    responses={
        401: {"model": ErrorResponse, "description": "需要 bearer token"},
        403: {"model": ErrorResponse, "description": "需要 agent 或 admin 角色"},
        404: {"model": ErrorResponse, "description": "工单不存在"},
        409: {"model": ErrorResponse, "description": "工单已关闭 / 不是由你认领的"},
        422: {"model": ErrorResponse, "description": "回复为空或太长"},
    },
)
def reply_ticket(
    ticket_id: str,
    req: ReplyRequest,
    principal: Annotated[dict, Depends(require_agent)],
) -> dict:
    _require_tickets_enabled()
    result = _ticket_or_error(
        tickets.reply_ticket,
        ticket_id,
        principal.get("id", ""),
        principal.get("name", ""),
        req.content,
    )
    # 审计里留正文前 120 字：出问题时要知道"客服当时到底回了什么"。
    auth_store.audit(
        principal.get("name", ""), "ticket.reply", ticket_id, req.content[:120]
    )
    return {"ticket_id": ticket_id, **result}


@app.post(
    "/tickets/{ticket_id}/close",
    response_model=TicketStatusResponse,
    tags=["tickets"],
    summary="关闭工单（agent / admin）",
    description="默认只有认领人能关；**admin 可以强制关闭**（写审计）。重复关闭是幂等的。"
                "关闭后这个会话再要人工会新建一张单。",
    responses={
        401: {"model": ErrorResponse, "description": "需要 bearer token"},
        403: {"model": ErrorResponse, "description": "需要 agent 或 admin 角色"},
        404: {"model": ErrorResponse, "description": "工单不存在"},
        409: {"model": ErrorResponse, "description": "不是由你认领的，且你没有 admin 角色"},
    },
)
def close_ticket(
    ticket_id: str, principal: Annotated[dict, Depends(require_agent)]
) -> dict:
    _require_tickets_enabled()
    agent_id = str(principal.get("id") or "")
    is_admin = auth_store.has_role(principal, "admin")
    before = _ticket_or_error(tickets.get_ticket, ticket_id)
    ticket = _ticket_or_error(
        tickets.close_ticket, ticket_id, agent_id, allow_override=is_admin
    )
    override = bool(is_admin and before and before.get("claimed_by") != agent_id)
    auth_store.audit(
        principal.get("name", ""),
        "ticket.close",
        ticket_id,
        "admin-override" if override else "",
    )
    return {"ticket": _ticket_info(ticket)}


@app.post(
    "/tickets/{ticket_id}/transfer",
    response_model=TicketStatusResponse,
    tags=["tickets"],
    summary="转派工单或放回队列（admin）",
    description="把工单转给另一个客服；`assignee=null` 表示放回排队。"
                "用来处理认领人离职 / 账号被吊销 / 长期不关单。每次转派都写审计。",
    responses={
        401: {"model": ErrorResponse, "description": "需要 bearer token"},
        403: {"model": ErrorResponse, "description": "需要 admin 角色"},
        404: {"model": ErrorResponse, "description": "工单不存在"},
        409: {"model": ErrorResponse, "description": "工单已关闭"},
        422: {"model": ErrorResponse, "description": "目标账号不存在 / 已吊销 / 不是客服"},
    },
)
def transfer_ticket(
    ticket_id: str,
    req: TicketTransferRequest,
    principal: Annotated[dict, Depends(require_admin)],
) -> dict:
    _require_tickets_enabled()
    target = None
    if req.assignee:
        target = auth_store.get_principal(req.assignee)
        if target is None or target.get("revoked_at"):
            raise HTTPException(status_code=422, detail="目标账号不存在或已吊销")
        if not auth_store.has_role(target, "agent"):
            raise HTTPException(status_code=422, detail="目标账号不是客服")
    ticket = _ticket_or_error(tickets.transfer_ticket, ticket_id, target)
    auth_store.audit(
        principal.get("name", ""),
        "ticket.transfer",
        ticket_id,
        target["name"] if target else "queue",
    )
    return {"ticket": _ticket_info(ticket)}


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
        429: {"model": ErrorResponse, "description": "请求太频繁，或今日演示 / 个人额度已用完"},
        503: {"model": ErrorResponse, "description": "账号用量账本暂时不可用（fail closed）"},
    },
)
def chat(
    req: ChatRequest, request: Request, principal: Annotated[dict, Depends(require_access)]
) -> dict:
    _demo_precheck(request)
    _chat_rate_limit(principal)

    # 先确保会话存在再读历史——顺序反过来的话，新会话的第一问会因为
    # "查不到这个 session_id"而拿不到空历史（这一版不会报错，只是白查一次）。
    session_id = _session_for(req.session_id, principal)

    # 人工已经接手：这一轮不进模型。用户的话仍然落盘，客服下一次刷新能看到；
    # 返回的 `decision=human` 和工单信息让界面维持在"人工服务中"。
    claimed = _human_takeover_ticket(session_id, principal)
    if claimed is not None:
        answer = _human_mode_answer(claimed)
        session.append_turn(
            session_id, req.question, answer, owner_id=principal.get("id", "")
        )
        return {
            "handoff": _ticket_info(claimed),
            "request_id": trace.new_request_id(),
            "session_id": session_id,
            "answer": answer,
            "sources": [],
            "decision": "human",
            "plan": [],
            "tokens": {},
        }

    # 预算闸放在认领判断之后：人工接手的那一轮没有模型调用，也就没有成本。
    # 如果它在前面，"今日额度用完"会连人工客服一起挡掉——用户明明有人接待，
    # 却一个字都发不出去。限流已经在上面单独做过，刷屏的成本仍然兜住。
    _chat_budget_check(principal)

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
    _record_chat_usage(principal, final)
    handoff = _attach_handoff(final, session_id, principal)

    # **先答完再落盘**。反过来的话，模型调用失败也会留下一条"问了没答"的
    # 记录。`append_turn` 内部一问一答是同一个提交，所以这里要么成对写入，
    # 要么一行都不写。
    session.append_turn(
        session_id, req.question, final.get("answer") or "", owner_id=principal.get("id", "")
    )

    return {
        "handoff": handoff,
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
# 事件只有六种，客户端按 `type` 分派即可：
#
#     {"type":"start", "request_id":"...", "session_id":"..."}
#     {"type":"thinking", "kind":"stage"|"reasoning", "text":"..."}
#     {"type":"delta", "text":"..."}
#     {"type":"reset"}  # 新一轮：丢掉上一轮已流出的正文
#     {"type":"final", "request_id":"...", "session_id":"...", "answer":"...",
#      "sources":[...], "decision":"...", "plan":[...], "tokens":{...},
#      "handoff":{"ticket_id":"...","status":"open","queue_position":0}|null,
#      "thinking":"...", "elapsed_ms":123}
#     {"type":"error", "error":"...", "request_id":"..."}
#
# **`final.answer` 是权威版本。** 汇总流到一半失败时服务端会退回拼接，
# 前面收到的 delta 拼起来可能和它不一致——流式天生带这个窗口，不是 bug。
# `thinking` 只用于展示过程，**不参与**答案拼接；final 里的那份是完整副本，
# 客户端可以拿它覆盖自己累加的结果（掉帧时也不会缺一段）。
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
                "`start` / `thinking` / `delta` / `reset` / `final` / `error`。\n\n"
                "**`final.answer` 是权威版本**：汇总中途失败时会退回拼接，"
                "前面 `delta` 拼起来可能和它不一致。\n\n"
                "`thinking` 是过程展示：`stage` 来自路由/worker/决策的真实状态，"
                "`reasoning` 是模型返回的思维链（普通模型没有这一路）。它不参与"
                "答案拼接。\n\n"
                "真逐 token 有两条路：多个单元给出结论时的**汇总模型**，以及"
                "**单 worker** 时 worker 自己支持的 token 回调。没有可流的东西"
                "（直接转人工、模型不支持流）才作为一个 `delta` 整段到达——"
                "不把已完成的字符串切成小块假装在流。",
    dependencies=[Depends(require_demo_access)],
    responses={
        401: {"model": ErrorResponse, "description": "需要 bearer token（开了 AUTH_ENABLED 时）"},
        403: {"model": ErrorResponse, "description": "演示模式但邀请码不对"},
        429: {"model": ErrorResponse, "description": "请求太频繁，或今日演示 / 个人额度已用完"},
        503: {"model": ErrorResponse, "description": "账号用量账本暂时不可用（fail closed）"},
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
    _chat_rate_limit(principal)

    session_id = _session_for(req.session_id, principal)

    # 和 /chat 同一条规矩：人工接手后不再进模型，只把用户这句话记下来。
    # 流式接口仍然按同一套事件发出去，客户端不需要多一个分支。
    claimed = _human_takeover_ticket(session_id, principal)
    if claimed is not None:
        answer = _human_mode_answer(claimed)
        handoff_info = _ticket_info(claimed)
        request_id = trace.new_request_id()

        # **先落盘再开始流。** 这句话是给客服看的，用户看到"已经记下"之后
        # 立刻关掉页面也不能丢；放进生成器里的话，客户端断开会让生成器
        # 被取消，append_turn 可能根本跑不到。人工模式不调模型，先写没有成本。
        session.append_turn(
            session_id, req.question, answer, owner_id=principal.get("id", "")
        )

        def human_events():
            yield _sse(
                {
                    "type": "start",
                    "request_id": request_id,
                    "session_id": session_id,
                }
            )
            yield _sse({"type": "delta", "text": answer})
            yield _sse(
                {
                    "type": "final",
                    "request_id": request_id,
                    "session_id": session_id,
                    "answer": answer,
                    "sources": [],
                    "decision": "human",
                    "plan": [],
                    "tokens": {},
                    "handoff": handoff_info,
                    "thinking": "",
                    "elapsed_ms": 0,
                }
            )

        return StreamingResponse(
            human_events(), media_type="text/event-stream", headers=SSE_HEADERS
        )

    # 和 /chat 一样：预算闸在认领判断之后（人工接管不花 token）。
    _chat_budget_check(principal)

    history = session.history(session_id)
    # 和 /chat 一样：记忆每轮现拼，不落盘。
    augmented = memory.augmented_history(session_id, req.question, history)

    def events():
        request_id = ""
        final = None
        elapsed_ms = None
        thinking: list[str] = []
        usage_reported = False

        def bill(tokens: dict) -> None:
            """流式中断/正常结束都从这里记一次账，两条闸各记各的。"""
            nonlocal usage_reported
            usage_reported = True
            demo.add_tokens(tokens or {})
            chat_quota.add_tokens(
                str(principal.get("id") or "").strip(), tokens or {}
            )

        try:
            for event in _call_answer_stream(
                req.question,
                augmented,
                principal.get("customer_id") or "",
                on_usage=bill,
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
                elif kind == "thinking":
                    text = event.get("text") or ""
                    thinking.append(text)
                    yield _sse(
                        {
                            "type": "thinking",
                            "kind": event.get("kind") or "stage",
                            "text": text,
                        }
                    )
                elif kind == "delta":
                    yield _sse({"type": "delta", "text": event.get("text") or ""})
                elif kind == "reset":
                    yield _sse({"type": "reset"})
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

        # 真实 supervisor 会在收尾（含断线）通过 on_usage 记账；只有替身
        # 生成器不认识 on_usage 时才走这里兜底。
        if not usage_reported:
            bill(final.get("tokens") or {})

        # **拿到完整答复之后再落盘。** 中途断开的请求不写会话——半句话接进
        # 下一轮，比没答更坏。和 /chat 的"先答完再落盘"是同一条规矩。
        # 和 /chat 一样：先落工单、再落盘，写进历史的是带工单号的那句。
        handoff = _attach_handoff(final, session_id, principal)
        session.append_turn(
            session_id, req.question, final.get("answer") or "", owner_id=principal.get("id", "")
        )

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
                "handoff": handoff,
                "thinking": "".join(thinking),
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
