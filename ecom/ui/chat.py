"""对话主视图：会话初始化、图片输入、流式答复、决策链回放。

从 `app.py` 拆出来，是因为这条链路本身就有 300 行，而 `app.py` 还同时是
"装配 + 侧边栏 + 直连视图"的入口。拆开之后：

    app.py        只负责装配（把 session / demo / answer_stream 传进来）
    ecom/ui/chat.py       对话这一条链路的全部逻辑

**依赖通过 ChatContext 传**，不 import app：测试可以 monkeypatch
`app.session` / `app.answer_stream` 之后仍然生效，也避免循环导入。
"""
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from ecom.auth import settings as auth_settings
from ecom.orchestrator import memory
from ecom.orchestrator.tools.image_understander import understand_image
from ecom.orchestrator.utils.image_intent import augment_question_with_image
from ecom.settings import VISION_MAX_IMAGE_MB
from ecom.ui.components import render_decision_chain

logger = logging.getLogger(__name__)


def _st():
    # 延迟 import：假 streamlit / 真 streamlit 可以在同一个测试进程里先后生效。
    import streamlit as st

    return st


@dataclass
class ChatContext:
    session: object
    demo: object
    answer_stream: Callable


# 空态的三张建议卡。**刻意各打一条路**：订单 / 知识库 / 数据分析各一张，
# 点一张就走通一个 worker——这本身就是这套多 Agent 的演示，而"随便问点什么"
# 的空输入不会演示出任何东西。
_SUGGESTIONS = {
    ":material/local_shipping: 订单 52644 到哪了": "订单 52644 到哪了",
    ":material/assignment_return: 退换货政策是什么": "退换货政策是什么",
    ":material/bar_chart: 哪个地区卖得最好": "哪个地区卖得最好",
}

# 演示模式下**不能出现"哪个地区卖得最好"那张卡**：数据分析被停用了，点了
# 只会得到知识库的"资料里没有"。换成一条知识库能真答上来的问题。
_DEMO_SUGGESTIONS = {
    ":material/local_shipping: 订单 52644 到哪了": "订单 52644 到哪了",
    ":material/assignment_return: 退换货政策是什么": "退换货政策是什么",
    ":material/schedule: 退款一般多久到账": "退款一般多久到账",
}


def init_agent_session(ctx: ChatContext) -> None:
    """这个浏览器会话还没会话号时，开一个。

    **刻意不做"从 URL 里恢复会话"**（`st.query_params["session"] = ...`）。
    会话号是凭证：拿到它就能读到这个会话的全部历史。塞进 URL 之后它会进
    浏览器历史、代理访问日志、跨站请求的 Referer 头——凭证最不该待的三个地方。
    """
    st = _st()
    if "agent_session_id" not in st.session_state:
        st.session_state["agent_session_id"] = ctx.session.ensure_session("")
    st.session_state.setdefault("agent_turns", [])


def start_new_session(ctx: ChatContext) -> None:
    """换一个会话号重新开始。**不清空旧会话**——它还在盘上，由 TTL 和 CLI 管。"""
    st = _st()
    st.session_state["agent_session_id"] = ctx.session.ensure_session("")
    st.session_state["agent_turns"] = []


def render_agent_view(ctx: ChatContext) -> None:
    st = _st()
    init_agent_session(ctx)
    turns = st.session_state["agent_turns"]

    picked = _render_empty_state(ctx) if not turns else None

    for turn in turns:
        _render_turn(turn)

    uploaded, send_image_only = _render_image_uploader()

    question = st.chat_input(
        "问点什么…（订单 / 售后政策 / 销售数据都能问）", key="chat_agent"
    ) or picked
    if not question and not (uploaded is not None and send_image_only):
        return

    # **先过演示闸，再碰会花钱的图片理解。** 图片理解在 supervisor 之外
    # 单独调一次视觉模型，如果把这一步放在限流/预算检查之前，演示访客可以
    # 绕开次数限制反复上传图片烧钱。
    if not _demo_precheck(ctx):
        return

    display_question = question or "（图片）"
    if uploaded is not None:
        augmented = _prepare_image_question(ctx, uploaded, question or "")
        if augmented is None:
            return
        question = augmented
        st.session_state["image_upload_nonce"] = (
            st.session_state.get("image_upload_nonce", 0) + 1
        )

    if not question:
        return
    _run_turn(ctx, question, display_question=display_question, prechecked=True)
    if uploaded is not None:
        st.rerun()


def _render_empty_state(ctx: ChatContext) -> str | None:
    """空态：一句引导 + 三张建议卡。**返回被点的那句问题**，不自己往下跑。"""
    st = _st()
    st.markdown("### :material/waving_hand: 想聊点什么？")
    if ctx.demo.enabled():
        st.caption("这是只读演示：订单进度、售后政策都能问——直接打字，或者点一张卡试试。")
    else:
        st.caption("订单进度、售后政策、销售数据都能问——直接打字，或者点一张卡试试。")
    suggestions = _DEMO_SUGGESTIONS if ctx.demo.enabled() else _SUGGESTIONS
    picked = st.pills(
        "试试这样问",
        list(suggestions),
        selection_mode="single",
        key="chat_suggestion",
        label_visibility="collapsed",
    )
    return suggestions.get(picked) if picked else None


def _render_image_uploader():
    """画上传器，返回 (上传的文件, 是否点了「只发送图片」)。"""
    st = _st()
    nonce = st.session_state.setdefault("image_upload_nonce", 0)
    with st.expander(
        "上传图片（可选：购物截图 / 售后凭证 / 发票 / 快递面单）",
        icon=":material/image:",
    ):
        st.caption(
            f"支持 PNG / JPG / WEBP，最大 {VISION_MAX_IMAGE_MB:g}MB。"
            "图片只用于提取订单号、金额、商品和诉求，不会把原图写进会话。"
        )
        uploaded = st.file_uploader(
            "选择图片",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=False,
            key=f"chat_image_{nonce}",
        )
        send_image_only = (
            st.button("只发送图片", key=f"send_image_{nonce}", icon=":material/send:")
            if uploaded is not None
            else False
        )
    return uploaded, send_image_only


def _header(headers, name: str) -> str:
    """从 Streamlit 的 Headers / 普通 dict 里按大小写不敏感取一个头。"""
    if headers is None:
        return ""
    try:
        getter = getattr(headers, "get", None)
        if callable(getter):
            for key in (name, name.lower(), name.upper()):
                value = getter(key)
                if value:
                    return str(value)
        for key, value in headers.items():
            if str(key).lower() == name.lower():
                return str(value or "")
    except Exception:  # noqa: BLE001 —— 拿不到头就按"没有"处理
        return ""
    return ""


def _demo_client_key() -> str:
    """
    演示限流键：优先客户端 IP，拿不到再回退到浏览器会话。

    `session:<id>` 不能单独用：Streamlit 的 session 是按标签页/websocket 分的，
    用户新开一个标签页就换了一个桶，限流等于没做。`st.context.ip_address` 在
    直连时就是 TCP 对端地址；部署在反代后面时，只有显式打开
    `TRUST_PROXY_HEADERS` 才读 `X-Forwarded-For` —— 否则客户端可以自己伪造，
    把"按 IP 限流"变成"按我喜欢的任意字符串限流"。

    同一 NAT 后面的访客会共用一个桶，这是演示模式的已知取舍：它比"每个标签页
    一个桶"更接近文档里承诺的"按客户端"；真要区分人，得先有身份。
    """
    st = _st()
    try:
        context = getattr(st, "context", None)
        if auth_settings.TRUST_PROXY_HEADERS:
            for header in ("X-Forwarded-For", "X-Real-IP"):
                value = _header(getattr(context, "headers", None), header)
                # Caddy 会把真实对端追加在末尾；最左边是客户端自报的，可伪造。
                client = value.split(",")[-1].strip() if value else ""
                if client:
                    return f"ip:{client}"
        ip = str(getattr(context, "ip_address", "") or "").strip()
        if ip:
            return f"ip:{ip}"
    except Exception:  # noqa: BLE001 —— 上下文拿不到就回退，不影响问答
        logger.debug("读取 Streamlit 客户端上下文失败", exc_info=True)
    sid = str(st.session_state.get("agent_session_id") or "").strip()
    return f"session:{sid or 'anonymous'}"


def _demo_precheck(ctx: ChatContext) -> bool:
    """演示模式的次数和预算闸。**任何会花钱的动作之前都要先过这里。**"""
    if not ctx.demo.enabled():
        return True
    st = _st()
    if not ctx.demo.allow_request(_demo_client_key()):
        st.warning("问得有点快，等一分钟再试。")
        return False
    if ctx.demo.budget_exhausted():
        st.error("今天的演示额度已经用完，明天再来。")
        return False
    return True


def _call_answer_stream(ctx: ChatContext, question: str, history: list, on_usage):
    """兼容只接受两个参数的测试替身：真实实现才认识 `on_usage`。"""
    if "on_usage" in inspect.signature(ctx.answer_stream).parameters:
        return ctx.answer_stream(question, history, on_usage=on_usage)
    return ctx.answer_stream(question, history)


def _prepare_image_question(ctx: ChatContext, uploaded, question: str) -> str | None:
    """识别上传的图片，拼成主链路能读的问题；失败时返回 None 或原始文字。"""
    st = _st()
    try:
        data = uploaded.getvalue()
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取上传图片失败：%s", exc)
        st.warning("读取图片失败，请重新选择或手动输入问题和订单号。")
        return (question or "").strip() or None

    mime = getattr(uploaded, "type", "") or "image/png"
    with st.status(":shimmer[正在识别图片…]", type="compact", state="running") as status:
        intent = understand_image(data, mime)
        if intent.ok:
            status.update(label="图片识别完成", state="complete")
        else:
            status.update(label="图片识别失败", state="error")

    if not intent.ok:
        logger.warning("图片识别失败：%s", intent.error)
        st.warning("图片识别失败，请手动输入订单号或问题（详细原因已写入日志）。")
        return (question or "").strip() or None

    # 图片理解发生在主链路之前，它的 token 不进 supervisor 的账；演示预算要单独记。
    ctx.demo.add_tokens(intent.tokens)

    augmented = augment_question_with_image(question or "", intent)
    if not augmented:
        st.warning(
            "图片里没有识别出明确的诉求，请再打一句话，例如「订单 52644 到哪了」。"
        )
        return None

    if intent.needs_manual_order_id:
        st.info(
            "图片里没有识别到订单号；如果要查订单或申请退货，"
            "请在消息里手动补充订单号。"
        )
    return augmented


def _render_turn(turn: dict) -> None:
    """把一轮问答铺出来。**历史回放和新问答共用这一段**。"""
    st = _st()
    with st.chat_message("user"):
        st.markdown(turn.get("display_question") or turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["final"].get("answer") or "（没有生成答复）")
        sources = turn["final"].get("sources") or []
        if sources:
            st.caption("来源：" + "、".join(str(s) for s in sources))
        render_decision_chain(turn["final"], turn["elapsed_ms"])


def _run_turn(
    ctx: ChatContext,
    question: str,
    display_question: str | None = None,
    prechecked: bool = False,
) -> None:
    """跑一轮问答。`question` 喂模型，`display_question` 画给用户。

    `prechecked=True` 表示调用方已经在图片理解之前过了演示闸；否则在这里补上，
    避免别处调用时漏掉限流和预算。
    """
    st = _st()
    sid = st.session_state["agent_session_id"]
    turns = st.session_state["agent_turns"]
    display_question = question if display_question is None else display_question

    if not prechecked and not _demo_precheck(ctx):
        return

    with st.chat_message("user"):
        st.markdown(display_question)

    with st.chat_message("assistant"):
        started = datetime.now()
        history = ctx.session.history(sid)
        augmented = memory.augmented_history(sid, question, history)

        status = st.status(":shimmer[正在路由与汇总…]", type="compact", state="running")
        answer_box = st.empty()
        pieces: list[str] = []
        final = None
        billed = False

        def bill(tokens: dict) -> None:
            """一份请求只记一次账：正常结束和生成器被关闭都由它兜住。"""
            nonlocal billed
            if billed:
                return
            billed = True
            ctx.demo.add_tokens(tokens or {})

        try:
            for event in _call_answer_stream(ctx, question, augmented, bill):
                kind = event.get("type")
                if kind == "delta":
                    if not pieces:
                        status.update(label=":shimmer[正在生成答复…]", state="running")
                    pieces.append(event.get("text") or "")
                    answer_box.markdown("".join(pieces) + "▌")
                elif kind == "reset":
                    # 重路由：上一轮的单 worker 直流可能已经画在屏幕上了，
                    # 那是中间产物；不清掉就会和下一轮的正文拼在一起。
                    pieces.clear()
                    answer_box.markdown("")
                elif kind == "final":
                    final = event.get("state") or {}
        except Exception:
            logger.exception("对话流式请求失败")
            status.update(label="答复失败", state="error")
            st.error("这次请求失败了，请稍后再试或换一种问法。")
            return

        elapsed_ms = int((datetime.now() - started).total_seconds() * 1000)
        status.update(label=f"已完成（{elapsed_ms} ms）", state="complete")
        if final is None:
            status.update(label="答复失败", state="error")
            st.error("这次请求没有正常结束，请重试。")
            return

        # 真实 supervisor 在收尾（含生成器被提前关闭）通过 on_usage 报过账；
        # 只有两参数替身不会回调，才用 final 里的汇总兜底。
        if not billed:
            bill(final.get("tokens") or {})

        answer_box.markdown(final.get("answer") or "（没有生成答复）")
        sources = final.get("sources") or []
        if sources:
            st.caption("来源：" + "、".join(str(s) for s in sources))
        render_decision_chain(final, elapsed_ms)

    # 先答完再落盘：模型调用失败时不会留下一条"问了没答"的记录。
    ctx.session.append_turn(sid, question, final.get("answer") or "")
    turns.append(
        {
            "question": question,
            "display_question": display_question,
            "final": final,
            "elapsed_ms": elapsed_ms,
        }
    )
