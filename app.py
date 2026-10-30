"""
统一界面（Streamlit）。**主区域是一段对话**，另外三块收成侧边栏里的「直连视图」。

    uv sync --extra ui
    uv run streamlit run app.py

【为什么不并排放三份 app.py】
合并前每个仓库各有一个 app.py：rag 那个问知识库，sales 那个做数据分析，
而编排层干脆没有界面（只有 main.py / api.py）。三个界面并排放在同一个 README
里，读到的人得自己拼出"它们其实是同一个系统"——而这件事本该由**结构**说出来。

【界面长什么样，以及为什么长这样】
主区域只有对话。订单 / 数据看板 / 知识库三块**没有消失**，它们变成了侧边栏
「直连视图」里的三个入口——点哪个，主区域就换成哪个。

这个形状是照 DeepSeek / 豆包的客户端定的，但不是为了"像"：
  · 一次提问由 Supervisor 路由到某条路，**结果本来就该出现在对话里**。
    让用户先想"这该点哪个 Tab"是把路由的活推给了人——而路由正是这套系统
    在做的事。所以主区域是对话。
  · 但另外三个面仍然要看得到、点得着：面试时人家第一眼要能看出这是**四个
    子系统合起来的一套**，而不是一个聊天框。所以它们在侧边栏常驻。
  · 最像 DeepSeek 的那一处是「决策链」折叠块——它和 DeepSeek 的「深度思考」
    在结构上是同一个东西：把这一次推理的每一步折起来，标题上给一句结论。
    也就是说这个界面上最好看的那块，恰好也是**最能说明架构**的那块。

【四个视图各自的调用方式，刻意不一致，理由是一致的】
    对话     → import `answer()`        进程内
    知识库   → import `retrieve/rerank/generate_answer`   进程内
    数据看板 → import `run_analytics()`  **子进程**
    订单     → import `query_order()`   进程内（本地 SQLite）

第三行和别人不一样，理由在 ecom/orchestrator/workers/analytics.py：数据分析要
执行模型生成的代码，需要一个能被真杀死的进程。**界面不该是那条边界的例外**：
在这里为了"少起一个进程"改成进程内调用，等于换了一个没有超时、崩了会带走
整个页面的执行环境，而收益只是省掉一次 Python 启动。所以宁可绕一圈。

【实时 / 回放：哪一段是真的，哪一段不是】
对话视图现在走 `supervisor.answer_stream()`：汇总模型那次调用是**真的逐 token
流出来的**，生成到哪就画到哪。但它只覆盖一块——**只有一个结论时没有汇总可做**，
最终答复会作为一整段到达；这时候界面会一次性显示完，**不把它切成小块假装在流**。
知识库视图本来就是真流（`generate_answer()` 的 `streaming.sink`），两条路现在
都是真的，只是覆盖的范围不同。

**最终答复以 `final` 事件为准**：汇总如果流到一半失败，服务端会退回拼接，前面
已经画出去的字和最后那段可能对不上。界面在收到 `final` 后用完整答复覆盖一次，
不是不信任流，是流式本身就带着这个窗口。

【界面会藏住什么】
超时变成一个转圈动画，异常变成一个红框，"哪一层慢、哪一层挂了"全看不见。
排查时该用的仍然是 `uv run python main.py -v`。界面是演示用的，CLI 是查问题
用的，这个顺序不能反。

【会话持久化——以前这里写的是"不做"】
这段以前写的是：「多轮历史放 `st.session_state`，关掉页面就没了。真要持久化，
正确的落点是 ecom/orchestrator 那层的会话存储，不是在界面里塞一个 sqlite。」

后半句就是现在做的事——落点在 `ecom/orchestrator/session.py`，界面只是它的
一个使用者（和 `api.py`、`python -m ecom.orchestrator.cli.session` 平级）。
这条没有变：**界面里没有自己的 sqlite**。

顺带修掉一个真问题：以前那份 `agent_history` 是**只增不减**的——界面上聊到
第 50 轮，就会把 50 轮全塞给模型。现在每次提问都从存储里现读
`session.history()`，截断由 `HISTORY_TURNS` 统一管。

**边界要说清，别把它说大**：界面这边落盘买到的是"上下文有上限、且这段对话
在 CLI 里看得见也删得掉"，**不是"换个标签页还能接着聊"**。会话号只活在
`st.session_state` 里，开一个新标签页就是一段新对话。想跨标签页续上，得把
会话号放进 URL——**而那是错的**：它是凭证，进了 URL 就会被浏览器历史、代理
日志、Referer 头带走（所以侧边栏里也只显示前 12 位）。

【图片输入：先提取，再进主链路】
对话视图支持上传一张图（购物截图 / 售后凭证 / 发票 / 快递面单）。图片**不直接
喂给对话主链路**：先经 `image_understander` 提取成结构化意图（订单号 / 金额 /
商品 / 诉求），拼成一段带标记的文本，再走原来的 `answer()`。识别失败、图里没有
订单号都只是降级成"请手动输入"，不会崩。
关键边界：图片里的订单号只用于**查**。退货是写操作，必须要求用户在当前消息里
手动输入订单号——理由见 `workers/order.py` 的模块 docstring。

【不做的事，写清楚比留白好】
  · 不做登录 / 权限：这是离线演示，加一套假登录只会让演示多几步点击。
    （**后台管理端不在这份文件里**——它要能批准退货，所以有密码门禁，
      见仓库根的 `admin.py`。两个入口分开，正是为了让这条"不做"继续成立。）
  · 不做多副本：会话落在本地 sqlite，**多副本各写各的文件，上下文会岔**。
    和 api.py 那边是同一个已声明的边界（要真解决得换 Redis 之类的共享存储）。
  · 不在这个文件里写 CSS。整个仓库唯一一处"样式"是 `.streamlit/config.toml`，
    理由是原生主题令牌跟着 Streamlit 版本走，而 CSS 选择器指向内部类名。
"""
import logging
from collections.abc import Callable
from datetime import datetime
from typing import NamedTuple

import streamlit as st

from ecom import demo
from ecom.knowledge.vector_store import get_vector_store
from ecom.orchestrator import memory, session
from ecom.orchestrator.core.supervisor import answer_stream
from ecom.orchestrator.db.engine import get_db
from ecom.orchestrator.settings import HISTORY_TURNS, MAX_ROUNDS, WORKER_TIMEOUT
from ecom.orchestrator.tools.image_understander import understand_image
from ecom.orchestrator.utils.image_intent import augment_question_with_image
from ecom.settings import VISION_MAX_IMAGE_MB
from ecom.ui import components, views
from ecom.ui.views import ViewContext

logger = logging.getLogger(__name__)

# layout="centered" 是"像 DeepSeek"最直接的一根杠杆：对话收成一条窄列。
# 代价如实地讲——订单页的表格和看板页的图也跟着变窄了。接受，因为这两个现在
# 是"从侧边栏点进去看的次级视图"，不再是并列的主入口。
st.set_page_config(
    page_title="多 Agent 电商智能客服",
    page_icon=":material/shopping_cart:",
    layout="centered",
)


# ——————————————————————————————————————————————————————————————
# 连接类对象要 cache_resource，不能每次交互重建。
#
# 这是 Streamlit 最贵的一个坑：脚本每交互一次就**从头重跑**。向量库客户端、
# 数据库连接直接写在校验里，就是每点一下按钮重连一次——而症状是"越用越慢"，
# 不会报错，所以很难联想到是这里。跨轮要留住的东西放 session_state（局部
# 变量每次都重来），连接类放 cache_resource（按参数缓存、进程内共用）。
# ——————————————————————————————————————————————————————————————
@st.cache_resource(show_spinner=False)
def _store():
    return get_vector_store()


@st.cache_resource(show_spinner=False)
def _db():
    return get_db()


# 渲染件和三个直连视图已经搬到 ecom/ui/；这里保留同名别名，调用点不用动。
_clip = components.clip
_backend_label = components.backend_label
_render_decision_chain = components.render_decision_chain


# ═══════════════════════════════════════════════════════════════
# 视图一：对话——四路合一的那条路
# ═══════════════════════════════════════════════════════════════
def _init_agent_session() -> None:
    """这个浏览器会话还没会话号时，开一个。

    **刻意不做"从 URL 里恢复会话"**（`st.query_params["session"] = ...`）。
    会话号是凭证：拿到它就能读到这个会话的全部历史。塞进 URL 之后它会进
    浏览器历史、代理访问日志、跨站请求的 Referer 头——凭证最不该待的三个地方。

    代价是**新开标签页 = 新的一段对话**，上一次的接不上。要接上，会话号必须
    由调用方自己保管在安全的地方——这是 `api.py` 那种客户端能做、浏览器里
    没地方做的事。所以这个能力给 API 和 CLI，不给界面。

    屏幕上那份 `agent_turns`（带决策链）只属于当前这段浏览器会话，不落盘：
    决策链是"这一次请求怎么想的"，属于 trace，不属于对话内容。
    """
    if "agent_session_id" not in st.session_state:
        st.session_state["agent_session_id"] = session.ensure_session("")
    st.session_state.setdefault("agent_turns", [])


def _start_new_session() -> None:
    """换一个会话号重新开始。

    **不清空旧会话**——它还在盘上，由 `SESSION_TTL_DAYS` 那套过期和 CLI 的
    `delete` 管。在这里顺手删掉是错的：用户点的是"我想重新开始说"，不是
    "我想销毁记录"，两件事在界面上长得像，在语义上差很远。
    """
    st.session_state["agent_session_id"] = session.ensure_session("")
    st.session_state["agent_turns"] = []


# 空态的三张建议卡。**刻意各打一条路**：订单 / 知识库 / 数据分析各一张，
# 点一张就走通一个 worker——这本身就是这套多 Agent 的演示，而"随便问点什么"
# 的空输入不会演示出任何东西。
#
# 订单号用 52644 而不是另编一个：它是这个仓库到处在用的那个例子
# （api.py 的 examples、技术文档和项目说明里的 CLI 示例），而且它在种子数据里
# 有真实物流（目的地 San Salvador、「Late delivery」）。和文档对不上会让人
# 以为数据被换过了。
_SUGGESTIONS = {
    ":material/local_shipping: 订单 52644 到哪了": "订单 52644 到哪了",
    ":material/assignment_return: 退换货政策是什么": "退换货政策是什么",
    ":material/bar_chart: 哪个地区卖得最好": "哪个地区卖得最好",
}

# 演示模式下**不能出现"哪个地区卖得最好"那张卡**：数据分析被停用了，点了
# 只会得到知识库的"资料里没有"。换成一条知识库能真答上来的问题，既不让
# 演示第一步就落空，也不诱导别人去问一个我们知道办不到的事。
_DEMO_SUGGESTIONS = {
    ":material/local_shipping: 订单 52644 到哪了": "订单 52644 到哪了",
    ":material/assignment_return: 退换货政策是什么": "退换货政策是什么",
    ":material/schedule: 退款一般多久到账": "退款一般多久到账",
}


def render_agent_view() -> None:
    _init_agent_session()
    turns = st.session_state["agent_turns"]

    # 只有一条历史都没有时才画建议卡。这同时解决一个坑：pills 被点过之后会
    # 一直"选着"，而一旦有了一轮对话它就不再被渲染——Streamlit 会把不再渲染
    # 的 widget 状态回收掉。所以这里**不需要**手写 pop，也不会重复提交。
    picked = _render_empty_state() if not turns else None

    for turn in turns:
        _render_turn(turn)

    # 上传器放在输入框上方：先选图，再打字（或者只发图）。
    uploaded, send_image_only = _render_image_uploader()

    # chat_input 写在消息**之后**：Streamlit 自上而下渲染，只有这样它才会被
    # 钉在底部、上面的输出往上长。
    question = st.chat_input(
        "问点什么…（订单 / 售后政策 / 销售数据都能问）", key="chat_agent"
    ) or picked
    if not question and not (uploaded is not None and send_image_only):
        return

    display_question = question or "（图片）"
    if uploaded is not None:
        augmented = _prepare_image_question(uploaded, question or "")
        if augmented is None:
            return
        question = augmented
        # 换一个 key，让下一次 rerun 拿到一个空的 uploader。Streamlit 的
        # file_uploader 不能直接把 value 设成 None，只能换 key。
        st.session_state["image_upload_nonce"] = (
            st.session_state.get("image_upload_nonce", 0) + 1
        )

    if not question:
        return
    _run_turn(question, display_question=display_question)
    if uploaded is not None:
        st.rerun()


def _render_empty_state() -> str | None:
    """空态：一句引导 + 三张建议卡。**返回被点的那句问题**，不自己往下跑。

    **刻意不在这里 st.rerun()。** 那样会引出一个"待处理的问题"中间态，而中间态
    是下一轮 bug 的温床；更要紧的是先 append 再 rerun 会让一次点击**调两次模型**。
    直接把问题交回主流程往下走，一次点击就只有一次调用。
    """
    st.markdown("### :material/waving_hand: 想聊点什么？")
    if demo.enabled():
        st.caption("这是只读演示：订单进度、售后政策都能问——直接打字，或者点一张卡试试。")
    else:
        st.caption("订单进度、售后政策、销售数据都能问——直接打字，或者点一张卡试试。")
    suggestions = _DEMO_SUGGESTIONS if demo.enabled() else _SUGGESTIONS
    picked = st.pills(
        "试试这样问",
        list(suggestions),
        selection_mode="single",
        key="chat_suggestion",
        label_visibility="collapsed",
    )
    return suggestions.get(picked) if picked else None


def _render_image_uploader():
    """画上传器，返回 (上传的文件, 是否点了「只发送图片」)。

    key 里带一个 nonce：发完一张图后把它 +1，下一次 rerun 就是一个全新的
    uploader（`file_uploader` 不能直接把 value 设成 None，只能换 key）。
    """
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


def _prepare_image_question(uploaded, question: str) -> str | None:
    """识别上传的图片，拼成主链路能读的问题；失败时返回 None 或原始文字。

    三种降级：
      · 识别失败（没 key / 网络 / 非 JSON / 图太大）→ 提示手动输入；有文字就
        继续用文字，没文字就停在这里。
      · 识别成功但图里没有订单号 → 仍然把能认出的字段拼进去，同时提示用户
        查订单/退货要手动补订单号。
      · 识别成功但连诉求都没有、用户也没打字 → 提示再打一句话，不把空问题
        送进主链路。
    """
    try:
        data = uploaded.getvalue()
    except Exception as exc:  # noqa: BLE001 —— 读文件失败也不该让界面崩
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
        # 错误原文只进日志：视觉 API 的报错里可能带请求细节，不适合直接摆在
        # 公网页面上。用户需要知道的是"下一步做什么"。
        logger.warning("图片识别失败：%s", intent.error)
        st.warning("图片识别失败，请手动输入订单号或问题（详细原因已写入日志）。")
        return (question or "").strip() or None

    # 图片理解发生在主链路之前，它的 token 不进 supervisor 的账；演示模式的
    # 每日预算按"所有实际花掉的 token"算，所以这里单独记一笔。
    demo.add_tokens(intent.tokens)

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
    """把一轮问答铺出来。**历史回放和新问答共用这一段**。

    抽出来是因为它们的字段口径必须完全一致——两处显示同一件事时，措辞一致
    才能让人相信它们是同一套东西，而不是两套各自演化的视图。
    """
    with st.chat_message("user"):
        # `question` 可能是带图片块的"模型输入"，`display_question` 是用户
        # 真正打的那句话。没有后者时退回前者（旧会话/测试里的 turn 就是这种）。
        st.markdown(turn.get("display_question") or turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["final"].get("answer") or "（没有生成答复）")
        sources = turn["final"].get("sources") or []
        if sources:
            st.caption("来源：" + "、".join(str(s) for s in sources))
        _render_decision_chain(turn["final"], turn["elapsed_ms"])


def _run_turn(question: str, display_question: str | None = None) -> None:
    """跑一轮问答。

    `question` 是喂给模型的（可能带图片块），`display_question` 是画给用户看的。
    两者分开，是因为图片块对模型有用、对人只是噪音；而会话里存的必须是前者
    ——下一轮模型要靠它知道这张图说过什么。
    """
    sid = st.session_state["agent_session_id"]
    turns = st.session_state["agent_turns"]
    display_question = question if display_question is None else display_question

    # 两道入口闸放在**画出用户消息之前**：先显示问题再报"太频繁"，会让人
    # 以为这条问题已经被处理了。限流按会话号算——它挡不住换一个浏览器，
    # 真正兜住成本的是下面那个按天记账的 token 预算（见 ecom/demo.py）。
    if demo.enabled():
        if not demo.allow_request(f"session:{sid}"):
            st.warning("问得有点快，等一分钟再试。")
            return
        if demo.budget_exhausted():
            st.error("今天的演示额度已经用完，明天再来。")
            return

    with st.chat_message("user"):
        st.markdown(display_question)

    with st.chat_message("assistant"):
        started = datetime.now()
        # **每次现读，不缓存一份在 session_state 里。** 两个地方各存一份
        # 同一个历史，迟早会分叉（清空时忘了清另一份），而且会让
        # HISTORY_TURNS 这个上限在界面上失效。
        history = session.history(sid)
        # 记忆拼在真实历史前面，**每轮现拼**——不要写回 session_state，
        # 那份 history 是每轮现读的，写回去就跨轮重复累积了。
        augmented = memory.augmented_history(sid, question, history)

        # 状态条和正文是**两个容器**：状态条报进度，正文那块会被反复重画。
        # 正文不放进 status 里，是因为 status 收起后正文会跟着折进去——
        # 用户最后要读的是答复，不是那张进度条。
        status = st.status(":shimmer[正在路由与汇总…]", type="compact", state="running")
        answer_box = st.empty()
        pieces: list[str] = []
        final = None

        try:
            for event in answer_stream(question, augmented):
                kind = event.get("type")
                if kind == "delta":
                    # 只在**第一块**改一次状态文案：每块都 update 等于每块都
                    # 往浏览器发一次组件更新，token 越多白白越慢。
                    if not pieces:
                        status.update(label=":shimmer[正在生成答复…]", state="running")
                    pieces.append(event.get("text") or "")
                    # 末尾的光标只是提醒"还没完"；收到 final 就会被完整答复覆盖。
                    answer_box.markdown("".join(pieces) + "▌")
                elif kind == "final":
                    final = event.get("state") or {}
        except Exception:
            # **异常原文只进日志。** 模型/数据库的报错里可能带内部路径，而用户
            # 需要知道的是"这次没成，接下来怎么办"。
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

        # 记账放在"确认真的答完"之后：中途失败的请求没拿到完整用量，
        # 漏记的那部分由预算余量吸收；宁可少记，也不要在失败收尾时再抛一次。
        demo.add_tokens(final.get("tokens") or {})

        # **以 final 为准**，不用拼出来的 pieces：汇总中途失败时服务端会退回
        # 拼接，两者可能不一致（见模块开头）。
        answer_box.markdown(final.get("answer") or "（没有生成答复）")
        sources = final.get("sources") or []
        if sources:
            st.caption("来源：" + "、".join(str(s) for s in sources))
        _render_decision_chain(final, elapsed_ms)

    # 先答完再落盘：模型调用失败时不会留下一条"问了没答"的记录。
    session.append_turn(sid, question, final.get("answer") or "")
    turns.append(
        {
            "question": question,
            "display_question": display_question,
            "final": final,
            "elapsed_ms": elapsed_ms,
        }
    )


# 三个直连视图：实现在 ecom/ui/views.py。这里只做装配，所以测试继续
# monkeypatch `app._db` / `app._store` 就能换掉依赖——视图拿到的是函数，
# 到渲染那一刻才调用。
def _view_context() -> ViewContext:
    return ViewContext(
        db=_db, store=_store, demo=demo, worker_timeout=WORKER_TIMEOUT
    )


def render_order_view() -> None:
    views.render_order_view(_view_context())


def render_analytics_view() -> None:
    views.render_analytics_view(_view_context())


def render_knowledge_view() -> None:
    views.render_knowledge_view(_view_context())


# ═══════════════════════════════════════════════════════════════
# 导航
# ═══════════════════════════════════════════════════════════════
class View(NamedTuple):
    """一个视图的显示名、图标和渲染函数。"""

    name: str
    icon: str
    render: Callable[[], None]


# 键 → 视图。**顺序就是侧边栏里的顺序**，对话排第一：它是最常用的那条路。
VIEWS: dict[str, View] = {
    "chat": View("对话", ":material/forum:", render_agent_view),
    "order": View("订单", ":material/inventory_2:", render_order_view),
    "analytics": View("数据看板", ":material/monitoring:", render_analytics_view),
    "knowledge": View("知识库", ":material/menu_book:", render_knowledge_view),
}
DEFAULT_VIEW = "chat"


def render_sidebar() -> str:
    """画侧边栏，返回**当前该显示哪个视图**。

    几个必须守住的点，每一个背后都有测试盯着：

    · 环境那几行只能用 `st.markdown`，不能降级成 `st.caption`——
      `test_app_real_streamlit.py` 读的是侧边栏的 markdown。钻进 expander 是可以的
      （AppTest 会把内层一起收）。
    · 全侧边栏**只能有一个 `st.code`**（侧边栏最容易被截图发出去的地方，而那段
      是凭证），所以 main.py 那句提示只能用带反引号的 caption。
    · 导航的值走**函数返回值**，不是读 `session_state["nav_view"]`——两套测试环境
      （真的 Streamlit 和那份手写的假模块）只有这条路是共通的。
    """
    with st.sidebar:
        st.markdown("#### :material/smart_toy: 电商智能客服")
        if demo.enabled():
            st.caption("只读演示：一次提问由 Supervisor 交给知识库或订单 worker。")
        else:
            st.caption("一次提问由 Supervisor 决定交给谁：知识库 / 订单 / 数据分析。")

        if st.button("新建会话", key="sidebar_new_session", icon=":material/add:"):
            _start_new_session()
            st.rerun()

        if demo.enabled():
            # 演示模式只开放对话。不画 pills 而不是画一个"点不动的"——后者
            # 会让人以为界面坏了；这里直接说明只剩哪一条路。
            view = DEFAULT_VIEW
            with st.container(border=True):
                st.caption("直连视图")
                st.markdown("演示模式只开放「对话」；订单和知识库仍可通过提问触达。")
        else:
            with st.container(border=True):
                st.caption("直连视图")
                view = (
                    st.pills(
                        "视图",
                        list(VIEWS),
                        default=DEFAULT_VIEW,
                        format_func=lambda key: f"{VIEWS[key].icon} {VIEWS[key].name}",
                        selection_mode="single",
                        key="nav_view",
                        label_visibility="collapsed",
                    )
                    or DEFAULT_VIEW
                )

        _init_agent_session()
        if demo.enabled():
            # **演示模式下不显示会话号。** 它是凭证，而公网页面上多一个
            # 可复制的东西并不会让演示更好看，只会多一个能读到会话历史的入口。
            st.caption("演示会话不显示会话号；关掉页面就是一段新对话。")
        else:
            # 只显示前 12 位。它是**凭证**（拿到它就能读到这个会话的全部历史），
            # 侧边栏会被人截图发出去，所以不整条摆在这里——要看全的用
            # `python -m ecom.orchestrator.cli.session list`。
            st.code(st.session_state["agent_session_id"][:12] + "…", language=None)
            st.caption("会话号只露前 12 位：它是凭证，拿到就能读到这段对话的全部历史。")

        with st.expander("环境", icon=":material/monitor_heart:", expanded=False):
            if demo.enabled():
                st.markdown("- 模式：**只读演示**（退货申请、数据分析、后台均未开放）")
            st.markdown(f"- 向量库后端：`{_backend_label()}`")
            try:
                st.markdown(f"- 订单库：`{_db().backend}`")
            except Exception as exc:  # noqa: BLE001
                st.markdown(f"- 订单库：打不开（{_clip(exc, 60)}）")
            st.markdown(f"- 路由上限：{MAX_ROUNDS} 轮")
            st.markdown(f"- worker 超时：{WORKER_TIMEOUT}")
            st.markdown(f"- 每轮回读历史：{HISTORY_TURNS} 轮")

        st.caption(
            "界面会把「哪一层慢 / 哪一层挂了」藏起来——超时变成转圈，异常变成红框。"
            "排查时用 `uv run python main.py -v`，那里把每次请求的决策链摊开。"
        )
        st.caption(f"启动于 {datetime.now():%Y-%m-%d %H:%M}")

    return view


def _demo_gate() -> bool:
    """
    公网演示的邀请码门。返回 True = 可以继续；False = 已经画完门禁页。

    **演示模式没开时直接放行**，所以本地/CI 不受影响。开了之后：
      · 没配邀请码 → 拒绝所有人（fail closed），并把原因说出来；
      · 配了 → 输对一次就记进这个浏览器会话，之后不再问。
    邀请码不是账号：没有用户、没有审计、也不能单独吊销某一个人。
    """
    if not demo.enabled() or st.session_state.get("demo_ok"):
        return True

    st.title("多 Agent 电商智能客服", icon=":material/shopping_cart:")
    st.caption("这是一个只读演示环境：可以查订单、问售后政策，不能提交退货申请。")

    if not demo.invite_configured():
        st.error("演示环境还没有配置邀请码（DEMO_INVITE_CODE），暂时无法进入。")
        return False

    code = st.text_input("邀请码", type="password", key="demo_invite")
    if st.button("进入", key="demo_enter"):
        if demo.check_invite(code):
            st.session_state["demo_ok"] = True
            st.rerun()
        else:
            st.error("邀请码不正确。")
    return False


def main() -> None:
    if not _demo_gate():
        return
    view = render_sidebar()
    current = VIEWS[view]
    st.title(current.name, icon=current.icon)
    current.render()


if __name__ == "__main__":
    main()
