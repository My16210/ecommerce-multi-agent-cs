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

【实时 / 回放：这条边界必须写在看得见的地方】
对话视图里**没有 token 可以流**——`answer()` 是一次性阻塞返回 dict，没有回调。
所以我们不让它假装在流：转的是 `st.status(type="compact")` 那个圈（那是真的，
它盖住了 `answer()` 阻塞的整个时长），步骤是**事后回放**的。
**知识库视图是另一回事**：`generate_answer()` 有 `streaming.sink` 那个汇，
所以那里是**真的逐 token 流出来的**。两条路的差异本身就是证据——不是不会做，
是那条路没有接口。

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

【不做的事，写清楚比留白好】
  · 不做登录 / 权限：这是离线演示，加一套假登录只会让演示多几步点击。
    （**后台管理端不在这份文件里**——它要能批准退货，所以有密码门禁，
      见仓库根的 `admin.py`。两个入口分开，正是为了让这条"不做"继续成立。）
  · 不做多副本：会话落在本地 sqlite，**多副本各写各的文件，上下文会岔**。
    和 api.py 那边是同一个已声明的边界（要真解决得换 Redis 之类的共享存储）。
  · 不在这个文件里写 CSS。整个仓库唯一一处"样式"是 `.streamlit/config.toml`，
    理由是原生主题令牌跟着 Streamlit 版本走，而 CSS 选择器指向内部类名。
"""
from collections.abc import Callable
from datetime import datetime
from typing import NamedTuple

import streamlit as st

from ecom.knowledge.generator.answer import generate_answer
from ecom.knowledge.retriever.reranker import rerank
from ecom.knowledge.retriever.retriever import retrieve
from ecom.knowledge.utils import streaming
from ecom.knowledge.vector_store import get_vector_store
from ecom.orchestrator import memory, session
from ecom.orchestrator.core.aggregate import WORKER_LABELS
from ecom.orchestrator.core.supervisor import answer
from ecom.orchestrator.db.engine import get_db
from ecom.orchestrator.settings import HISTORY_TURNS, MAX_ROUNDS, WORKER_TIMEOUT
from ecom.orchestrator.tools.order_tools import (
    OrderNotFound,
    describe_status,
    get_returns,
    query_order,
)
from ecom.orchestrator.workers.analytics import run_analytics
from ecom.orchestrator.workers.order import describe_route

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


def _backend_label() -> str:
    from ecom.knowledge.settings import VECTOR_BACKEND

    return VECTOR_BACKEND


def _clip(text, limit: int = 160) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


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


def render_agent_view() -> None:
    _init_agent_session()
    turns = st.session_state["agent_turns"]

    # 只有一条历史都没有时才画建议卡。这同时解决一个坑：pills 被点过之后会
    # 一直"选着"，而一旦有了一轮对话它就不再被渲染——Streamlit 会把不再渲染
    # 的 widget 状态回收掉。所以这里**不需要**手写 pop，也不会重复提交。
    picked = _render_empty_state() if not turns else None

    for turn in turns:
        _render_turn(turn)

    # chat_input 写在消息**之后**：Streamlit 自上而下渲染，只有这样它才会被
    # 钉在底部、上面的输出往上长。
    question = st.chat_input(
        "问点什么…（订单 / 售后政策 / 销售数据都能问）", key="chat_agent"
    ) or picked
    if not question:
        return

    _run_turn(question)


def _render_empty_state() -> str | None:
    """空态：一句引导 + 三张建议卡。**返回被点的那句问题**，不自己往下跑。

    **刻意不在这里 st.rerun()。** 那样会引出一个"待处理的问题"中间态，而中间态
    是下一轮 bug 的温床；更要紧的是先 append 再 rerun 会让一次点击**调两次模型**。
    直接把问题交回主流程往下走，一次点击就只有一次调用。
    """
    st.markdown("### :material/waving_hand: 想聊点什么？")
    st.caption("订单进度、售后政策、销售数据都能问——直接打字，或者点一张卡试试。")
    picked = st.pills(
        "试试这样问",
        list(_SUGGESTIONS),
        selection_mode="single",
        key="chat_suggestion",
        label_visibility="collapsed",
    )
    return _SUGGESTIONS.get(picked) if picked else None


def _render_turn(turn: dict) -> None:
    """把一轮问答铺出来。**历史回放和新问答共用这一段**。

    抽出来是因为它们的字段口径必须完全一致——两处显示同一件事时，措辞一致
    才能让人相信它们是同一套东西，而不是两套各自演化的视图。
    """
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["final"].get("answer") or "（没有生成答复）")
        sources = turn["final"].get("sources") or []
        if sources:
            st.caption("来源：" + "、".join(str(s) for s in sources))
        _render_decision_chain(turn["final"], turn["elapsed_ms"])


def _run_turn(question: str) -> None:
    sid = st.session_state["agent_session_id"]
    turns = st.session_state["agent_turns"]

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # 【这里实时的只有这个圈，这是刻意的】
        # `answer()` 是一次性阻塞返回 dict，没有任何 token 回调，所以对话这条路
        # **没有 token 可以流**。我们没有让它假装在流——转的是这个 compact 状态
        # 本身（它盖住 `answer()` 阻塞的整段时长，这部分是真的），返回之后再
        # 一次性填内容。想让它真流，得给 supervisor 开第二条回答路径，而记账
        # （trace.record）和 token 汇总都在 invoke() 之后，两条路各写一份必然分叉
        # ——这个仓库的规矩是编排层的回答路径只有一条。
        with st.status(
            ":shimmer[正在路由与汇总…]", type="compact", state="running"
        ) as status:
            started = datetime.now()
            # **每次现读，不缓存一份在 session_state 里。** 两个地方各存一份
            # 同一个历史，迟早会分叉（清空时忘了清另一份），而且会让
            # HISTORY_TURNS 这个上限在界面上失效。
            history = session.history(sid)
            # 记忆拼在真实历史前面，**每轮现拼**——不要写回 session_state，
            # 那份 history 是每轮现读的，写回去就跨轮重复累积了。
            final = answer(
                question, memory.augmented_history(sid, question, history)
            )
            elapsed_ms = int((datetime.now() - started).total_seconds() * 1000)
            status.update(label=f"已完成（{elapsed_ms} ms）", state="complete")

        st.markdown(final.get("answer") or "（没有生成答复）")
        sources = final.get("sources") or []
        if sources:
            st.caption("来源：" + "、".join(str(s) for s in sources))
        _render_decision_chain(final, elapsed_ms)

    # 先答完再落盘：模型调用失败时不会留下一条"问了没答"的记录。
    session.append_turn(sid, question, final.get("answer") or "")
    turns.append({"question": question, "final": final, "elapsed_ms": elapsed_ms})


def _render_decision_chain(final: dict, elapsed_ms: int) -> None:
    """`main.py -v` 那份内容的可视化版。

    刻意保留同样的字段、同样的措辞——两处显示同一件事时，措辞一致才能让人
    相信它们是同一个系统的两个视图，而不是两套各自演化的东西。

    【为什么外面是 status、里面是 expander，而不是两层 status】
    `st.status` 的 docstring 里明写着 "don't nest status containers"。而
    `st.expander` 有**同一个 `type` 参数**（"compact" / "step" 都认），
    渲染出来的步骤时间线是一样的——所以内层用 expander，观感不变而不破规矩。

    【折叠着也能看】标题上直接写了路由和决策，不展开就知道这一次是怎么走的。
    这比原来那个「显示决策链」开关好：开关要先被发现，而标题是白送的。
    """
    plan = "、".join(final.get("plan") or []) or "（空）"
    with st.status(
        f"决策链　路由 {plan}　决策 {final.get('decision')}",
        type="compact",
        state="complete",
        expanded=False,
    ):
        st.markdown(
            f"**路由**：{plan}　来源 `{final.get('router_source')}`　"
            f"置信度 {final.get('plan_confidence', 0):.2f}"
        )
        if final.get("plan_reason"):
            st.caption(final["plan_reason"])

        for r in final.get("results") or []:
            label = WORKER_LABELS.get(r.get("worker"), r.get("worker"))
            state = "成功" if r.get("ok") else "失败"
            mark = "（未答上）" if r.get("insufficient") else ""
            with st.expander(
                f"第 {r.get('round')} 轮 {label}：{state}{mark}　{r.get('elapsed_ms')}ms",
                type="step",
                expanded=False,
            ):
                if not r.get("ok"):
                    # 错误原文在 CLI 里给的是运维看的；这里同理——用户看到的是上面
                    # 那句"失败"，展开的人是在排查。
                    st.caption(f"错误：{r.get('error')}")
                elif r.get("answer"):
                    st.caption(f"结论：{_clip(r['answer'], 120)}")

        st.caption(
            f"决策：{final.get('decision')}（{final.get('decision_reason')}）　"
            f"耗时 {elapsed_ms}ms　token {final.get('tokens')}"
        )


# ═══════════════════════════════════════════════════════════════
# 视图二：订单——`query_order` 那个工具本身
# ═══════════════════════════════════════════════════════════════
def render_order_view() -> None:
    db = _db()

    # 这里**直接查了库**，没走工具层，理由写明白：
    # order_tools 里没有"列出全部订单"——Agent 永远是从用户话里拿到订单号或
    # 客户名，不需要这个能力。界面需要它，只是为了让人能点着玩、不必先背一个
    # 订单号。所以它属于界面，不属于工具层：给 Agent 加一个它用不到的宽查询口，
    # 只是多一个能被误用的入口。
    rows = db.query(
        "SELECT order_id, customer_name, status, amount, region, created_at "
        "FROM orders ORDER BY created_at DESC LIMIT 200"
    )
    if not rows:
        st.warning("订单库是空的。先在终端跑一次：\n\n```\nuv run python main.py --init-db\n```")
        return

    labels = {
        f"{r['order_id']}　{r.get('customer_name') or '-'}　{r['status']}　￥{r['amount']}"
        + f"　{r.get('region') or '-'}": r["order_id"]
        for r in rows
    }
    picked = st.selectbox("选一笔订单", list(labels))

    try:
        detail = query_order(labels[picked], db)
    except OrderNotFound as exc:
        st.error(str(exc))
        return

    order, logistics = detail["order"], detail["logistics"]

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        # 状态用 describe_status 带上中文注释：库里存的是 COMPLETE / PENDING_PAYMENT
        # 这种英文原值（见 order_tools.STATUS_GLOSS），给演示的人看纯英文会以为没翻译完。
        c1.metric("状态", describe_status(order["status"]))
        # 金额一律 .2f：数据库里是 REAL，直接 f-string 会打出 629.9200000000001 这种
        # 东西。worker 那边的回答已经格式化过了，界面不跟着做就会两边不一致。
        c2.metric("金额", f"￥{order['amount']:.2f}")
        c3.metric("客户", order.get("customer_name") or "-")
        c4.metric("地区", order.get("region") or "-")

    # [这里原来写的是 `if paid_at else '（未支付）'`] 换成真实数据后这是**假话**：
    # DataCo 里根本没有支付时间这一列，`paid_at` 为空表示"数据集没有这个字段"，
    # 不是"这笔订单没付钱"。同一句话里的空值有两种含义就不能用同一个说法兜——
    # 空着不说，比说错强。（worker 那边的 `_fmt_status` 是同一个道理，两处一致。）
    st.caption(
        f"下单 {order.get('created_at')}"
        + (f"　支付 {order.get('paid_at')}" if order.get("paid_at") else "")
    )

    st.space("medium")
    st.subheader("物流", icon=":material/local_shipping:")
    if not logistics:
        st.info("这单没有物流记录")
        st.caption(
            "订单状态为 CANCELED / SUSPECTED_FRAUD 的订单本来就没有发货，"
            "不是数据缺失。（公开数据集里没有运单表，只有配送状态。）"
        )
    else:
        # 库里只有一行当前状态（没有轨迹表），所以这里**不画进度条**：画一个凭
        # status 推出来的四段进度，看起来更漂亮，但那是在数据没有的粒度上伪造信息。
        #
        # 承运商 / 运单号 / 发货地在公开数据里**根本不存在**，所以"没有"是常态。
        # 原来这里写的是「（承运商未知）　运单号 -」，对每一笔订单都成立——
        # 满屏的"未知"读起来像系统坏了，而不是"我们本来就没有这个信息"。有的才显示。
        if logistics.get("carrier") or logistics.get("tracking_no"):
            st.markdown(
                f"**{logistics.get('carrier') or '承运商未记录'}**　"
                f"运单号 `{logistics.get('tracking_no') or '-'}`"
            )
        st.markdown(
            f"{describe_route(logistics)}　当前：**{logistics.get('status') or '-'}**"
        )
        st.caption(f"最后更新 {logistics.get('updated_at') or '-'}")

    returns = get_returns(order["order_id"], db)
    if returns:
        st.space("medium")
        st.subheader("退货申请", icon=":material/assignment_return:")
        # 不传 use_container_width：那是旧写法，新版 Streamlit 已改用 width=，
        # 而两边的名字不通用（装错版本会 TypeError）。`width` 不传就是撑满。
        st.dataframe(returns, width="stretch")
    else:
        st.caption("没有退货申请")


# ═══════════════════════════════════════════════════════════════
# 视图三：数据看板——走子进程的那条路
# ═══════════════════════════════════════════════════════════════
def render_analytics_view() -> None:
    st.caption(
        "问一句关于销售数据的话，后台会**让模型写一段 pandas 代码、在你机器上跑**，"
        f"再把人话结论写回来。所以这一步真的会跑代码——超时 {WORKER_TIMEOUT.get('analytics')}s，"
        "到点子进程会被杀掉。"
    )

    question = st.text_input(
        "要分析什么？",
        value="",
        placeholder="例如：哪个地区卖得最好 / 各商品销量排名 / 月度销售趋势",
        key="analytics_question",
    )
    if not st.button("开始分析", key="analytics_run", icon=":material/play_arrow:") or not question.strip():
        return

    # 和对话视图同一条规矩：**实时的只有这个圈，步骤是事后回放的**。
    # `run_analytics()` 起的子进程里做完整条链（生成代码 → 执行 → 自查 → 写结论），
    # 中间没有回调可以挂。不假装在流。
    with st.status(
        ":shimmer[生成代码 → 执行 → 自查 → 写结论…]", type="compact", state="running"
    ) as status:
        try:
            result = run_analytics(question.strip())
        except Exception as exc:  # noqa: BLE001 —— 子进程边界已经在里面收过一遍了
            status.update(label="没跑成", state="error")
            st.error(f"数据分析没跑成：{exc}")
            return
        status.update(
            label=f"已完成（{result.get('elapsed_ms')} ms）", state="complete"
        )

    data = result.get("data") or {}
    if data.get("review_passed") is False:
        # 自查没过不等于结果一定错，但必须让人看得见——否则一份"没通过审查"
        # 的报告会被当成和别的报告一样可信。
        st.warning("这份报告的代码自查没有通过，结论可能不可靠（报告里也标了）。")

    st.markdown(result.get("answer") or "（没有生成报告）")

    chart_path = data.get("chart_path")
    if chart_path and data.get("chart_generated"):
        try:
            # 图按原始尺寸显示：数据分析那边生成的图本来就有合理的 figsize。
            st.image(chart_path)
        except Exception as exc:  # noqa: BLE001
            st.caption(f"图生成成功但渲染失败（{exc}），文件在 {chart_path}")
    else:
        st.caption("这一问没有出图（不是每个问题都需要图）。")

    st.caption(f"耗时 {result.get('elapsed_ms')}ms　token {result.get('tokens')}")


# ═══════════════════════════════════════════════════════════════
# 视图四：知识库——检索 + 引用，**逐 token 真的流出来**
# ═══════════════════════════════════════════════════════════════
def render_knowledge_view() -> None:
    store = _store()
    try:
        count = store.count()
    except Exception as exc:  # noqa: BLE001
        st.error(f"向量库打不开：{exc}")
        return

    if not count:
        st.warning("向量库是空的。先在终端跑一次：\n\n```\nuv run python -m ecom.knowledge.cli.ingest\n```")
        return

    st.caption(f"{_backend_label()} 后端，已索引 {count} 个 chunk。答复下方是它引用的段落。")

    # **必须是 setdefault，不能写成 `.get("kb_history", [])`。** 下面写历史用的是
    # `history += [...]`，它是**原地改**这个 list；`.get` 在键不存在时返回的是一张
    # 没人持有的新 list，改完就丢——多轮上下文会静默失效，而画面上看不出来。
    history = st.session_state.setdefault("kb_history", [])
    turns = st.session_state.setdefault("kb_turns", [])

    if st.button("清空对话", key="kb_clear", icon=":material/delete_sweep:"):
        st.session_state["kb_history"] = []
        st.session_state["kb_turns"] = []
        st.rerun()

    for turn in turns:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            _render_citations(turn["chunks"])

    question = st.chat_input("问知识库…（退换货政策、会员积分、促销规则…）", key="chat_kb")
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.status(":shimmer[检索中…]", type="compact", state="running") as status:
            try:
                hits = retrieve(question, store)
                chunks = rerank(question, hits)
            except Exception as exc:  # noqa: BLE001
                # 检索这一路失败**不降级**（只有 rerank 失败才降级，那是它自己的事）。
                # 向量库连不上时报"未命中"会让人以为知识库里没这段内容——那是
                # 完全不同的结论，见 retriever.retrieve 的降级语义。
                status.update(label="检索失败", state="error")
                st.error(f"检索失败：{exc}")
                return
            status.update(label=f"检索到 {len(chunks)} 段", state="complete")

        # 【这里是真流式——和对话视图刻意不同】
        # 对话那条路不能流，是因为 `answer()` 没有回调；这一条能流，是因为
        # `generate_answer()` 读 `streaming.current()`（见 streaming.py 的 docstring：
        # 那个汇的调用方只有它一个）。所以这里挂的是真汇，不是模拟。
        #
        # 挂汇这件事只改"什么时候看到"，不改"答案是什么"——generate_answer
        # 里两条路拼出来的文本有测试钉着一致。
        buffer: list = []
        placeholder = st.empty()

        def _on_token(piece: str) -> None:
            buffer.append(piece)
            placeholder.markdown("".join(buffer) + "▌")

        try:
            with streaming.sink(_on_token):
                text = generate_answer(question, chunks, [], history)
        except Exception as exc:  # noqa: BLE001
            placeholder.empty()
            st.error(f"生成失败：{exc}")
            return

        placeholder.markdown(text)
        _render_citations(chunks)

    history += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": text},
    ]
    turns.append({"question": question, "answer": text, "chunks": chunks})


def _render_citations(chunks: list) -> None:
    if not chunks:
        st.caption("这一问没有召回到任何段落")
        return
    with st.expander(
        f"引用到的段落（{len(chunks)} 条）", icon=":material/attachment:", expanded=False
    ):
        for chunk in chunks:
            if not isinstance(chunk, dict):
                st.markdown(f"- {chunk}")
                continue
            score = chunk.get("rerank_score", chunk.get("score"))
            head = f"**{chunk.get('source') or '未知来源'}**"
            if chunk.get("chunk_id") is not None:
                head += f" #{chunk.get('chunk_id')}"
            if score is not None:
                head += f"　相关性 {float(score):.3f}"
            st.markdown(head)
            st.caption(_clip(chunk.get("text"), 200))
            st.space("small")


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
        st.caption("一次提问由 Supervisor 决定交给谁：知识库 / 订单 / 数据分析。")

        if st.button("新建会话", key="sidebar_new_session", icon=":material/add:"):
            _start_new_session()
            st.rerun()

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
        # 只显示前 12 位。它是**凭证**（拿到它就能读到这个会话的全部历史），
        # 侧边栏会被人截图发出去，所以不整条摆在这里——要看全的用
        # `python -m ecom.orchestrator.cli.session list`。
        st.code(st.session_state["agent_session_id"][:12] + "…", language=None)
        st.caption("会话号只露前 12 位：它是凭证，拿到就能读到这段对话的全部历史。")

        with st.expander("环境", icon=":material/monitor_heart:", expanded=False):
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


def main() -> None:
    view = render_sidebar()
    current = VIEWS[view]
    st.title(current.name, icon=current.icon)
    current.render()


if __name__ == "__main__":
    main()
