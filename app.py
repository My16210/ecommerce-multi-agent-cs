"""
统一界面（Streamlit）。四个 Tab = 这个项目四个能做事的入口，共用一份进程。

    uv sync --extra ui
    uv run streamlit run app.py

【为什么是一份 app.py，而不是三份并排放】
合并前每个仓库各有一个 app.py：rag 那个问知识库，sales 那个做数据分析，
而编排层干脆没有界面（只有 main.py / api.py）。三个界面并排放在同一个 README
里，读到的人得自己拼出"它们其实是同一个系统"——而这件事本该由**结构**说出来。

所以这里的四个 Tab 不是"三个 App 加一个新 Tab"，是**一件事的四个面**：
智能客服是总的入口，另外三个是它下面那三条路各自的可视化。点开订单 Tab
看到的是提示词里那个 `query_order` 工具，不是另一个项目。

【三个 Tab 各自的调用方式，刻意不一致，理由是一致的】
    智能客服 → import `answer()`        进程内
    知识库   → import `retrieve/rerank/generate_answer`   进程内
    数据看板 → import `run_analytics()`  **子进程**
    订单     → import `query_order()`   进程内（本地 SQLite）

前三行不一致的地方在后两行。数据分析走子进程，和 CLI / FastAPI 走的是同一个
函数、同一条边界——因为那个边界守的是"它要执行模型生成的代码，需要一个能被
真杀死的进程"（见 ecom/orchestrator/workers/analytics.py）。**界面不该是那条
边界的例外**：在界面里为了"少起一个进程"改成进程内调用，等于换了一个没有
超时、崩了会带走整个页面的执行环境，而收益只是省掉一次 Python 启动。所以
这里宁可绕一圈。

【界面会藏住什么——这条必须写在使用者看得见的地方】
超时变成一个转圈动画，异常变成一个红框，"哪一层慢、哪一层挂了"全看不见。
排查时该用的仍然是 `uv run python main.py -v`。界面是演示用的，CLI 是查问题
用的，这个顺序不能反。

【会话持久化——以前这里写的是"不做"】
这段以前写的是：「多轮历史放 `st.session_state`，关掉页面就没了。真要持久化，
正确的落点是 ecom/orchestrator 那层的会话存储，不是在界面里塞一个 sqlite。」

后半句就是现在做的事——落点在 `ecom/orchestrator/session.py`，界面只是它的
一个使用者（和 `api.py`、`python -m ecom.orchestrator.cli.session` 平级）。
这条没有变：**界面里没有自己的 sqlite**。所以下面这段代码里一个 `sqlite3`
都找不到，只有 `session.xxx()` 的调用。

顺带修掉一个真问题：以前那份 `agent_history` 是**只增不减**的——界面上聊到
第 50 轮，就会把 50 轮全塞给模型。现在每次提问都从存储里现读
`session.history()`，截断由 `HISTORY_TURNS` 统一管。**喂给模型多少**和
**留在盘上多久**是两个参数，这条边界以前在界面上是失控的。

**边界要说清，别把它说大**：界面这边落盘买到的是"上下文有上限、且这段对话
在 CLI 里看得见也删得掉"，**不是"换个标签页还能接着聊"**。会话号只活在
`st.session_state` 里，开一个新标签页就是一段新对话。想跨标签页续上，得把
会话号放进 URL——**而那是错的**：它是凭证，进了 URL 就会被浏览器历史、代理
日志、Referer 头带走（所以侧边栏里也只显示前 12 位）。这个代价值得付，
因为它换来的是"凭证不出现在它不该出现的地方"。

【不做的事，写清楚比留白好】
  · 不做登录 / 权限：这是离线演示，加一套假登录只会让演示多几步点击。
    （**后台管理端不在这份文件里**——它要能批准退货，所以有密码门禁，
      见仓库根的 `admin.py`。两个入口分开，正是为了让这条"不做"继续成立。）
  · 不做多副本：会话落在本地 sqlite，**多副本各写各的文件，上下文会岔**。
    和 api.py 那边是同一个已声明的边界（要真解决得换 Redis 之类的共享存储）。
"""
from datetime import datetime

import streamlit as st

from ecom.knowledge.generator.answer import generate_answer
from ecom.knowledge.retriever.reranker import rerank
from ecom.knowledge.retriever.retriever import retrieve
from ecom.knowledge.utils import streaming
from ecom.knowledge.vector_store import get_vector_store
from ecom.orchestrator import session
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

st.set_page_config(page_title="多 Agent 电商智能客服", page_icon="🛒", layout="wide")


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
# Tab 1：智能客服——四路合一的那条路
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


def render_agent_tab() -> None:
    st.caption(
        f"一次提问由 Supervisor 决定交给谁：知识库 / 订单 / 数据分析，"
        f"最多回环 {MAX_ROUNDS} 轮。"
    )
    _init_agent_session()
    turns = st.session_state["agent_turns"]

    col_a, col_b, _ = st.columns([1, 1, 4])
    with col_a:
        if st.button("新建会话", key="agent_clear"):
            _start_new_session()
            st.rerun()
    with col_b:
        verbose = st.toggle("显示决策链", value=True, key="agent_verbose")

    for turn in turns:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["final"].get("answer") or "（没有生成答复）")
            sources = turn["final"].get("sources") or []
            if sources:
                st.caption("来源：" + "、".join(str(s) for s in sources))
            if verbose:
                _render_decision_chain(turn["final"], turn["elapsed_ms"])

    question = st.chat_input("问点什么…（订单 / 售后政策 / 销售数据都能问）")
    if not question:
        return

    sid = st.session_state["agent_session_id"]

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("路由中…"):
            started = datetime.now()
            # **每次现读，不缓存一份在 session_state 里。** 两个地方各存一份
            # 同一个历史，迟早会分叉（清空时忘了清另一份），而且会让
            # HISTORY_TURNS 这个上限在界面上失效。
            history = session.history(sid)
            final = answer(question, history)
            elapsed_ms = int((datetime.now() - started).total_seconds() * 1000)
        st.markdown(final.get("answer") or "（没有生成答复）")
        sources = final.get("sources") or []
        if sources:
            st.caption("来源：" + "、".join(str(s) for s in sources))
        if verbose:
            _render_decision_chain(final, elapsed_ms)

    # 先答完再落盘：模型调用失败时不会留下一条"问了没答"的记录。
    session.append_turn(sid, question, final.get("answer") or "")
    turns.append({"question": question, "final": final, "elapsed_ms": elapsed_ms})


def _render_decision_chain(final: dict, elapsed_ms: int) -> None:
    """`main.py -v` 那份内容的可视化版。

    刻意保留同样的字段、同样的措辞——两处显示同一件事时，措辞一致才能让人
    相信它们是同一个系统的两个视图，而不是两套各自演化的东西。
    """
    plan = "、".join(final.get("plan") or []) or "（空）"
    with st.expander(f"决策链　路由 {plan}　决策 {final.get('decision')}", expanded=False):
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
            st.markdown(
                f"· 第 {r.get('round')} 轮 **{label}**：{state}{mark}　"
                f"{r.get('elapsed_ms')}ms　token {r.get('tokens')}"
            )
            if not r.get("ok"):
                # 错误原文在 CLI 里给的是运维看的；这里同理——用户看到的是上面那句
                # "失败"，展开的人是在排查。
                st.caption(f"　错误：{r.get('error')}")
            elif r.get("answer"):
                st.caption(f"　结论：{_clip(r['answer'], 120)}")

        st.caption(
            f"决策：{final.get('decision')}（{final.get('decision_reason')}）　"
            f"耗时 {elapsed_ms}ms　token {final.get('tokens')}"
        )


# ═══════════════════════════════════════════════════════════════
# Tab 2：订单——`query_order` 那个工具本身
# ═══════════════════════════════════════════════════════════════
def render_order_tab() -> None:
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
    order_id = labels[picked]

    try:
        detail = query_order(order_id, db)
    except OrderNotFound as exc:
        st.error(str(exc))
        return

    order, logistics = detail["order"], detail["logistics"]

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

    st.divider()
    st.subheader("物流")
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

    returns = get_returns(order_id, db)
    if returns:
        st.divider()
        st.subheader("退货申请")
        # 不给宽度参数。`use_container_width` 是旧写法，新版 Streamlit 已标记
        # 弃用并改用 `width`——而两边的名字不通用。这里刻意用默认值：一个演示
        # 界面里"表格窄一点"远不值得换成一个装错版本就报 TypeError 的调用。
        st.dataframe(returns)
    else:
        st.caption("没有退货申请")


# ═══════════════════════════════════════════════════════════════
# Tab 3：数据看板——走子进程的那条路
# ═══════════════════════════════════════════════════════════════
def render_analytics_tab() -> None:
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
    if not st.button("开始分析", key="analytics_run") or not question.strip():
        return

    with st.spinner("生成代码 → 执行 → 自查 → 写结论，可能要几十秒…"):
        try:
            result = run_analytics(question.strip())
        except Exception as exc:  # noqa: BLE001 —— 子进程边界已经在里面收过一遍了
            st.error(f"数据分析没跑成：{exc}")
            return

    data = result.get("data") or {}
    if data.get("review_passed") is False:
        # 自查没过不等于结果一定错，但必须让人看得见——否则一份"没通过审查"
        # 的报告会被当成和别的报告一样可信。
        st.warning("这份报告的代码自查没有通过，结论可能不可靠（报告里也标了）。")

    st.markdown(result.get("answer") or "（没有生成报告）")

    chart_path = data.get("chart_path")
    if chart_path and data.get("chart_generated"):
        st.divider()
        try:
            # 同上面的理由：不传 use_container_width。图按原始尺寸显示，
            # 数据分析那边生成的图本来就有合理的 figsize。
            st.image(chart_path)
        except Exception as exc:  # noqa: BLE001
            st.caption(f"图生成成功但渲染失败（{exc}），文件在 {chart_path}")
    else:
        st.caption("这一问没有出图（不是每个问题都需要图）。")

    st.caption(f"耗时 {result.get('elapsed_ms')}ms　token {result.get('tokens')}")


# ═══════════════════════════════════════════════════════════════
# Tab 4：知识库——检索 + 引用，逐 token 流出来
# ═══════════════════════════════════════════════════════════════
def render_knowledge_tab() -> None:
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

    history = st.session_state.setdefault("kb_history", [])
    turns = st.session_state.setdefault("kb_turns", [])

    if st.button("清空对话", key="kb_clear"):
        st.session_state["kb_history"] = []
        st.session_state["kb_turns"] = []
        st.rerun()

    for turn in turns:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            _render_citations(turn["chunks"])

    question = st.chat_input("问知识库…（退换货政策、会员积分、促销规则…）")
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("检索中…"):
            try:
                hits = retrieve(question, store)
                chunks = rerank(question, hits)
            except Exception as exc:  # noqa: BLE001
                # 检索这一路失败**不降级**（只有 rerank 失败才降级，那是它自己的事）。
                # 向量库连不上时报"未命中"会让人以为知识库里没这段内容——那是
                # 完全不同的结论，见 retriever.retrieve 的降级语义。
                st.error(f"检索失败：{exc}")
                return

        # 流式：先把答案逐字画出来，最后再落一次完整文本。
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
    with st.expander(f"📎 引用到的段落（{len(chunks)} 条）"):
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
            st.divider()


# ═══════════════════════════════════════════════════════════════
def main() -> None:
    st.title("🛒 多 Agent 电商智能客服")

    with st.sidebar:
        st.header("环境")
        st.markdown(f"- 向量库后端：`{_backend_label()}`")
        try:
            st.markdown(f"- 订单库：`{_db().backend}`")
        except Exception as exc:  # noqa: BLE001
            st.markdown(f"- 订单库：打不开（{_clip(exc, 60)}）")
        st.markdown(f"- 路由上限：{MAX_ROUNDS} 轮")
        st.markdown(f"- worker 超时：{WORKER_TIMEOUT}")
        st.markdown(f"- 每轮回读历史：{HISTORY_TURNS} 轮")

        st.divider()
        st.header("会话")
        _init_agent_session()
        st.caption(
            "多轮历史落在 `data/orchestrator/sessions.db`；"
            "新标签页会是新的一段对话（凭证不进 URL，见 app.py 里的说明）。"
        )
        # 只显示前 12 位。它是**凭证**（拿到它就能读到这个会话的全部历史），
        # 侧边栏会被人截图发出去，所以不整条摆在这里——要看全的用
        # `python -m ecom.orchestrator.cli.session list`。
        st.code(st.session_state["agent_session_id"][:12] + "…", language=None)
        if st.button("新建会话", key="sidebar_new_session"):
            _start_new_session()
            st.rerun()

        st.divider()
        st.caption(
            "界面会把「哪一层慢 / 哪一层挂了」藏起来——超时变成转圈，异常变成红框。"
            "排查时用 `uv run python main.py -v`，那里把每次请求的决策链摊开。"
        )
        st.caption(f"启动于 {datetime.now():%Y-%m-%d %H:%M}")

    tab_agent, tab_order, tab_dash, tab_kb = st.tabs(
        ["💬 智能客服", "📦 订单", "📊 数据看板", "📚 知识库"]
    )
    with tab_agent:
        render_agent_tab()
    with tab_order:
        render_order_tab()
    with tab_dash:
        render_analytics_tab()
    with tab_kb:
        render_knowledge_tab()


if __name__ == "__main__":
    main()
