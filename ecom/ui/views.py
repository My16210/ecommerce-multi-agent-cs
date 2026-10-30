"""三个"直连视图"：订单 / 数据看板 / 知识库。

它们不在对话主链路里，各自直接调一个子系统。从 `app.py` 拆出来的理由只有一个：
**主链路（会话、流式、图片、侧边栏）已经够长，三个视图再堆在同一个文件里，
改订单表就得到 900 行里去找。**

依赖通过 `ViewContext` 传进来，而不是让视图 import `app`：这样测试可以
把 `db` / `store` 换成假对象，也避免 `app → views → app` 的循环导入。
"""
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class ViewContext:
    """视图需要的东西。`db` / `store` 是**可调用对象**——调用时机由视图决定。"""

    db: Callable[[], object]
    store: Callable[[], object]
    demo: object
    worker_timeout: dict


def _st():
    # 延迟 import：`ecom.ui` 被非 Streamlit 环境 import 时（例如只跑 CLI 测试）
    # 不应该因为缺 streamlit 就在这里炸；真正渲染时再要它。
    import streamlit as st

    return st


def render_order_view(ctx: ViewContext) -> None:
    from ecom.orchestrator.tools.order_tools import (
        OrderNotFound,
        describe_status,
        get_returns,
        query_order,
    )
    from ecom.orchestrator.workers.order import describe_route

    st = _st()
    db = ctx.db()
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
        c1.metric("状态", describe_status(order["status"]))
        c2.metric("金额", f"￥{order['amount']:.2f}")
        c3.metric("客户", order.get("customer_name") or "-")
        c4.metric("地区", order.get("region") or "-")

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
        st.dataframe(returns, width="stretch")
    else:
        st.caption("没有退货申请")


def render_analytics_view(ctx: ViewContext) -> None:
    from ecom.orchestrator.workers.analytics import run_analytics

    st = _st()
    if ctx.demo.enabled():
        st.warning("这是只读演示环境，数据分析（会在服务器上执行生成的代码）没有开放。")
        return

    st.caption(
        "问一句关于销售数据的话，后台会**让模型写一段 pandas 代码、在你机器上跑**，"
        f"再把人话结论写回来。所以这一步真的会跑代码——超时 "
        f"{ctx.worker_timeout.get('analytics')}s，到点子进程会被杀掉。"
    )

    question = st.text_input(
        "要分析什么？",
        value="",
        placeholder="例如：哪个地区卖得最好 / 各商品销量排名 / 月度销售趋势",
        key="analytics_question",
    )
    if not st.button("开始分析", key="analytics_run", icon=":material/play_arrow:") or not question.strip():
        return

    with st.status(
        ":shimmer[生成代码 → 执行 → 自查 → 写结论…]", type="compact", state="running"
    ) as status:
        try:
            result = run_analytics(question.strip())
        except Exception as exc:  # noqa: BLE001
            status.update(label="没跑成", state="error")
            st.error(f"数据分析没跑成：{exc}")
            return
        status.update(label=f"已完成（{result.get('elapsed_ms')} ms）", state="complete")

    data = result.get("data") or {}
    if data.get("review_passed") is False:
        st.warning("这份报告的代码自查没有通过，结论可能不可靠（报告里也标了）。")

    st.markdown(result.get("answer") or "（没有生成报告）")

    chart_path = data.get("chart_path")
    if chart_path and data.get("chart_generated"):
        try:
            st.image(chart_path)
        except Exception as exc:  # noqa: BLE001
            st.caption(f"图生成成功但渲染失败（{exc}），文件在 {chart_path}")
    else:
        st.caption("这一问没有出图（不是每个问题都需要图）。")

    st.caption(f"耗时 {result.get('elapsed_ms')}ms　token {result.get('tokens')}")


def render_knowledge_view(ctx: ViewContext) -> None:
    from ecom.knowledge.generator.answer import generate_answer
    from ecom.knowledge.retriever.reranker import rerank
    from ecom.knowledge.retriever.retriever import retrieve
    from ecom.knowledge.utils import streaming
    from ecom.ui.components import backend_label, render_citations

    st = _st()
    store = ctx.store()
    try:
        count = store.count()
    except Exception as exc:  # noqa: BLE001
        st.error(f"向量库打不开：{exc}")
        return

    if not count:
        st.warning("向量库是空的。先在终端跑一次：\n\n```\nuv run python -m ecom.knowledge.cli.ingest\n```")
        return

    st.caption(f"{backend_label()} 后端，已索引 {count} 个 chunk。答复下方是它引用的段落。")

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
            render_citations(turn["chunks"])

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
                status.update(label="检索失败", state="error")
                st.error(f"检索失败：{exc}")
                return
            status.update(label=f"检索到 {len(chunks)} 段", state="complete")

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
        render_citations(chunks)

    history += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": text},
    ]
    turns.append({"question": question, "answer": text, "chunks": chunks})
