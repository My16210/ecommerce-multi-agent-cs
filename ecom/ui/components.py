"""Streamlit 界面里可复用的渲染件。

`app.py` 只留"装配 + 对话主链路"；决策链、引用列表、文本裁剪这种
**不依赖会话状态**的东西放这里，三个直连视图共用。
"""

from ecom.orchestrator.core.aggregate import WORKER_LABELS


def _st():
    # **不要在模块顶层 import streamlit**：测试会先 import 真 streamlit，
    # 再把假的那份塞进 sys.modules；模块缓存住真的，假的就永远轮不到。
    # 每次渲染现取，才能让两种运行时各自生效。
    import streamlit as st

    return st


def clip(text, limit: int = 160) -> str:
    """折叠空白并截断。`main.py` 有同名实现，两处口径一致（都是"一行、带省略号"）。"""
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def backend_label() -> str:
    from ecom.knowledge.settings import VECTOR_BACKEND

    return VECTOR_BACKEND


def render_decision_chain(final: dict, elapsed_ms: int) -> None:
    """`main.py -v` 那份内容的可视化版。

    刻意保留同样的字段、同样的措辞——两处显示同一件事时，措辞一致才能让人
    相信它们是同一个系统的两个视图，而不是两套各自演化的东西。
    """
    st = _st()
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
                    st.caption(f"错误：{r.get('error')}")
                elif r.get("answer"):
                    st.caption(f"结论：{clip(r['answer'], 120)}")

        st.caption(
            f"决策：{final.get('decision')}（{final.get('decision_reason')}）　"
            f"耗时 {elapsed_ms}ms　token {final.get('tokens')}"
        )


def render_citations(chunks: list) -> None:
    st = _st()
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
            st.caption(clip(chunk.get("text"), 200))
            st.space("small")
