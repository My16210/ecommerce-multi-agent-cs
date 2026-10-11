"""
后台管理端（Streamlit）。**这是给人用的第二个入口，不是 app.py 的一部分。**

    uv sync --extra ui
    uv run streamlit run admin.py

三个页面：退货审核 / 订单与物流浏览 / 知识库审核队列。

【为什么是独立的 admin.py，而不是给 app.py 加一个"管理"Tab】
`app.py` 的 docstring 里写着「不做登录 / 权限：这是离线演示，加一套假登录只会
让演示多几步点击」。那句话现在**依然成立**——因为后台不是它。
把两者合成一个文件的那一天，那句注释就得划掉，而 app.py 的定位（"面向客户的
演示，clone 下来点开就能用"）会被一个密码框毁掉。**分开，两边的前提就都还在。**

【鉴权：fail closed，和 api.py 刚好相反】
`ADMIN_PASSWORD` 从 `.env` 读。**没配就直接拒绝启动**（st.error + st.stop），
而不是"没配就放行"。

这个不对称是刻意的，不是忘了统一：
    api.py    读为主，面向"clone 下来就能 curl 一下"。没配 API_KEY 就放行，
              但 import 时打 WARNING——**读可以默认开放**。
    admin.py  能对**花钱的操作**（批准退货）下结论。没配口令就起不来
              ——**写必须默认关闭**。
默认值的方向不是风格问题：一个"以为配了其实没配"的后台，比一个明知裸奔的
后台危险得多。

【它是演示级的鉴权，这句话要留在使用的人看得见的地方】
明文口令，没有用户、没有角色、没有审计（"审核人"是人手填的一个字符串）。
它挡住的是"随手扫到这个端口的人"。别把它说成"做了权限系统"。

**第二层是端口绑定**：`docker-compose.yml` 里那个服务绑的是
`127.0.0.1:8502`，不是 `0.0.0.0`。两层挡的不是同一件事——口令挡"能连到端口
的人"，绑定挡"根本连不到这个端口的人"。两层都要有。

【三个页面里只有一个能写】
只有「退货审核」页有写操作。另外两页是**只读的浏览**：
  · 订单与物流：直接查库（复用 order_tools 的查询函数，不重写 SQL）
  · 知识库审核队列：这不是新功能——`ecom/knowledge/cli/review.py` 的
    `approve_item()` 本来就是"为了给界面复用"才从 cmd_approve 里拆出来的，
    这个接缝一直留着没人用，现在接上。
"""
import secrets

import streamlit as st

from ecom.knowledge.cli import review as kb_review
from ecom.knowledge.review import queue as kb_queue
from ecom.orchestrator.admin import returns
from ecom.orchestrator.tools.order_tools import (
    RETURN_STATUSES,
    OrderNotFound,
    describe_status,
    list_customer_orders,
    query_order,
)
from ecom.settings import ADMIN_PASSWORD

st.set_page_config(page_title="电商客服后台", page_icon="🔐", layout="wide")


# ═══════════════════════════════════════════════════════════════════════════
# 门禁
# ═══════════════════════════════════════════════════════════════════════════
def check_password(entered: str, expected: str) -> bool:
    """
    口令比对。**一个不依赖 streamlit 的纯函数。**

    抽出来是因为分权这件事真正的保证在这里。而**测试分权的测试很容易变成假绿**：
    现有那套假 streamlit 不记录关键字参数，`st.error("x")` 和 `st.error(msg)`
    在调用记录里长得一模一样——想用"断言出现了 st.error"来证明"没通过口令就
    被挡住了"，那个断言实际证明不了什么。所以判断逻辑单独一个函数，由不依赖
    streamlit 的单元测试盯着。

    比较用 `secrets.compare_digest` 而不是 `==`：`==` 在第一个不同的字节上就
    返回，逐字节的时间差在统计上是可以被利用的。两侧都先 `.encode()`——碰上
    非 ASCII 的 str，`compare_digest` 会直接抛 TypeError，而中文口令是很自然
    会被设出来的东西。
    """
    if not expected:
        # 没配置就是"谁都别进"。**不是"没配置就放行"**——那个默认值是错的。
        return False
    return secrets.compare_digest(entered.encode("utf-8"), expected.encode("utf-8"))


def require_login() -> None:
    if not ADMIN_PASSWORD:
        st.error(
            "没有配置 ADMIN_PASSWORD，后台拒绝启动。\n\n"
            "在 .env 里加一行 `ADMIN_PASSWORD=你的口令`，然后重启。\n\n"
            "**这是刻意的 fail closed**：这个后台能批准退货（花钱的操作），"
            "所以宁可起不来，也不要一个没有门禁的版本。"
        )
        st.stop()

    if st.session_state.get("admin_ok"):
        return

    st.title("🔐 电商客服后台")
    st.caption("这个后台能对花钱的操作下结论（批准退货），所以有门禁。")
    entered = st.text_input("口令", type="password", key="admin_pw")
    if st.button("进入", key="admin_login"):
        if check_password(entered, ADMIN_PASSWORD):
            st.session_state["admin_ok"] = True
            st.rerun()
        else:
            # 不区分"口令错"和"没填"：区分了等于告诉试口令的人"这个值有点对"。
            st.error("口令不对。")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════
# 页面一：退货审核（**唯一能写的页面**）
# ═══════════════════════════════════════════════════════════════════════════
def render_returns_page() -> None:
    st.subheader("退货审核")
    st.caption(
        "**批准只登记人的决定**：不改订单状态、不执行退款（这个项目没有支付网关可调）。"
        "决定一旦落下就不再改动——再审一次不会覆盖前一次的结论。"
    )

    counts = returns.stats()
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("待审核", counts.get(RETURN_STATUSES[0], 0))
    col_b.metric("已通过", counts.get(RETURN_STATUSES[1], 0))
    col_c.metric("已拒绝", counts.get(RETURN_STATUSES[2], 0))

    status = st.selectbox("看哪个状态", list(RETURN_STATUSES), index=0, key="ret_status")
    rows = returns.list_returns(status=status, limit=200)
    if not rows:
        st.info(f"没有「{status}」的退货申请。")
        return

    for row in rows:
        reviewed = row.get("reviewed_at")
        title = f"{row['return_id']}　订单 {row['order_id']}　{row['status']}"
        with st.expander(title):
            st.markdown(f"**申请理由**：{row['reason']}")
            st.markdown(f"**提交时间**：{row['created_at']}")

            with st.container():
                _render_order_snapshot(row["order_id"])

            if reviewed:
                who = row.get("reviewer") or "（未署名）"
                st.success(
                    f"审核痕迹：{reviewed}　{who}　{row.get('decision')}　"
                    f"{row.get('review_note') or '（没写备注）'}"
                )
                continue

            note = st.text_area("审核备注", key=f"note_{row['return_id']}", height=80)
            reviewer = st.text_input("审核人", key=f"who_{row['return_id']}")
            col_pass, col_reject, _ = st.columns([1, 1, 4])
            with col_pass:
                if st.button("通过", key=f"pass_{row['return_id']}"):
                    _review(row["return_id"], RETURN_STATUSES[1], note, reviewer)
            with col_reject:
                if st.button("拒绝", key=f"rej_{row['return_id']}"):
                    _review(row["return_id"], RETURN_STATUSES[2], note, reviewer)


def _review(return_id: str, decision: str, note: str, reviewer: str) -> None:
    result = returns.review_return(return_id, decision, note=note, reviewer=reviewer)
    if not result["ok"]:
        st.error(result["message"])
    elif result["already"]:
        # already 不是失败：它表示"你要的结论不在里面 / 已经在里面了"。
        # 用 warning 而不是 error——用红色会让审核人以为操作失败了，然后再点一次。
        st.warning(result["message"])
    else:
        st.success(result["message"])
        st.rerun()


def _render_order_snapshot(order_id: str) -> None:
    """审核一笔退货之前，把订单和物流摆出来——不然审核人只能凭申请理由做决定。"""
    try:
        found = query_order(order_id)
    except OrderNotFound:
        st.warning(f"订单 {order_id} 在库里查不到（数据被换过了？）")
        return

    order = found["order"]
    st.markdown(
        f"**订单**　{order['order_id']}　{describe_status(order['status'])}　"
        f"金额 {order['amount']}　下单 {order['created_at']}　地区 {order.get('region') or '未知'}"
    )
    logistics = found.get("logistics")
    if logistics:
        st.markdown(
            f"**物流**　{logistics.get('status') or '无状态'}　"
            f"目的地 {logistics.get('to_city') or '未知'}　"
            f"更新时间 {logistics.get('updated_at') or '未知'}"
        )
        # 承运商和运单号在真实数据里是空的（见 schema.sql），这里如实显示"没有"，
        # 后面接一句为什么——空着的字段最容易被当成"忘了填"。
        st.caption("承运商 / 运单号：公开数据集里没有这两项，所以是空的（不是漏填）。")
    else:
        st.caption("这笔订单没有物流记录。")


# ═══════════════════════════════════════════════════════════════════════════
# 页面二：订单与物流浏览（只读）
# ═══════════════════════════════════════════════════════════════════════════
def render_orders_page() -> None:
    st.subheader("订单与物流浏览")
    st.caption(
        "查询复用 `order_tools` 里那两个函数（模型调的是同一份实现）——"
        "**这个页面不另外写一份 SQL**，否则界面看到的和模型看到的迟早不一样。"
    )

    order_id = st.text_input("订单号", key="admin_order_id")
    if order_id.strip():
        try:
            found = query_order(order_id.strip())
        except OrderNotFound:
            st.warning(f"查不到订单号 {order_id.strip()}")
        else:
            st.markdown("**订单**")
            st.json(found["order"], expanded=True)
            st.markdown("**物流**")
            if found["logistics"]:
                st.json(found["logistics"], expanded=True)
            else:
                st.caption("这笔订单没有物流记录。")

    st.divider()
    customer_id = st.text_input("客户号（例如 C20755）", key="admin_customer_id")
    if customer_id.strip():
        rows = list_customer_orders(customer_id.strip())
        if not rows:
            st.warning(f"客户 {customer_id.strip()} 名下没有订单")
        else:
            st.dataframe(rows, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# 页面三：知识库审核队列（只读 + 复用现成的 approve_item）
# ═══════════════════════════════════════════════════════════════════════════
def render_kb_queue_page() -> None:
    st.subheader("知识库审核队列")
    st.caption(
        "模型说「这个问题值得进知识库」，但**它的话不算数**——要人批准，"
        "才会渲染成 md 落到 `data/knowledge/documents/reviewed/`。"
        "和退货审核是同一条原则：模型能判定，不能拍板。"
    )

    counts = kb_queue.stats()
    st.caption("　".join(f"{k}：{v}" for k, v in counts.items()) or "队列是空的")

    items = kb_queue.list_items(status="pending", limit=100)
    if not items:
        st.info("待审核的条目是空的。跑几轮问答之后这里会有东西。")
        return

    for item in items:
        with st.expander(f"#{item['id']}　{item['question']}"):
            st.markdown(item.get("answer") or "（模型没有生成答案）")
            st.caption(
                f"分类 {item.get('category')}　评分 {item.get('score')}　"
                f"被问过 {item.get('ask_count', 1)} 次　"
                f"知识库命中 {item.get('kb_hits')} 条"
            )
            note = st.text_input("审核备注", key=f"kb_note_{item['id']}")
            col_ok, col_no, _ = st.columns([1, 1, 4])
            with col_ok:
                if st.button("批准进知识库", key=f"kb_ok_{item['id']}"):
                    # 复用 CLI 那个函数，一行都不重写——它的返回值就是为界面设计的
                    # （`就已经从 cmd_approve 里拆出来给界面复用`）。
                    result = kb_review.approve_item(item["id"], note=note or None)
                    if result["ok"]:
                        st.success(result["message"])
                        st.rerun()
                    else:
                        st.error(result["message"])
            with col_no:
                if st.button("拒绝", key=f"kb_no_{item['id']}"):
                    kb_queue.set_status(item["id"], "rejected", note=note or None)
                    st.warning(f"已拒绝 #{item['id']}（不进知识库）")
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    require_login()

    with st.sidebar:
        st.header("🔐 后台")
        st.caption("演示级鉴权：一个明文口令，没有用户、没有角色。")
        if st.button("退出", key="admin_logout"):
            st.session_state["admin_ok"] = False
            st.rerun()
        st.divider()
        st.caption(
            "**这里不能做的事**：不执行退款、不改订单状态、不改模型的判定。"
            "后台只登记人的决定。"
        )

    st.title("🔐 电商客服后台")

    tab_returns, tab_orders, tab_kb = st.tabs(["📦 退货审核", "🔍 订单与物流", "📚 知识库队列"])
    with tab_returns:
        render_returns_page()
    with tab_orders:
        render_orders_page()
    with tab_kb:
        render_kb_queue_page()


if __name__ == "__main__":
    main()
