

import streamlit as st

from ecom.analytics.core.graph import build_graph
from ecom.analytics.core.state import make_initial_state

st.set_page_config(page_title="销售数据分析Agent", page_icon="📊")
st.title("📊 销售数据分析Agent")
st.write("输入问题，AI自动写代码分析数据并生成报告")

# 跨问题保留对话历史（Streamlit 每次交互都会重跑脚本，必须放 session_state）
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("请输入你要分析的问题", placeholder="例如：哪个地区卖得最好？")

if st.button("开始分析", type="primary"):
    if not question:
        st.warning("请先输入问题")
    else:
        with st.spinner("正在分析..."):
            app = build_graph()
            state = make_initial_state(question)
            state["messages"] = st.session_state.messages
            result = app.invoke(state)

        # 记住这轮对话，下个问题能带上文
        st.session_state.messages = result.get("messages", [])
        st.session_state.history.append(
            {
                "question": question,
                "report": result["final_report"],
                "chart_path": result.get("chart_path", ""),
                "chart_generated": bool(result.get("chart_generated")),
            }
        )

# 展示所有轮次的结果
for item in st.session_state.history:
    st.divider()
    st.markdown(f"**问：** {item['question']}")
    st.write(item["report"])
    if item["chart_generated"] and item["chart_path"]:
        st.image(item["chart_path"])
