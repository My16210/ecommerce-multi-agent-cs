

from ecom.analytics.core.graph import build_graph
from ecom.analytics.core.state import make_initial_state

if __name__ == "__main__":
    app = build_graph()
    messages = []  # 跨问题的历史对话

    while True:
        question = input("\n请输入你要分析的问题（输入q退出）: ")
        if question.lower() == "q":
            break

        print("=" * 60)
        state = make_initial_state(question)
        state["messages"] = messages  # 带上历史，让 Agent 记得上一轮问过什么
        result = app.invoke(state)

        print("\n" + "=" * 60)
        print(result["final_report"] if result["final_report"] else "（报告还没写）")
        if result.get("chart_generated") and result.get("chart_path"):
            print(f"图表已保存到：{result['chart_path']}")
        messages = result.get("messages", [])  # 更新历史对话
