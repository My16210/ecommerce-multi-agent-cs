"""
一次性 JSON 入口：给编排层的 Supervisor 当**子进程**调用。

和 main.py / app.py 的区别只有输入输出形式：
  - main.py   交互式，问一句答一句，给人看
  - app.py    Streamlit，给人看
  - 本脚本    一次性，stdout **只吐一行 JSON**，给程序看

【它叫 analyze，不叫 analyze_json】
原来叫 `analyze_json.py`。合并时改掉了：这个名字该说它的**契约**是
"stdout 只吐一行 JSON"，而不是说它处理的是"json 文件"——它从来也不处理文件。

【为什么编排层不直接 import 这个包，非要起子进程】
合并前是被逼的：上层仓库和本仓库都叫顶层包 `src`，同一个进程里两个 `src`
必然撞名，`from src.x import y` 会静默解析到错误的那一个。
合并后撞名不可能了，**但子进程留着**——理由换成了"这个包执行的是模型生成的
代码，需要一个能被真杀死的边界"。完整版在 `ecom/orchestrator/workers/analytics.py`。

说清楚这一点是有用的：它决定了下面那段 stdout 重定向**不是**为了绕开撞名，
而是因为跨了进程，每条 stdout 都会污染这行 JSON。

【为什么要把 stdout 临时换成 stderr】
本项目的节点里可能有 print（调试残留、matplotlib 的提示等）。只要有一条混进
stdout，上层解析就会失败——而且是"偶尔失败"，最难查。所以运行期间把 stdout
整个重定向到 stderr，JSON 最后用**保存下来的真 stdout** 写。

用法（**从仓库根目录起**，`-m` 靠它找到包）：
    python -m ecom.analytics.cli.analyze "哪个地区卖得最好"
输出（一行）：
    {"ok": true, "report": "...", "chart_path": "...", "tokens": {...}}
"""
import contextlib
import json
import sys
import time

REAL_STDOUT = sys.stdout  # 重定向之前先抓在手里


def main() -> int:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        REAL_STDOUT.write(json.dumps({"ok": False, "error": "没有收到问题"}, ensure_ascii=False))
        REAL_STDOUT.write("\n")
        return 2

    started = time.perf_counter()
    payload = {"ok": False, "error": ""}
    try:
        # 延迟 import：让上面的参数校验能在依赖缺失时也给出可读的 JSON 错误
        from ecom.analytics.core.graph import build_graph
        from ecom.analytics.core.state import make_initial_state

        with contextlib.redirect_stdout(sys.stderr):
            app = build_graph()
            result = app.invoke(make_initial_state(question))

        payload = {
            "ok": True,
            "report": result.get("final_report") or "",
            "chart_path": result.get("chart_path") or "",
            "chart_generated": bool(result.get("chart_generated")),
            # 审查有没有真的通过，要如实带出去——报告里可能已标注了风险
            "review_passed": bool(result.get("review_passed")),
            "tokens": {
                "in": int(result.get("deepseek_prompt_tokens", 0))
                + int(result.get("qwen_prompt_tokens", 0)),
                "out": int(result.get("deepseek_completion_tokens", 0))
                + int(result.get("qwen_completion_tokens", 0)),
            },
        }
    except Exception as exc:  # noqa: BLE001 —— 子进程边界上，任何异常都要变成一行 JSON
        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    payload["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
    REAL_STDOUT.write(json.dumps(payload, ensure_ascii=False))
    REAL_STDOUT.write("\n")
    REAL_STDOUT.flush()
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
