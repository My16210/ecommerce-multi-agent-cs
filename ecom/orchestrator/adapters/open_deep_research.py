"""把 ecom_cs 的 call 级成本采集接到 open_deep_research（外部项目）。

**不改对方的业务代码。** 采集靠 LangChain 的 callback：调用它的图时把
`CallRecorder` 放进 `config["callbacks"]`，每一次 chat model 调用都会经过
`on_chat_model_start` / `on_llm_end`，从中取 model、tokens、latency 和
LangGraph 的节点名。adapter 只在本仓库；对方仓库只多了一个 `.venv`。

【为什么不用 monkeypatch init_chat_model】
callback 是 LangChain 自己的扩展点，图和它的子图都会继承调用方 config 里的
callbacks。monkeypatch 会把"采集"绑到某个内部函数名上，对方一改 import 就静默失效。
**如果 callback 在这版 LangGraph 上没传到模型**，再退回 monkeypatch，并在报告里写清楚。

【没有搜索 key 的降级模式】
open_deep_research 支持 `search_api=none`：研究员只剩 think_tool / ResearchComplete，
没有网页检索。这里显式用这个模式，**不 mock 搜索**。报告的结论只覆盖推理链路。

用法（必须用 open_deep_research 的 venv，因为依赖在那边）：
    cd D:/worksoftspace/open_deep_research
    .venv/Scripts/python.exe D:/worksoftspace/ecom_cs/ecom/orchestrator/adapters/open_deep_research.py --index 1
    # 不传 --index 就跑全部 3 题
"""
import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage

ECOM_ROOT = Path(__file__).resolve().parents[3]          # .../ecom_cs
ODS_ROOT = Path(os.getenv("ODS_ROOT", r"D:\worksoftspace\open_deep_research"))
RUNS_DIR = ECOM_ROOT / "data" / "orchestrator" / "ods_runs"

# **这个文件自己叫 open_deep_research.py**。以脚本方式运行时，Python 会把脚本
# 所在目录放到 sys.path 前面；那里面有个同名普通模块，会抢在对方的命名空间包
# 前面被 import，报「'open_deep_research' is not a package」。把脚本目录从
# sys.path 里摘掉，再显式加对方 src。
_HERE = os.path.normcase(os.path.dirname(os.path.abspath(__file__)))
sys.path[:] = [
    p for p in sys.path
    if p and os.path.normcase(os.path.abspath(p)) != _HERE
]
sys.path.insert(0, str(ECOM_ROOT))
sys.path.insert(0, str(ODS_ROOT / "src"))

from ecom.agentcost import cost_of_call, fmt_cost, summarize_cost  # noqa: E402, I001


QUESTIONS = [
    "比较 LangGraph 和 AutoGen 在多智能体编排上的取舍，给出 3 条工程建议。",
    "设计一个面向电商客服的多智能体系统，说明路由、并行、汇总三个环节各自的失败模式。",
    "一个 RAG 系统的 Recall@5 是 78%，如何定位瓶颈？给出一个分阶段的排查方案。",
]


def _load_env(path: Path) -> None:
    """把 ecom_cs 的 .env 读进环境变量；不打印任何值。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _configure_provider() -> tuple[str, str]:
    """选一个 provider，把它的 key/base_url 映射成 OpenAI 兼容变量。

    只支持 qwen / deepseek（我们有的两个 key）。返回 (配置前缀, 模型名)。
    open_deep_research 的 `get_api_key_for_model()` 只认 openai/anthropic/google
    前缀，所以这里统一走 `openai:<model>` + base_url，避免改它的代码。
    """
    provider = os.getenv("ODS_PROVIDER", "qwen").lower()
    _load_env(ECOM_ROOT / ".env")
    if provider == "deepseek":
        key = os.getenv("DEEPSEEK_API_KEY", "")
        base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    else:
        key = os.getenv("QWEN_API_KEY", "")
        base = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        model = os.getenv("QWEN_MODEL", "qwen-plus")
    if not key:
        raise RuntimeError(f"缺少 {provider.upper()} 的 API key（ecom_cs/.env）")
    os.environ["OPENAI_API_KEY"] = key
    os.environ["OPENAI_BASE_URL"] = base
    os.environ["OPENAI_API_BASE"] = base
    # 不让 LangSmith 意外把这次运行传到外部。
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    return provider, model


def _node_name(metadata: dict) -> str:
    name = (metadata or {}).get("langgraph_node") or (metadata or {}).get("ls_run_name")
    return f"ods/{name}" if name else "ods/unknown"


def _model_name(serialized: dict, metadata: dict, kwargs: dict) -> str:
    for candidate in (
        (metadata or {}).get("ls_model_name"),
        ((serialized or {}).get("kwargs") or {}).get("model"),
        ((serialized or {}).get("kwargs") or {}).get("model_name"),
        ((kwargs or {}).get("invocation_params") or {}).get("model"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return "unknown"


def _usage(response) -> tuple[int, int, bool]:
    """从 LLMResult 里取 in/out；返回 (in, out, given)。"""
    generations = getattr(response, "generations", None) or []
    if generations and generations[0]:
        message = getattr(generations[0][0], "message", None)
        usage = getattr(message, "usage_metadata", None) or {}
        if usage:
            return (
                int(usage.get("input_tokens", 0) or 0),
                int(usage.get("output_tokens", 0) or 0),
                True,
            )
    llm_output = getattr(response, "llm_output", None) or {}
    usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    if usage:
        return (
            int(usage.get("prompt_tokens", 0) or 0),
            int(usage.get("completion_tokens", 0) or 0),
            True,
        )
    return 0, 0, False


class CallRecorder(BaseCallbackHandler):
    """一次问答内所有 chat model 调用的记录器。"""

    def __init__(self):
        self.calls: list[dict] = []
        self._starts: dict = {}
        self._meta: dict = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        self._starts[run_id] = time.perf_counter()
        self._meta[run_id] = {
            "node": _node_name(metadata or {}),
            "model": _model_name(serialized, metadata or {}, kwargs),
        }

    def on_llm_end(self, response, *, run_id, **kwargs):
        started = self._starts.pop(run_id, None)
        meta = self._meta.pop(run_id, {"node": "ods/unknown", "model": "unknown"})
        tokens_in, tokens_out, given = _usage(response)
        self.calls.append({
            "node": meta["node"],
            "kind": "generate",
            "model": meta["model"],
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": int((time.perf_counter() - started) * 1000) if started else 0,
            "source": "llm",
            "usage_given": given,
        })

    def on_llm_error(self, error, *, run_id, **kwargs):
        started = self._starts.pop(run_id, None)
        meta = self._meta.pop(run_id, {"node": "ods/unknown", "model": "unknown"})
        self.calls.append({
            "node": meta["node"],
            "kind": "generate",
            "model": meta["model"],
            "tokens_in": 0,
            "tokens_out": 0,
            "latency_ms": int((time.perf_counter() - started) * 1000) if started else 0,
            "source": "unknown",
            "usage_given": False,
        })


def _base_config(model: str, recorder: CallRecorder) -> dict:
    """所有角色先用同一个模型；有 TAVILY_API_KEY 时走真实检索，否则退回降级模式。"""
    return {
        "callbacks": [recorder],
        "recursion_limit": 100,
        "configurable": {
            "allow_clarification": False,
            # 有 TAVILY_API_KEY 就走真实检索；没有则退回降级模式（只覆盖推理链路）。
            # 显式设置 ODS_SEARCH_API 可以覆盖这个判断。
            "search_api": os.getenv("ODS_SEARCH_API")
            or ("tavily" if os.getenv("TAVILY_API_KEY") else "none"),
            "max_researcher_iterations": 2,
            "max_react_tool_calls": 2,
            "max_concurrent_research_units": 2,
            "max_structured_output_retries": 2,
            "summarization_model": f"openai:{model}",
            "research_model": f"openai:{model}",
            "compression_model": f"openai:{model}",
            "final_report_model": f"openai:{model}",
            "summarization_model_max_tokens": 1000,
            "research_model_max_tokens": 2000,
            "compression_model_max_tokens": 2000,
            "final_report_model_max_tokens": 3000,
        },
    }


async def _run_one(deep_researcher, question: str, model: str, timeout: int) -> dict:
    recorder = CallRecorder()
    config = _base_config(model, recorder)
    started = time.perf_counter()
    error = ""
    report = ""
    try:
        result = await asyncio.wait_for(
            deep_researcher.ainvoke(
                {"messages": [HumanMessage(content=question)]}, config
            ),
            timeout=timeout,
        )
        report = str(result.get("final_report") or "")
    except Exception as exc:  # noqa: BLE001 —— 外部系统边界，失败要如实记
        error = f"{type(exc).__name__}: {exc}"
    return {
        "question": question,
        "model": model,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "report_chars": len(report),
        "report_head": report[:300],
        "error": error,
        "calls": recorder.calls,
        "cost": summarize_cost(recorder.calls),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="只跑第 N 题（1 起）；0 = 全部")
    parser.add_argument("--timeout", type=int, default=240, help="每题超时秒数")
    args = parser.parse_args()

    provider, model = _configure_provider()
    from open_deep_research.deep_researcher import deep_researcher

    questions = QUESTIONS if not args.index else [QUESTIONS[args.index - 1]]
    rows = []
    for number, question in enumerate(questions, 1 if not args.index else args.index):
        print(f"\n=== [{number}] {question} ===")
        row = asyncio.run(_run_one(deep_researcher, question, model, args.timeout))
        row["index"] = number
        rows.append(row)
        for call in row["calls"]:
            print(
                f"  {call['node']:<28} {call['model']:<14} "
                f"in={call['tokens_in']:<6} out={call['tokens_out']:<5} "
                f"{call['latency_ms']:>6}ms  {fmt_cost(cost_of_call(call))}"
            )
        cost = row["cost"]
        print(
            f"  -> 调用 {len(row['calls'])} 次，成本 {fmt_cost(cost['known_usd'])}"
            + (f" + {cost['unknown_calls']} 条未知" if cost["unknown_calls"] else "")
            + (f"，错误：{row['error']}" if row["error"] else f"，报告 {row['report_chars']} 字")
        )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / f"ods-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = summarize_cost([call for row in rows for call in row["calls"]])
    print(f"\nprovider={provider} model={model}  运行记录：{out}")
    print(
        f"总成本 {fmt_cost(total['known_usd'])}"
        + (f" + {total['unknown_calls']} 条未知" if total["unknown_calls"] else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
