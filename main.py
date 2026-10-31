"""
命令行入口。三种用法：

    uv run python main.py "订单 52644 到哪了"        # 问一句
    uv run python main.py                            # 交互模式（多轮）
    uv run python main.py --init-db                  # 重建订单库

    --verbose   打印路由决策和每个 worker 的耗时/结论
    --json      只输出一行 JSON（给脚本/其他程序用）
    --stream    边生成边打印答复（和 --json 互斥；只有汇总那一步是真逐 token）

【为什么先做 CLI 而不是先做界面】
界面会掩盖问题：超时被转成一个转圈动画、异常被转成一个红色弹窗，
"哪一层慢、哪一层挂了"全看不见。CLI 把每次请求的决策链摊在眼前，
调试期用它、演示时才上界面——顺序反了会多花很多时间在"猜哪里慢"上。
"""
import argparse
import json
import time

from ecom.orchestrator.core.aggregate import WORKER_LABELS
from ecom.orchestrator.core.supervisor import (
    HANDOFF_REQUESTED,
    answer,
    answer_stream,
    degrade_handoff,
)
from ecom.orchestrator.db.engine import get_db
from ecom.orchestrator.db.seed import seed
from ecom.orchestrator.settings import MAX_ROUNDS, WORKER_TIMEOUT

# GBK 控制台下把打不出来的字符换掉。实现在 utils/console.py——`cli/memory.py`
# 打印记忆摘要时也用同一份（同一个坑，两处各写一份迟早分叉，而且**只在别人的
# 终端上才看得见**）。这里保留 `_safe` 这个名字只是为了不动下面那一堆调用点。
from ecom.orchestrator.utils.console import safe as _safe
from ecom.orchestrator.utils.logger import setup_logger

logger = setup_logger(__name__)

LINE = "=" * 66
THIN = "-" * 56


def show(question: str, final: dict, verbose: bool, elapsed_ms: int,
         streamed: bool = False) -> None:
    """打印一轮结果。`streamed=True` 表示正文已经边生成边打过了，这里不再重复。"""
    if not streamed:
        print(_safe(LINE))
        print(_safe(final["answer"]))
    _show_meta(question, final, verbose, elapsed_ms)


def _show_meta(question: str, final: dict, verbose: bool, elapsed_ms: int) -> None:
    sources = final.get("sources") or []
    if sources:
        print(_safe(f"\n{THIN}\n来源：{'、'.join(str(s) for s in sources)}"))

    if verbose:
        print(_safe(f"\n{THIN}"))
        print(_safe(f"问题：{question}"))
        print(
            _safe(
                f"路由：{'、'.join(final.get('plan') or []) or '（空）'}"
                f"　来源 {final.get('router_source')}　置信度 {final.get('plan_confidence', 0):.2f}"
            )
        )
        print(_safe(f"      理由：{final.get('plan_reason')}"))
        for r in final.get("results") or []:
            label = WORKER_LABELS.get(r.get("worker"), r.get("worker"))
            state = "成功" if r.get("ok") else "失败"
            mark = "（未答上）" if r.get("insufficient") else ""
            print(
                _safe(
                    f"  · 第{r.get('round')}轮 {label}：{state}{mark} "
                    f"{r.get('elapsed_ms')}ms token={r.get('tokens')}"
                )
            )
            if not r.get("ok"):
                print(_safe(f"      错误：{r.get('error')}"))
            elif r.get("answer"):
                print(_safe(f"      结论：{_clip(r['answer'], 120)}"))
        # 汇总方式已经写在 decision_reason 里了（aggregate 拼的），这里不再重复。
        print(_safe(f"决策：{final.get('decision')}（{final.get('decision_reason')}）"))

    print(_safe(f"{THIN}\n耗时 {elapsed_ms} ms　token {final.get('tokens')}"))


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _run_streaming(question: str, chat_history: list, verbose: bool, started: float) -> dict:
    """
    边生成边打。正文直接落到 stdout，所以最后**不重复打印**。

    汇总中途失败时服务端会退回拼接，已经打出去的字收不回来——终端没有"重画
    这一行"的能力。这时把最终答复完整再打一遍，并说明为什么要重复，而不是
    假装前后一致。
    """
    print(_safe(LINE))
    pieces: list[str] = []
    final: dict = {}
    # 转人工时 supervisor 先吐一句占位话术，**有没有真工单要等 final 才知道**。
    # 终端打出去的字收不回来，所以只要已收的字还是这句的前缀就先扣住。
    held = ""
    for event in answer_stream(question, chat_history):
        kind = event.get("type")
        if kind == "delta":
            piece = event.get("text") or ""
            pieces.append(piece)
            if HANDOFF_REQUESTED.startswith(held + piece):
                held += piece
                continue
            if held:
                print(_safe(held), end="", flush=True)
                held = ""
            print(_safe(piece), end="", flush=True)
        elif kind == "reset":
            # 重路由：上一轮已经打出去的字收不回来，但至少不能把它当成新一轮
            # 正文继续拼。终端没有重画能力，所以明确分隔并把记录清空。
            pieces.clear()
            print(_safe("\n（重新规划，以下是新一轮答复）"))
        elif kind == "final":
            # CLI 没有身份层，开不出工单：这里把"好的，需要人工客服。"换成
            # "当前环境未开通人工客服"。真带了 handoff 的话原样保留。
            final = degrade_handoff(event.get("state") or {})
            if held and final.get("decision") == "human":
                # 占位话术作废，而且它确实没打出去——当作从没收到过。
                held = ""
                pieces.clear()
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    print()

    if held:
        # 没有 final 就断流了：扣下的就是屏幕上的全部内容，补打出来。
        print(_safe(held), end="", flush=True)
        pieces.append(held)
    streamed = "".join(pieces).strip()
    answer = str(final.get("answer") or "").strip()
    if answer and answer != streamed:
        if streamed:
            print(_safe("（上面是生成过程中的片段，以完整答复为准）"))
        print(_safe(answer))

    _show_meta(question, final, verbose, elapsed_ms)
    return final


def run_question(question: str, chat_history: list, verbose: bool, as_json: bool,
                 stream: bool = False) -> dict:
    started = time.perf_counter()

    if stream and not as_json:
        return _run_streaming(question, chat_history, verbose, started)

    # CLI 不经过 API，没有身份和会话库：转人工只会得到一句占位话术。这里换成
    # "当前环境未开通人工客服"，不假装已经转接；有工单时（将来接入）保持原样。
    final = degrade_handoff(answer(question, chat_history))
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    if as_json:
        print(
            json.dumps(
                {
                    "request_id": final.get("request_id"),
                    "answer": final.get("answer"),
                    "sources": final.get("sources"),
                    "decision": final.get("decision"),
                    "plan": final.get("plan"),
                    "tokens": final.get("tokens"),
                    "elapsed_ms": elapsed_ms,
                },
                ensure_ascii=False,
            )
        )
    else:
        show(question, final, verbose, elapsed_ms)
    return final


def interactive(verbose: bool, as_json: bool, stream: bool = False) -> int:
    # chat_history 喂给 router（指代消解）和各 worker；汇总节点刻意不读它
    # （见 core/state.py 的上下文隔离说明）。
    history: list = []
    print(_safe("多 Agent 电商客服。输入问题回车，quit / 退出 结束。"))
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"} or question in {"退出", "再见"}:
            return 0
        final = run_question(question, history, verbose, as_json, stream)
        history += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": final.get("answer") or ""},
        ]


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(description="多 Agent 电商客服（Supervisor + Worker）")
    parser.add_argument("question", nargs="*", help="要问的问题；不给就进交互模式")
    parser.add_argument("--verbose", "-v", action="store_true", help="打印路由与每个 worker 的明细")
    parser.add_argument("--json", action="store_true", help="只输出一行 JSON")
    parser.add_argument(
        "--stream", action="store_true", help="边生成边打印答复（和 --json 互斥）"
    )
    parser.add_argument("--init-db", action="store_true", help="重建订单库后退出")
    args = parser.parse_args(argv)

    if args.stream and args.json:
        parser.error("--stream 和 --json 不能同时用：JSON 必须是完整的一行，流式会把它切开")

    if args.init_db:
        db = get_db()
        count = seed(db, force=True)
        where = db.path if db.backend == "sqlite" else "MySQL"
        print(_safe(f"订单库已重建：{where}（订单 {count} 条，物流 {db.count('logistics')} 条）"))
        return 0

    print(_safe(f"路由上限 {MAX_ROUNDS} 轮　worker 超时 {WORKER_TIMEOUT}"))

    question = " ".join(args.question).strip()
    if not question:
        return interactive(args.verbose, args.json, args.stream)

    final = run_question(question, [], args.verbose, args.json, args.stream)
    # 退出码反映"系统有没有正常工作"，不反映"业务上办没办成"：
    # 转人工是正常结果，不该让脚本判成失败。
    return 0 if final.get("decision") in {"answer", "human"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
