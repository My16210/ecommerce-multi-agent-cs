"""
命令行入口。三种用法：

    uv run python main.py "订单 100002 到哪了"        # 问一句
    uv run python main.py                            # 交互模式（多轮）
    uv run python main.py --init-db                  # 重建订单库

    --verbose   打印路由决策和每个 worker 的耗时/结论
    --json      只输出一行 JSON（给脚本/其他程序用）

【为什么先做 CLI 而不是先做界面】
界面会掩盖问题：超时被转成一个转圈动画、异常被转成一个红色弹窗，
"哪一层慢、哪一层挂了"全看不见。CLI 把每次请求的决策链摊在眼前，
调试期用它、演示时才上界面——顺序反了会多花很多时间在"猜哪里慢"上。
"""
import argparse
import json
import sys
import time

from cs.core.aggregate import WORKER_LABELS
from cs.core.supervisor import answer
from cs.db.engine import get_db
from cs.db.seed import seed
from cs.settings import MAX_ROUNDS, WORKER_TIMEOUT
from cs.utils.logger import setup_logger

logger = setup_logger(__name__)

LINE = "=" * 66
THIN = "-" * 56


def _safe(text: str) -> str:
    """
    GBK 控制台下把打不出来的字符换掉。

    和 rag 项目同一个坑：Windows 控制台默认 cp936，`¥`（U+00A5）、`✅` 这类字符
    会直接让 print 抛 UnicodeEncodeError——**在回答已经生成之后才崩**，
    用户看到的是崩溃而不是答案。
    项目内的约定是用 ￥（U+FFE5）这类 GBK 里有的替代字符，这里是最后一道保险。
    """
    try:
        text.encode(sys.stdout.encoding or "utf-8")
        return text
    except (UnicodeEncodeError, LookupError):
        encoding = sys.stdout.encoding or "utf-8"
        return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def show(question: str, final: dict, verbose: bool, elapsed_ms: int) -> None:
    print(_safe(LINE))
    print(_safe(final["answer"]))

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


def run_question(question: str, chat_history: list, verbose: bool, as_json: bool) -> dict:
    started = time.perf_counter()
    final = answer(question, chat_history)
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


def interactive(verbose: bool, as_json: bool) -> int:
    # chat_history 只喂给汇总那一步（见 core/state.py 的上下文隔离说明）。
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
        final = run_question(question, history, verbose, as_json)
        history += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": final.get("answer") or ""},
        ]


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(description="多 Agent 电商客服（Supervisor + Worker）")
    parser.add_argument("question", nargs="*", help="要问的问题；不给就进交互模式")
    parser.add_argument("--verbose", "-v", action="store_true", help="打印路由与每个 worker 的明细")
    parser.add_argument("--json", action="store_true", help="只输出一行 JSON")
    parser.add_argument("--init-db", action="store_true", help="重建订单库后退出")
    args = parser.parse_args(argv)

    if args.init_db:
        db = get_db()
        count = seed(db, force=True)
        where = db.path if db.backend == "sqlite" else "MySQL"
        print(_safe(f"订单库已重建：{where}（订单 {count} 条，物流 {db.count('logistics')} 条）"))
        return 0

    print(_safe(f"路由上限 {MAX_ROUNDS} 轮　worker 超时 {WORKER_TIMEOUT}"))

    question = " ".join(args.question).strip()
    if not question:
        return interactive(args.verbose, args.json)

    final = run_question(question, [], args.verbose, args.json)
    # 退出码反映"系统有没有正常工作"，不反映"业务上办没办成"：
    # 转人工是正常结果，不该让脚本判成失败。
    return 0 if final.get("decision") in {"answer", "human"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
