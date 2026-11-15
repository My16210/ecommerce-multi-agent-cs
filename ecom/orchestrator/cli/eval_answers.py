"""答案评测：跑一批题，判命中，出成本，并把每次运行的全文落盘。

**判定用「关键事实包含」，不做 LLM judge**：现在的精度不够养判官，
判官一抖，整张表都得重新解释。等判断题量大到人工判不过来再上 judge，
并且必须先量它和人工的一致率。

**为什么要把 answer 写进 run artifact**：CLI 这条路不写 sessions.db，
只 `print` 的话，跑完就只剩一个通过/不通过。要回答"答案变短是不是漏了
信息"，必须能逐题 diff 改前改后的正文。生产 trace 刻意不存全文
（用户内容 + 没有 TTL），评测证据落在这里。

用法：
    python -m ecom.orchestrator.cli.eval_answers
    python -m ecom.orchestrator.cli.eval_answers --limit 2
    python -m ecom.orchestrator.cli.eval_answers --category policy
    python -m ecom.orchestrator.cli.eval_answers --out /tmp/run.jsonl
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from ecom.orchestrator.utils.pricing import fmt_cost
from ecom.orchestrator.utils.trace import LOGS_DIR

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "data" / "orchestrator" / "answer_eval.jsonl"
RUNS_DIR = ROOT / "data" / "orchestrator" / "answer_eval_runs"


def judge(answer: str, item: dict) -> bool:
    """关键事实全部命中、且 any 组至少中一个、且没有出现禁词。

    比较统一转小写：模型会把商品名写成全小写（实测 A02 答的是
    "perfect fitness perfect rip deck"），而金标是首字母大写。
    不转的话，答对了也会判错——而它看起来就像模型答错。
    """
    low = (answer or "").lower()
    for key in item.get("must_include") or []:
        if str(key).lower() not in low:
            return False
    any_keys = item.get("must_include_any") or []
    if any_keys and not any(str(key).lower() in low for key in any_keys):
        return False
    for key in item.get("must_not_include") or []:
        if str(key).lower() in low:
            return False
    return True


def parse_result(stdout: str) -> dict | None:
    """从 `main.py --json` 的输出里取最后一行合法 JSON。

    日志可能混在 stdout（不同终端/配置下），所以不假定"整个 stdout 就是一行"。
    从后往前找，和 analytics 子进程解析 `_parse_stdout` 同一个理由。
    """
    for line in reversed((stdout or "").strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "answer" in data:
            return data
    return None


def run_one(question: str) -> dict:
    """跑一次 main.py，返回 {"answer", "tokens", "cost", ...}。

    用 `--json` 而不是解析 `-v` 的格式化文本：正文里有换行、分隔线、
    成本文案，正则迟早会切错；JSON 是给程序看的契约。
    """
    result = subprocess.run(
        [sys.executable, "main.py", question, "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    parsed = parse_result(result.stdout)
    if parsed is None:
        return {
            "answer": "",
            "tokens": {},
            "cost": None,
            "request_id": "",
            "elapsed_ms": 0,
            "error": (result.stderr or result.stdout or "没有可解析的 JSON")[-300:],
        }
    return parsed


def trace_diagnostics(request_id: str, path: Path | None = None) -> dict:
    """按 request_id 从 trace 里取 analytics 的自报诊断字段。

    `main.py --json` 是结果契约，不该为了评测塞 worker 私有字段；trace 本来
    就是放这种可观测信息的地方。找不到就返回空 dict（评测照常跑，字段记 0）。
    """
    if not request_id:
        return {}
    log = path or (LOGS_DIR / "requests.jsonl")
    try:
        lines = log.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("request_id") != request_id:
            continue
        return dict((entry.get("workers") or {}).get("analytics") or {})
    return {}


def cost_text(cost) -> str:
    """成本的人类可读格式：已知部分 + 未知条数，未知绝不省略。"""
    if not isinstance(cost, dict):
        return "未知"
    text = fmt_cost(cost.get("known_usd"))
    unknown = int(cost.get("unknown_calls") or 0)
    if unknown:
        text += f" + {unknown} 条未知"
    return text


def write_run(rows: list, path: Path) -> Path:
    """把一次运行的逐题结果写成 JSONL。answer 全文在这里，不在生产 trace。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（调试用）")
    parser.add_argument("--category", default="", help="只跑某一类：policy / order / analytics")
    parser.add_argument("--out", type=Path, default=None, help="运行结果 JSONL 的路径")
    args = parser.parse_args()

    items = [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.category:
        items = [item for item in items if item.get("category") == args.category]
    if args.limit:
        items = items[: args.limit]

    passed = 0
    rows = []
    for item in items:
        result = run_one(item["q"])
        answer = str(result.get("answer") or "")
        ok = judge(answer, item)
        passed += 1 if ok else 0
        diag = trace_diagnostics(result.get("request_id", ""))
        retry_count = int(diag.get("retry_count", 0) or 0)
        review_round = int(diag.get("review_round", 0) or 0)
        rows.append({
            "id": item["id"],
            "category": item.get("category", ""),
            "question": item["q"],
            "answer": answer,
            "passed": ok,
            "tokens": result.get("tokens") or {},
            "cost": result.get("cost"),
            "request_id": result.get("request_id", ""),
            "elapsed_ms": result.get("elapsed_ms", 0),
            "retry_count": retry_count,
            "review_round": review_round,
            "error": result.get("error", ""),
        })
        print(
            f"[{'PASS' if ok else 'FAIL'}] {item['id']} {item['q']}  "
            f"成本 {cost_text(result.get('cost'))}  重试 {retry_count}  审查 {review_round} 轮"
        )
        if not ok:
            need = item.get("must_include") or []
            any_keys = item.get("must_include_any") or []
            print(f"       需要：{'/'.join(need) or '(任选其一: ' + '/'.join(any_keys) + ')'}")
            print(f"       实答：{answer[:150]}")

    out = args.out or RUNS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    write_run(rows, out)

    print(f"\n命中 {passed}/{len(items)}")
    print("逐题成本：" + " | ".join(cost_text(row["cost"]) for row in rows))
    print(f"结果已写入 {out}")


if __name__ == "__main__":
    main()
