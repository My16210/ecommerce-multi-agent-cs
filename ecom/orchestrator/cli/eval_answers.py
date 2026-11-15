"""答案评测：跑一批题，判命中，出成本。

**判定用「关键事实包含」，不做 LLM judge**：现在的精度不够养判官，
判官一抖，整张表都得重新解释。等判断题量大到人工判不过来再上 judge，
并且必须先量它和人工的一致率。

用法：
    python -m ecom.orchestrator.cli.eval_answers
    python -m ecom.orchestrator.cli.eval_answers --limit 2
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "data" / "orchestrator" / "answer_eval.jsonl"


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


def run_one(question: str) -> tuple[str, str]:
    """跑一次 main.py，切出答案正文和成本行。"""
    result = subprocess.run(
        [sys.executable, "main.py", question, "-v"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    out = result.stdout or ""
    # 输出结构：日志 → ====（66 个等号）→ 答案正文 → ---- 分隔 → 详情
    parts = re.split(r"\n=+\n", out)
    answer = parts[1] if len(parts) > 1 else ""
    answer = re.split(r"\n-{10,}\n", answer)[0].strip()
    match = re.search(r"成本\s+(.+)", out)
    cost = match.group(1).strip() if match else "?"
    return answer, cost


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（调试用）")
    parser.add_argument("--category", default="", help="只跑某一类：policy / order / analytics")
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
    costs = []
    for item in items:
        answer, cost = run_one(item["q"])
        ok = judge(answer, item)
        passed += 1 if ok else 0
        costs.append(cost)
        print(f"[{'PASS' if ok else 'FAIL'}] {item['id']} {item['q']}  成本 {cost}")
        if not ok:
            need = item.get("must_include") or []
            any_keys = item.get("must_include_any") or []
            print(f"       需要：{'/'.join(need) or '(任选其一: ' + '/'.join(any_keys) + ')'}")
            print(f"       实答：{answer[:150]}")

    print(f"\n命中 {passed}/{len(items)}")
    print("逐题成本：" + " | ".join(costs))


if __name__ == "__main__":
    main()
