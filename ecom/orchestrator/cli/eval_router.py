"""路由评测：把"这句话该交给谁"从口头判断变成一张可复跑的分数表。

【为什么先做规则基线】
`route()` 有三层：LLM → 关键词规则 → 默认 knowledge。没有评测时，大家只看
到"真跑那次路由对了"，看不到它错在哪一类问题。规则层是**确定性**的，适合当
基线：它每一次都给出同一个答案，所以分数的变化只可能来自题集或规则本身。

    python -m ecom.orchestrator.cli.eval_router
    python -m ecom.orchestrator.cli.eval_router --show-errors
    python -m ecom.orchestrator.cli.eval_router --llm --limit 20

`--llm` 会真的调模型（每题一次路由调用，不跑 worker/汇总），要花钱，所以默认
不跑；规则模式完全离线。

【指标怎么读】
- `exact`：计划集合完全相等。最严，多带一个 worker 也算错。
- `covered`：期望的 worker 全被调到了。多带不算错——路由错一次的代价不对称
  （漏调 = 少答一半，多调 = 多花一次调用），所以这个指标更接近"用户会不会
  觉得答漏了"。
- 每个 worker 的 P/R/F1：多带会把 precision 拉低，漏调会把 recall 拉低。

【题集是标注，不是真理】
`expected` 是"按产品意图应该调谁"，由人写；有歧义的题在 `note` 里写清楚。
改代码让分数变好不是目的——**改题让分数变好才是自欺**。
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from ecom.orchestrator.core.router import WORKER_NAMES, _cold_plan, _rule_route
from ecom.orchestrator.utils.entities import carried_order_id
from ecom.settings import DATA_DIR

DEFAULT_DATASET = DATA_DIR / "orchestrator" / "router_eval.jsonl"


def load_cases(path: Path = DEFAULT_DATASET) -> list:
    """读 JSONL。坏行直接报错并指出行号——静默跳过会让分母悄悄变小。"""
    cases = []
    seen = set()
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno} 不是合法 JSON：{exc}") from exc

        case_id = str(item.get("id") or "").strip()
        question = str(item.get("question") or "").strip()
        expected = [str(name) for name in item.get("expected") or []]
        if not case_id or not question or not expected:
            raise ValueError(f"{path}:{lineno} 缺 id/question/expected")
        if case_id in seen:
            raise ValueError(f"{path}:{lineno} id 重复：{case_id}")
        unknown = [name for name in expected if name not in WORKER_NAMES]
        if unknown:
            raise ValueError(f"{path}:{lineno} 未知 worker：{unknown}")
        seen.add(case_id)
        cases.append(
            {
                "id": case_id,
                "category": str(item.get("category") or "uncategorized"),
                "question": question,
                "expected": expected,
                "history": list(item.get("history") or []),
                "note": str(item.get("note") or ""),
            }
        )
    return cases


def predict_rule(case: dict) -> dict:
    """规则层基线。没命中关键词时走 `_cold_plan()`，和 route() 的兜底一致。"""
    plan = _rule_route(
        case["question"], carried_order_id(case["question"], case["history"])
    )
    return {"plan": plan or _cold_plan(), "source": "rule" if plan else "fallback"}


def predict_llm(case: dict) -> dict:
    """真模型路由（要花钱）。导入放函数里，规则模式不需要模型依赖。"""
    from ecom.orchestrator.core.router import route

    result = route(case["question"], chat_history=case["history"])
    return {"plan": result["plan"], "source": result["source"]}


def evaluate(cases: list, predictor) -> dict:
    """跑一遍并算分。predictor(case) → {"plan": [...], "source": "..."}。"""
    rows = []
    exact = covered = extra = missed = 0
    per_worker = {name: Counter() for name in WORKER_NAMES}
    per_category = {}

    for case in cases:
        predicted = predictor(case)
        plan = list(predicted.get("plan") or [])
        expected = list(case["expected"])
        expected_set, plan_set = set(expected), set(plan)

        is_exact = expected_set == plan_set
        is_covered = expected_set <= plan_set
        exact += int(is_exact)
        covered += int(is_covered)
        extra += len(plan_set - expected_set)
        missed += len(expected_set - plan_set)

        for name in WORKER_NAMES:
            if name in expected_set and name in plan_set:
                per_worker[name]["tp"] += 1
            elif name in expected_set:
                per_worker[name]["fn"] += 1
            elif name in plan_set:
                per_worker[name]["fp"] += 1

        bucket = per_category.setdefault(case["category"], {"n": 0, "exact": 0, "covered": 0})
        bucket["n"] += 1
        bucket["exact"] += int(is_exact)
        bucket["covered"] += int(is_covered)

        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "expected": expected,
                "plan": plan,
                "source": predicted.get("source", ""),
                "exact": is_exact,
                "covered": is_covered,
            }
        )

    total = len(cases)
    return {
        "total": total,
        "exact": exact,
        "covered": covered,
        "extra": extra,
        "missed": missed,
        "by_category": per_category,
        "by_worker": per_worker,
        "rows": rows,
    }


def format_report(report: dict, dataset: Path = DEFAULT_DATASET) -> str:
    total = report["total"] or 1
    lines = [
        f"路由评测：{report['total']} 题（{dataset}）",
        f"exact   {report['exact']}/{report['total']}"
        f"（{report['exact'] / total:.1%}）　"
        f"covered {report['covered']}/{report['total']}"
        f"（{report['covered'] / total:.1%}）",
        f"多带 worker {report['extra']} 次，漏调 {report['missed']} 次",
        "",
        "按类别：",
    ]
    for category, bucket in sorted(report["by_category"].items()):
        n = bucket["n"] or 1
        lines.append(
            f"  {category:<12} n={bucket['n']:<3} "
            f"exact {bucket['exact'] / n:5.1%}  covered {bucket['covered'] / n:5.1%}"
        )

    lines.extend(["", "按 worker："])
    for name in WORKER_NAMES:
        counter = report["by_worker"][name]
        tp, fp, fn = counter["tp"], counter["fp"], counter["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        lines.append(
            f"  {name:<12} P={precision:5.1%}  R={recall:5.1%}  F1={f1:5.1%}"
            f"  (tp={tp} fp={fp} fn={fn})"
        )
    return "\n".join(lines)


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(description="路由评测（默认规则基线，不联网）")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--llm", action="store_true", help="用真模型路由（会花钱）")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题，给 --llm 控预算")
    parser.add_argument("--json", type=Path, default=None, help="把完整报告写成 JSON")
    parser.add_argument("--show-errors", action="store_true", help="列出 exact=False 的题")
    args = parser.parse_args(argv)

    cases = load_cases(args.dataset)
    if args.limit is not None:
        cases = cases[: max(0, args.limit)]
    if not cases:
        print("题集是空的，没什么可评的。")
        return 1

    predictor = predict_llm if args.llm else predict_rule
    report = evaluate(cases, predictor)
    report["mode"] = "llm" if args.llm else "rule"
    report["dataset"] = str(args.dataset)
    print(format_report(report, args.dataset))

    if args.show_errors:
        print("\n错题：")
        for row in report["rows"]:
            if not row["exact"]:
                print(
                    f"  {row['id']} [{row['category']}] {row['question']}\n"
                    f"      期望 {row['expected']} → 实际 {row['plan']}（{row['source']}）"
                )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n报告已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
