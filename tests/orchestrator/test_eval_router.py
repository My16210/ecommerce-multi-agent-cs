"""路由评测的测试：指标算得对、题集完整、CLI 能跑。"""
import json
from pathlib import Path

from ecom.orchestrator.cli import eval_router

HOLDOUT = Path(eval_router.DEFAULT_DATASET).with_name("router_holdout.jsonl")


def _case(case_id, question, expected, category="order"):
    return {
        "id": case_id,
        "category": category,
        "question": question,
        "expected": expected,
        "history": [],
        "note": "",
    }


def test_dataset_has_unique_ids_and_known_workers():
    cases = eval_router.load_cases()
    assert len(cases) >= 40
    assert len({case["id"] for case in cases}) == len(cases)
    assert all(case["expected"] for case in cases)
    assert all(case["category"] for case in cases)


def test_holdout_dataset_is_frozen_and_not_scored():
    """holdout 只验格式，**不在这里断言分数**。

    holdout 的用途是"改完规则之后看有没有掉"，不是"照着它调词"。一旦把
    50% 写成断言，下一个人就会为了让它变绿而加关键词，护栏就没了。
    要看分数时手动跑：
        python -m ecom.orchestrator.cli.eval_router \\
            --dataset data/orchestrator/router_holdout.jsonl
    """
    assert HOLDOUT.exists(), f"holdout 集不在：{HOLDOUT}"
    cases = eval_router.load_cases(HOLDOUT)
    assert len(cases) == 20
    assert len({case["id"] for case in cases}) == len(cases)
    # 两套题不重题：重了就等于把 holdout 掺进训练集。
    train_ids = {case["id"] for case in eval_router.load_cases()}
    assert {case["id"] for case in cases}.isdisjoint(train_ids)


def test_metrics_count_exact_covered_and_per_worker():
    cases = [
        _case("1", "q1", ["order"]),
        _case("2", "q2", ["order", "knowledge"], category="multi"),
        _case("3", "q3", ["analytics"], category="analytics"),
    ]
    plans = {
        "1": ["order"],                    # exact
        "2": ["order", "knowledge", "analytics"],  # covered，但多带
        "3": ["knowledge"],                # 漏调
    }
    report = eval_router.evaluate(
        cases, lambda case: {"plan": plans[case["id"]], "source": "test"}
    )

    assert report["total"] == 3
    assert report["exact"] == 1
    assert report["covered"] == 2
    assert report["extra"] == 2
    assert report["missed"] == 1
    assert report["by_category"]["multi"]["exact"] == 0
    assert report["by_worker"]["order"]["tp"] == 2
    assert report["by_worker"]["analytics"]["fn"] == 1
    assert report["by_worker"]["knowledge"]["fp"] == 1


def test_report_contains_all_sections():
    report = eval_router.evaluate(
        [_case("1", "q", ["order"])], lambda case: {"plan": ["order"], "source": "rule"}
    )
    text = eval_router.format_report(report)
    assert "exact" in text
    assert "按类别" in text
    assert "按 worker" in text


def test_cli_rule_mode_reads_a_custom_dataset(tmp_path, capsys):
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "T1",
                "category": "order",
                "question": "订单 52644 到哪了",
                "expected": ["order"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    assert eval_router.main(["--dataset", str(dataset)]) == 0
    out = capsys.readouterr().out
    assert "1 题" in out
    assert "exact" in out
