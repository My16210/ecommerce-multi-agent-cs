"""路由评测的测试：指标算得对、题集完整、CLI 能跑。"""
import json

from ecom.orchestrator.cli import eval_router


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
