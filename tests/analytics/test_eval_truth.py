"""守评测真值：它得能算对，而且算它的工具必须和被测链路不是同一个。

第一组测试是**回归锁**：把真值换成算出来的那一刻，必须先证明新的算法给出的
答案和原来那串硬编码常数一致（华东 / iPhone15 / 华中 / 1846604 / 华东）。
不一致只有两种可能——要么新算法写错了，要么原来那串常数本来就是错的。
两种都得停下来看，不能直接改成"以新的为准"。

第二组守"独立实现"这个前提。它比看起来重要：这个前提没有任何运行时症状，
某天有人在 truth.py 里 import pandas 图省事，评测照样全绿，
而"用被测工具验证被测工具"这件事就悄悄发生了。
"""
from pathlib import Path

from ecom.analytics.eval.truth import EVAL_SETS, get_eval_set, load_rows

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SALES_CSV = REPO_ROOT / "data" / "analytics" / "datasets" / "sales" / "data.csv"


def test_sales_truth_reproduces_the_old_hardcoded_answers():
    """回归锁：新算法必须复现出原先写死的那五个答案。"""
    expected = get_eval_set("sales", SALES_CSV)

    assert expected["哪个地区卖得最好？"]["value"] == "华东"
    assert expected["销量最高的商品是什么？"]["value"] == "iPhone15"
    assert expected["哪个地区的退货率最高？"]["value"] == "华中"
    assert expected["总销售额是多少？"]["value"] == 1846604
    assert expected["iPhone15在哪个地区卖得最多？"]["value"] == "华东"


def test_sales_truth_keeps_the_agreed_caliber():
    """口径锁：销售额**含**退货订单（NOTES.md 里的默认口径）。

    这条不是重复上一条。上一条只证明"总数对得上"，这条说明总数为什么是这个数——
    如果哪天有人把退货订单剔掉了，总额会变小，上一条会红但看不出原因，这条直接指出口径变了。
    """
    rows = load_rows(SALES_CSV)
    all_amount = sum(float(r["金额"]) for r in rows)
    refunded_amount = sum(float(r["金额"]) for r in rows if r["是否退货"].strip() == "1")

    assert refunded_amount > 0, "这份数据里必须有退货订单，否则这条口径锁等于没锁"
    assert get_eval_set("sales", SALES_CSV)["总销售额是多少？"]["value"] == all_amount
    assert all_amount != refunded_amount


def test_every_question_has_a_matching_truth():
    questions = set(get_eval_set("sales", SALES_CSV))
    assert all(q.endswith(("？", "?")) for q in questions)


def test_truth_module_only_imports_the_standard_library():
    """独立实现的前提：truth.py 只能依赖标准库。

    用 ast 数 import，而不是在源码里搜 "import pandas" 这个字符串。
    第一版就是搜字符串的，结果被本文件上面那句"不许 import pandas"自己误伤了——
    文本匹配分不清"代码"和"提到这段代码的话"，ast 分得清。
    顺带这也比原来更强：它管的是一个**白名单**，任何新加的重依赖都会撞上来说清楚。

    不查 sys.modules 的原因：那会被别的测试顺手 import 进来的 pandas 污染，
    查了等于没查。
    """
    import ast

    src = (REPO_ROOT / "ecom" / "analytics" / "eval" / "truth.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    allowed = {"__future__", "csv", "collections", "math", "pathlib", "statistics"}
    extra = imported - allowed
    assert not extra, (
        f"truth.py 必须独立于被测链路，只允许标准库 {sorted(allowed)}；"
        f"它多 import 了：{sorted(extra)}"
    )


def test_unknown_dataset_raises_instead_of_returning_empty():
    """未注册的数据集要大声报错，不能返回空 dict。

    空 dict 的后果是评测跑出 0/0——要么报"准确率 0%"（看起来像模型不行），
    要么在算平均耗时的地方除以零崩掉（看不出跟数据集有关）。
    """
    try:
        get_eval_set("no_such_dataset", SALES_CSV)
    except KeyError as e:
        assert "no_such_dataset" in str(e)
    else:
        raise AssertionError("未注册的数据集应该抛 KeyError")


def test_registered_datasets_are_real_callables():
    for name, builder in EVAL_SETS.items():
        assert callable(builder), f"{name} 的构造器不是可调用的"
