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

import pytest

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


# ---------------------------------------------------------------------------
# dataco：口径和 sales 那份不一样，错的地方也不一样
# ---------------------------------------------------------------------------
DATACO_CSV = REPO_ROOT / "data" / "analytics" / "datasets" / "dataco" / "data.csv"


def test_every_registered_dataset_has_its_table():
    """注册了数据集就得有对应的表。

    缺表的症状很隐蔽：评测会拿着一个不存在的路径去算真值，抛的是一个和
    "数据集"毫无关系的 FileNotFoundError；而换个入口（比如界面上手动选数据集）
    可能连抛都不抛。约定写在这里：datasets/<名字>/data.csv。
    """
    for name in EVAL_SETS:
        path = REPO_ROOT / "data" / "analytics" / "datasets" / name / "data.csv"
        assert path.exists(), f"数据集 {name} 注册了但没有 {path}"


def test_dataco_truth_uses_the_gross_column_not_the_discounted_one():
    """口径锁：销售额必须用 `Sales`（未打折），不是 `Order Item Total`（折后）。

    两列差约 10%，选错了答案就系统性偏低——而表还在、题还答得出，看不出错。
    """
    rows = load_rows(DATACO_CSV)
    gross = sum(float(r["Sales"]) for r in rows)
    net = sum(float(r["Order Item Total"]) for r in rows)

    # 用近似而不是相等：真值先按大区各自求和再相加，这里的 gross 是按行顺序相加，
    # 浮点加法不满足结合律，两条路会差在 1e-7 这个量级。这**不是** bug，
    # 而是评测题里必须给 tol 的原因——顺带在这里把它钉住，免得有人把它当回归问题查。
    assert get_eval_set("dataco", DATACO_CSV)["总销售额是多少？"]["value"] == pytest.approx(gross, rel=1e-9)
    assert abs(gross - net) > 0.05 * gross, "这份数据里两列必须真的差得出来，否则这条锁等于没锁"


def test_dataco_truth_is_computed_instead_of_written_down(tmp_path):
    """真值必须跟着数据走。

    喂一份自己造的小表进去，五个答案全都得跟着这张表变。这是"真值是算出来的"
    最强的证明——比断言"当前等于 LATAM"强，后者只说明这一刻是对的，
    而写死一个常数同样能通过那种断言（这正是这个模块要修的病）。
    """
    from ecom.analytics.eval.truth import TARGET_PRODUCT

    # 这张小表是**照着每条口径反着设计**的：换一种算法就会得到另一个答案。
    #   销售额   Alpha 20 < Beta 103            → Beta
    #   销量     TARGET 3 < Other Thing 8       → Other Thing
    #   合计     123.0
    #   TARGET 的行只在 Alpha                    → Alpha
    #   延迟率   按行 = Slow 1/2 > Fast 1/4 → Slow
    #            若错按件数加权 = Fast 5/8 > Slow 1/3 → Fast
    # 最后一条是关键：它把"以行为分母"这条口径锁死了，而不是顺带对上。
    csv_text = (
        "Order Id,Market,Product Name,Order Item Quantity,Sales,Shipping Mode,Late_delivery_risk\n"
        f"1,Alpha,{TARGET_PRODUCT},2,10.0,Slow,0\n"
        f"1,Alpha,{TARGET_PRODUCT},1,10.0,Slow,1\n"
        "2,Beta,Other Thing,5,100.0,Fast,1\n"
        "2,Beta,Other Thing,1,1.0,Fast,0\n"
        "2,Beta,Other Thing,1,1.0,Fast,0\n"
        "2,Beta,Other Thing,1,1.0,Fast,0\n"
    )
    path = tmp_path / "tiny.csv"
    path.write_text(csv_text, encoding="utf-8")

    truth = get_eval_set("dataco", path)

    assert truth["哪个大区的销售额最高？"]["value"] == "Beta"
    assert truth["销量最高的商品是什么？"]["value"] == "Other Thing"
    assert truth["总销售额是多少？"]["value"] == 123.0
    assert truth[f"{TARGET_PRODUCT}在哪个大区卖得最多？"]["value"] == "Alpha"
    assert truth["延迟交付风险最高的发货方式是哪个？"]["value"] == "Slow"


def test_registered_datasets_are_real_callables():
    for name, builder in EVAL_SETS.items():
        assert callable(builder), f"{name} 的构造器不是可调用的"
