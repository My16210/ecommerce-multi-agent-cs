"""
调参脚本的切分逻辑。

留出验证的全部价值都建立在"这一半数据从没参与过调参"上。切分一旦失效——
比如跟着 index 走、或者同一道题能同时落进两半——留出集就不再是留出集，
而脚本照样会打印出一张看着很严谨的表。所以这里钉的是切分的**不变量**，
不是某个具体的分法。
"""
from ecom.knowledge.cli.tune_rrf import split_cases, wilson_half_width


def _cases(n: int) -> list:
    return [{"id": f"q{i}", "question": f"问题{i}", "expected_source": "a.md"} for i in range(n)]


def test_split_has_no_overlap_and_covers_everything():
    tune, holdout = split_cases(_cases(100))
    tune_ids = {case["id"] for case in tune}
    holdout_ids = {case["id"] for case in holdout}

    assert not tune_ids & holdout_ids          # 互不相交，这是留出验证的前提
    assert tune_ids | holdout_ids == {f"q{i}" for i in range(100)}


def test_split_is_deterministic():
    """同一种子每次切得一样——否则"留出集"下一次运行就换了一批题"""
    assert split_cases(_cases(50)) == split_cases(_cases(50))


def test_split_does_not_depend_on_order_or_index():
    """切分跟着 id 走，不跟着位置走

    如果按 index 奇偶切，评测集文件一重排，同一道题就会换到另一半去——
    那"它从没被调参看过"这句话就不成立了，而且是**静默**失效的。
    """
    cases = _cases(60)
    shuffled = list(reversed(cases))

    tune_a, holdout_a = split_cases(cases)
    tune_b, holdout_b = split_cases(shuffled)

    assert {c["id"] for c in tune_a} == {c["id"] for c in tune_b}
    assert {c["id"] for c in holdout_a} == {c["id"] for c in holdout_b}


def test_split_ratio_is_roughly_respected():
    tune, holdout = split_cases(_cases(1000), ratio=0.5)
    # 哈希切分只是近似均匀，给一个宽松的区间即可
    assert 400 < len(tune) < 600
    assert 400 < len(holdout) < 600


def test_split_seed_changes_the_partition():
    """换种子应该换一批划分——不然 seed 参数是假的"""
    tune_a, _ = split_cases(_cases(200), seed="a")
    tune_b, _ = split_cases(_cases(200), seed="b")
    assert {c["id"] for c in tune_a} != {c["id"] for c in tune_b}


def test_split_falls_back_to_question_when_id_is_missing():
    """没有 id 就用问题文本——老评测集没有 id 字段，不该因此崩掉"""
    cases = [{"question": f"问题{i}"} for i in range(20)]
    tune, holdout = split_cases(cases)
    assert len(tune) + len(holdout) == 20


# --------------------------------------------------------------------------
# 置信区间
# --------------------------------------------------------------------------


def test_wilson_half_width_shrinks_with_more_samples():
    """样本越大区间越窄——这正是"要更大的评测集才能量出小差异"的根据"""
    assert wilson_half_width(0.6, 1000) < wilson_half_width(0.6, 100)


def test_wilson_half_width_handles_degenerate_input():
    assert wilson_half_width(0.5, 0) == 0.0
