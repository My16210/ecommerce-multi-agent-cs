"""
画像脚本里的纯计算与"该说的话"。

`percentile` 值得测，因为它算错了不会有任何报错——只会让画像里的 p95 悄悄偏掉，
而分位数恰恰是这个脚本存在的理由（看的是分布，不是平均值）。
`_print_chain` 的免责那行也值得测：联网失败时如果不标注，"搜一次要 5.6 秒"和
"搜不到时重试要 5.6 秒"这两个指向相反优化方向的结论就会被混为一谈。
"""
from ecom.knowledge.cli.profile import _print_chain, percentile

# --------------------------------------------------------------------------
# 分位数
# --------------------------------------------------------------------------


def test_percentile_empty_and_single():
    assert percentile([], 0.5) == 0.0
    assert percentile([5.0], 0.95) == 5.0


def test_percentile_edges_are_min_and_max():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 1.0) == 5.0
    assert percentile(values, 0.5) == 3.0


def test_percentile_interpolates_between_samples():
    # 4 个点的 p50 落在第 2、3 个之间（位置 1.5），线性插值正好是中间值
    assert percentile([1.0, 2.0, 4.0, 8.0], 0.5) == 3.0


def test_high_quantile_on_tiny_sample_is_near_max():
    """样本少时 p95 会贴近最大值——这是正确行为，不是 bug

    所以输出里 p95 必须和样本数一起看；这一点在脚本的使用说明里也提了。
    """
    assert percentile([1.0, 2.0], 0.95) > 1.9


# --------------------------------------------------------------------------
# 联网失败要如实标注
# --------------------------------------------------------------------------


def _snapshot(durations: dict) -> dict:
    return {
        "durations": durations,
        "counts": {name: 1 for name in durations},
        "metrics": {},
    }


def test_chain_warns_when_web_search_returned_nothing(capsys):
    snapshots = [
        _snapshot({"embed": 0.1, "web_search": 5.0, "generate": 1.0}),
        _snapshot({"embed": 0.1, "web_search": 4.0, "generate": 1.0}),
    ]
    _print_chain(snapshots, web_ok=0)
    out = capsys.readouterr().out

    assert "联网搜索返回空" in out
    # 免责的关键是点明"这段是重试耗时"，不然数字会被误读成一次成功搜索的耗时
    assert "重试" in out


def test_chain_stays_quiet_when_web_search_worked(capsys):
    snapshots = [_snapshot({"embed": 0.1, "web_search": 0.5, "generate": 1.0})]
    _print_chain(snapshots, web_ok=1)
    out = capsys.readouterr().out

    assert "联网搜索返回空" not in out


def test_chain_prints_critical_path_not_just_a_sum(capsys):
    """图里检索和联网是并行的，端到端不是相加

    把"各阶段相加"当成端到端，会高估延迟，也就看不见并行到底省没省下东西。
    """
    snapshots = [_snapshot({"embed": 0.1, "vector_search": 0.01, "web_search": 1.0, "generate": 2.0})]
    _print_chain(snapshots, web_ok=1)
    out = capsys.readouterr().out

    assert "关键路径" in out
    assert "并行省下" in out
