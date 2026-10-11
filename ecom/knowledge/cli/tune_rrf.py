"""
RRF 两路权重的调参 + 留出验证
===============================

为什么要有这个脚本，而不是"拍一个权重写进配置"
----------------------------------------------
chunk 级评测集上，等权 RRF 有个稳定的性质：**术语型吃满、改写型吃亏**。
想让它更好，最自然的动作是给 BM25 加权重。但**在同一份评测集上挑一个
数字最好看的权重，就是过拟合**——挑出来的不是"更懂检索的权重"，是"更懂这 152 道题
的权重"。8 个候选里挑最好的，挑中噪声的概率相当高（见下面的置信区间）。

换语料后在 152 题上实测，这个性质照旧，只是数字变了：
等权（比例 1.00）术语型 100.0% / 改写型 81.4%；
把 BM25 权重压到 0.25，术语型掉到 96.0%、改写型升到 83.7%——
拿术语型换改写型，总量没动（两档 Recall 都是 88.2%）。
留出集上比例 0.25 比等权高 4.8%，但 95% 区间约 ±9.8%，**落在噪声里**，
所以结论和上一版一致：保持等权。（换语料前那版是"术语型 +10.0pp、改写型 -8.8pp"，
方向相同、幅度不同，正说明这类数字不能跨语料照搬。）

所以这个脚本把"调参"和"验证"分到两份互不相交的数据上：

  1. 按题 id 的哈希**确定性地**切成 tune / holdout 两半（同一种子每次切得一样）；
  2. 只在 tune 那一半上做网格搜索，挑出目标指标最高的权重；
  3. 拿这一组权重到 holdout 上跑**一次**，和等权基线比；
  4. 如果 holdout 上没有稳定收益，就照实说没有，默认保持等权。

第 4 步是它存在的主要理由：一个旋钮如果只在调参的那半数据上有效，
那它不是功能，是噪声。

为什么不重跑检索
----------------
embedding 那 100ms/次占了大头，而**融合是纯计算**。所以两路原始名次只取一次，
之后 8 个权重组合全在这批名次上重算——快，而且保证了所有候选看到的是同一批输入，
不同组合之间的差异只来自权重本身。

用法
----
    uv run python tune_rrf.py                  # 目标：总体 Recall@5
    uv run python tune_rrf.py --metric mrr
    uv run python tune_rrf.py --top-k 10
"""
import argparse
import hashlib

from ecom.knowledge.retriever.fusion import reciprocal_rank_fusion
from ecom.knowledge.retriever.retriever import route_rankings
from ecom.knowledge.settings import RRF_K, TOP_K
from ecom.knowledge.utils.eval import load_eval_set, match_positions
from ecom.knowledge.vector_store import get_vector_store

# 只在"BM25 权重 / 向量权重"这个比例上搜，向量权重固定 1.0——
# 融合分只在同一题内部比大小，绝对尺度不影响排序，所以只有比例有意义。
WEIGHT_RATIOS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]

TYPE_LABELS = {"paraphrase": "改写型", "term": "术语型"}


def split_cases(cases: list, ratio: float = 0.5, seed: str = "rrf") -> tuple:
    """
    按 id 的哈希确定性地切分。**不能用 index 奇偶**——评测集一旦重排文件顺序，
    切分就会跟着变，那"同一条题永远在同一半"这个前提就没了，
    留出集也就不再是"从没被看过"的。
    """
    tune, holdout = [], []
    threshold = ratio * 1000
    for case in cases:
        digest = hashlib.md5(f"{seed}:{case.get('id', case['question'])}".encode())
        bucket = int(digest.hexdigest()[:8], 16) % 1000
        (tune if bucket < threshold else holdout).append(case)
    return tune, holdout


def score(subset: list, rankings: dict, weights: dict, top_k: int) -> dict:
    """在一半数据上按给定权重跑一遍，返回总体与分题型的指标"""
    hits = 0
    rr_sum = 0.0
    groups = {}

    for case in subset:
        vector_hits, keyword_hits = rankings[case["id"]]
        fused = reciprocal_rank_fusion(
            {"vector": vector_hits, "bm25": keyword_hits}, k=RRF_K, weights=weights
        )[:top_k]

        positions = match_positions(case, fused)
        hit = bool(positions)
        hits += hit
        rr_sum += 1 / positions[0] if hit else 0.0

        group = groups.setdefault(case.get("type", "全部"), {"hits": 0, "total": 0, "rr": 0.0})
        group["total"] += 1
        group["hits"] += hit
        group["rr"] += 1 / positions[0] if hit else 0.0

    total = len(subset) or 1
    return {
        "recall": hits / total,
        "mrr": rr_sum / total,
        "groups": groups,
    }


def _fmt_row(label: str, metrics: dict, width: int = 10) -> str:
    parts = [f"{label:<{width}}", f"{metrics['recall']:>9.1%}", f"{metrics['mrr']:>8.3f}"]
    for name in ("paraphrase", "term"):
        group = metrics["groups"].get(name)
        parts.append(f"{group['hits'] / group['total']:>9.1%}" if group and group["total"] else " " * 9)
    return "  ".join(parts)


def _header(width: int = 10) -> str:
    return "  ".join([
        f"{'':<{width}}", f"{'Recall@K':>9}", f"{'MRR':>8}", f"{'改写型':>9}", f"{'术语型':>9}",
    ])


def wilson_half_width(p: float, n: int) -> float:
    """95% 置信区间的半宽（正态近似）。用来判断"这点差异算不算数"""
    if n <= 0:
        return 0.0
    return 1.96 * ((p * (1 - p) / n) ** 0.5)


def main():
    parser = argparse.ArgumentParser(description="RRF 两路权重调参 + 留出验证")
    parser.add_argument("--metric", choices=["recall", "mrr"], default="recall",
                        help="在调参集上按哪个指标挑权重")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--set-path", default=None)
    args = parser.parse_args()

    cases = load_eval_set(args.set_path)
    if not cases:
        print("没找到评测集。先运行：uv run python build_eval_set.py")
        return

    tune, holdout = split_cases(cases)
    print(f"评测集 {len(cases)} 题 → 调参 {len(tune)} / 留出 {len(holdout)}"
          f"（按 id 哈希确定性切分）\n")

    store = get_vector_store()
    try:
        if store.count() == 0:
            print("向量库是空的，请先运行：uv run python ingest.py")
            return

        # 两路原始名次只取一次。embedding 是这里唯一贵的一步，融合是纯计算。
        print(f"取 {len(cases)} 题的两路名次（embedding 各一次）…")
        rankings = {}
        for case in cases:
            rankings[case["id"]] = route_rankings(case["question"], store)

        # ---------------------------------------------------------------- 调参
        print(f"\n【调参集】网格搜索（比例 = BM25 权重 / 向量权重，{'指标：' + args.metric}）")
        print(_header())
        print("-" * 56)

        tuned = []
        for ratio in WEIGHT_RATIOS:
            weights = {"vector": 1.0, "bm25": ratio}
            metrics = score(tune, rankings, weights, args.top_k)
            tuned.append((ratio, metrics))
            print(_fmt_row(f"{ratio:.2f}", metrics))

        best_ratio, best_metrics = max(tuned, key=lambda item: item[1][args.metric])
        print("-" * 56)
        print(f"→ 调参集上最好：比例 {best_ratio:.2f}"
              f"（{args.metric} = {best_metrics[args.metric]:.3f}）")

        # -------------------------------------------------------------- 留出集
        print("\n【留出集】只用一次。这一列才是「这个权重到底有没有用」的答案：")
        print(_header())
        print("-" * 56)

        baseline = score(holdout, rankings, {"vector": 1.0, "bm25": 1.0}, args.top_k)
        chosen = score(holdout, rankings, {"vector": 1.0, "bm25": best_ratio}, args.top_k)
        print(_fmt_row("等权基线", baseline))
        print(_fmt_row(f"比例 {best_ratio:.2f}", chosen))
        print("-" * 56)

        delta = chosen["recall"] - baseline["recall"]
        noise = wilson_half_width(baseline["recall"], len(holdout))
        print(f"差异：{delta:+.1%}"
              f"（n={len(holdout)} 时 95% 置信区间约 ±{noise:.1%}——"
              f"小于这个幅度的差异和 0 没有统计区别）")

        print()
        if delta > noise:
            print("→ 留出集上有超过噪声的收益，可以考虑启用：")
            print(f"   在 .env 里加 RRF_WEIGHT_VECTOR=1.0、RRF_WEIGHT_BM25={best_ratio:.2f}")
        elif delta > 0:
            print("→ 方向是对的，但幅度落在噪声里，**没有统计证据**说明它更好。")
            print("   保持等权（默认），除非以后有更大的评测集能把差异量出来。")
        else:
            print("→ 留出集上**没有收益**（甚至更差）。这就说明调参集上挑出来的那组权重")
            print("   是在拟合那一半数据的噪声，不是真的更懂检索。**保持等权**，不要启用。")
            print("   这正是「为什么默认不加权」那条结论的来源——可以自己复现。")
        print()
        print("  注：置信区间只考虑了单侧抽样波动，没算「在 8 个候选里挑最好的」"
              "这个动作本身带来的多重比较偏差。真实的显著门槛比上面这个数还高。")
    finally:
        store.close()


if __name__ == "__main__":
    main()
