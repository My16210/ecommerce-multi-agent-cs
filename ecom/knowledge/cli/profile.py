"""
性能画像：延迟分解 + 成本统计
==============================

回答两个问题：**时间花在哪、钱花在哪。**

为什么值得单独做，而不是"感觉还挺快"
------------------------------------
"感觉快"拿去做不了取舍。只有当你知道 184 个 chunk 的 BM25 检索是微秒级、
而一次 embedding 的 HTTPS 往返是几十毫秒，你才知道优化该往哪使——
以及**哪一块现在根本不该优化**。这就是画像的全部意义。

两条路的成本完全不同，所以分开量
--------------------------------
检索那条路（embedding 和 rerank 都走硅基流动的 BGE 系列）**是免费档**，
本地那几步（向量检索 / BM25 / RRF）既不花钱也不走网络。所以"优化检索成本"
在这里是个伪命题——检索的路可以随便慢，只要不慢在别的维度上。
真正花钱的只有对话模型的 token，而那部分按次计费、和检索质量没关系。

用法
----
    uv run python profile.py                     # 检索链路（不需要任何 key）
    uv run python profile.py --set --limit 60    # 用合成评测集的题，样本多才有分位数
    uv run python profile.py --chain 5           # 再跑 5 题完整链路（需要 QWEN key）

计时挂在生产代码的执行点上（ecom/knowledge/utils/timing.py），**不是在脚本里重拼一条链路**：
重拼的那条只要有一处和生产不一样（rerank 挪了位置、某一步加了缓存），
画像就开始骗人，而且照常能跑出好看的数。所以计时器跟着真正的函数走。
"""
import argparse

from ecom.knowledge.retriever.retriever import retrieve
from ecom.knowledge.settings import (
    QWEN_API_KEY,
    QWEN_PRICE_INPUT,
    QWEN_PRICE_OUTPUT,
    TOP_K,
)
from ecom.knowledge.utils.eval import TEST_CASES, load_eval_set
from ecom.knowledge.utils.timing import collect
from ecom.knowledge.vector_store import get_vector_store

# 阶段名 → 中文标签。顺序就是打印顺序，按"在线链路的先后"排。
STAGE_LABELS = [
    ("embed", "向量化（API 往返）"),
    ("vector_search", "向量检索（本地）"),
    ("bm25", "BM25 检索（本地）"),
    ("fusion", "RRF 融合（纯计算）"),
    ("rerank", "Rerank（API 往返）"),
    ("web_search", "联网搜索（网络）"),
    ("generate", "答案生成（API 往返）"),
    ("judge", "提问判定（API 往返）"),
]

# 检索那几步（走 retrieve 函数）和完整链路是两个量法，标签分开列
RETRIEVAL_STAGES = ["embed", "vector_search", "bm25", "fusion"]


# ---------------------------------------------------------------------------
# 小工具：分位数、按显示宽度对齐
# ---------------------------------------------------------------------------


def percentile(values: list, q: float) -> float:
    """线性插值分位数。样本少的时候 p95 会很接近最大值，这是正常的，不是 bug"""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _width(text: str) -> int:
    """按终端显示宽度算：中日韩全角字符占 2 格，否则中文表头会全部错位"""
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _width(text))


def _ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f}"


# ---------------------------------------------------------------------------
# 一、检索链路（离线，不需要 key）
# ---------------------------------------------------------------------------


def _collect_one(question: str, store, top_k: int) -> dict:
    """跑一次检索并采集各阶段耗时"""
    with collect() as collector:
        retrieve(question, store, top_k=top_k)
    return collector.snapshot()


def profile_retrieval(questions: list, store, top_k: int = TOP_K) -> dict:
    """
    逐题跑检索，返回每个阶段的耗时分布。

    先"热身"一次再把那次丢掉：首问会顺带做三件一次性的事——建 BM25 索引、
    初始化 embedding 客户端、第一次 HTTPS 握手。把它们的代价摊进平均延迟里，
    会让"平均每问 40ms"变成一个没有意义的数。所以首问单独报一行，
    因为它恰恰是用户**重启后第一次提问**感受到的那个延迟。
    """
    if not questions:
        return {}

    print("热身并测量首问（含 BM25 建索引 + 客户端初始化）…")
    cold = _collect_one(questions[0], store, top_k)
    cold_total = sum(cold["durations"].values())

    warm_from = questions[1:] or questions
    print(f"逐题采集 {len(warm_from)} 个样本…")
    snapshots = [_collect_one(question, store, top_k) for question in warm_from]

    return {"cold": cold, "cold_total": cold_total, "snapshots": snapshots}


def _print_retrieval(result: dict) -> None:
    snapshots = result["snapshots"]
    stages = [name for name, _ in STAGE_LABELS if name in RETRIEVAL_STAGES]

    if not result["cold"]["durations"]:
        print("没有采到任何阶段——检索可能直接抛异常了。")
        return

    print()
    print("=" * 68)
    print(f"检索链路延迟分解（{len(snapshots)} 个样本，单位毫秒）")
    print("=" * 68)
    header = _pad("阶段", 24) + _pad("平均", 10) + _pad("p50", 10) + _pad("p95", 10) + "占比"
    print(header)
    print("-" * 68)

    means = {}
    for name in stages:
        values = [s["durations"].get(name, 0.0) for s in snapshots]
        means[name] = sum(values) / len(values) if values else 0.0

    total_mean = sum(means.values()) or 1.0
    for name in stages:
        values = [s["durations"].get(name, 0.0) for s in snapshots]
        label = dict(STAGE_LABELS)[name]
        share = means[name] / total_mean
        print(
            _pad(label, 24)
            + _pad(_ms(means[name]), 10)
            + _pad(_ms(percentile(values, 0.5)), 10)
            + _pad(_ms(percentile(values, 0.95)), 10)
            + f"{share:>5.1%}"
        )

    print("-" * 68)
    print(_pad("合计", 24) + _pad(_ms(total_mean), 10))

    cold = result["cold"]["durations"]
    print()
    print(f"冷启动（首问，仅此一次）：{_ms(result['cold_total'])} ms，其中 "
          + "、".join(f"{dict(STAGE_LABELS)[n]} {_ms(cold[n])}" for n in stages if n in cold))
    print("  ↑ 这个数才是「重启后第一次提问」的真实体感，之后每一问都走上面那行平均。")


# ---------------------------------------------------------------------------
# 二、完整链路（需要 key）
# ---------------------------------------------------------------------------


def _print_chain(snapshots: list, web_ok: int = None) -> None:
    stages = [name for name, _ in STAGE_LABELS if any(name in s["counts"] for s in snapshots)]
    if not stages:
        return

    means = {}
    for name in stages:
        values = [s["durations"].get(name, 0.0) for s in snapshots]
        means[name] = sum(values) / len(values)

    print()
    print("=" * 68)
    print(f"完整链路（{len(snapshots)} 个样本，单位毫秒）")
    print("=" * 68)
    header = _pad("阶段", 24) + _pad("平均", 10) + _pad("p50", 10) + _pad("p95", 10) + "占比"
    print(header)
    print("-" * 68)

    serial_total = sum(means.values()) or 1.0
    for name in stages:
        values = [s["durations"].get(name, 0.0) for s in snapshots]
        print(
            _pad(dict(STAGE_LABELS)[name], 24)
            + _pad(_ms(means[name]), 10)
            + _pad(_ms(percentile(values, 0.5)), 10)
            + _pad(_ms(percentile(values, 0.95)), 10)
            + f"{means[name] / serial_total:>5.1%}"
        )
    print("-" * 68)
    print(_pad("各阶段相加", 24) + _pad(_ms(serial_total), 10))

    # 关键路径：图里 retrieve_node 和 web_search_node 是**并行**的，
    # 所以端到端不是所有阶段相加，而是 max(检索那一路, 联网) + 生成 + 判定。
    # 把两个数都印出来，正是为了让"并行到底省了多少"看得见。
    retrieval_path = sum(means.get(n, 0.0) for n in RETRIEVAL_STAGES + ["rerank"])
    parallel = max(retrieval_path, means.get("web_search", 0.0))
    critical = parallel + means.get("generate", 0.0) + means.get("judge", 0.0)

    print()
    print(f"关键路径估算：max(检索 {_ms(retrieval_path)}ms, 联网 {_ms(means.get('web_search', 0.0))}ms)"
          f" → 生成 {_ms(means.get('generate', 0.0))}ms → 判定 {_ms(means.get('judge', 0.0))}ms")
    print(f"            ≈ {_ms(critical)} ms（用户从回车到看到答案约等这么久，判定在答案之后）")
    print(f"  并行省下：{_ms(serial_total - critical)} ms（= 两路里较短的那一路）")

    # 联网这一项必须自带免责：ddgs 是免 key 的公共接口，被限流是常态，
    # 失败时它会在多个引擎之间逐个重试，这段重试时间照样计进 web_search 阶段。
    # 不标出来的话，"联网 5.6 秒"会被当成"搜一次要 5.6 秒"，而真相是"搜不到时
    # 重试要 5.6 秒"。这两个结论指向完全不同的优化方向。
    if web_ok is not None and web_ok < len(snapshots):
        print(f"  ※ 注：{len(snapshots) - web_ok}/{len(snapshots)} 题的联网搜索返回空"
              "（无结果或失败）。失败会在引擎间逐个重试，这段重试时间也算在"
              "「联网搜索」里——**这一项不能当作「一次成功搜索要多久」**。")

    _print_cost(snapshots)


def _print_cost(snapshots: list) -> None:
    count = len(snapshots)

    def total(key):
        return sum(s["metrics"].get(key, 0) for s in snapshots)

    gen_in, gen_out = total("llm_input_tokens"), total("llm_output_tokens")
    judge_in, judge_out = total("judge_input_tokens"), total("judge_output_tokens")
    embed_tokens = total("embed_tokens")

    print()
    print("=" * 68)
    print("成本（每题平均）")
    print("=" * 68)
    print("  检索（embedding + rerank）  ￥0.0000/题")
    print("    —— 硅基流动的 BGE 系列属免费档；本地那几步不花钱也不走网络。")
    print("       所以「优化检索成本」是个伪命题，真正花钱的只有下面这两行。")

    if gen_in or gen_out:
        gen_cost = (gen_in / 1e6) * QWEN_PRICE_INPUT + (gen_out / 1e6) * QWEN_PRICE_OUTPUT
        print(f"  答案生成  {gen_in / count:>7.0f} in + {gen_out / count:>6.0f} out tok"
              f"   ￥{gen_cost / count:.5f}/题")
    else:
        print("  答案生成  （未采到 token 用量——这轮可能是走的兜底文案，或模型没返回 usage）")

    if judge_in or judge_out:
        judge_cost = (judge_in / 1e6) * QWEN_PRICE_INPUT + (judge_out / 1e6) * QWEN_PRICE_OUTPUT
        print(f"  提问判定  {judge_in / count:>7.0f} in + {judge_out / count:>6.0f} out tok"
              f"   ￥{judge_cost / count:.5f}/题")
    else:
        print("  提问判定  （未采到 token 用量）")

    if gen_out and judge_out:
        per_q = ((gen_in + judge_in) / 1e6) * QWEN_PRICE_INPUT \
            + ((gen_out + judge_out) / 1e6) * QWEN_PRICE_OUTPUT
        print(f"  {'':<22}每千题约 ￥{per_q / count * 1000:.2f}")
    if embed_tokens:
        print(f"  （embedding 另用 {embed_tokens / count:.0f} tok/题，免费档）")
    print()
    print(f"  单价按 qwen-plus 输入 ￥{QWEN_PRICE_INPUT}/M、输出 ￥{QWEN_PRICE_OUTPUT}/M 计"
          "（settings 里可改，官方调价后会过期）")


def profile_chain(questions: list, store, chain_n: int, top_k: int = TOP_K) -> list:
    """跑完整链路：检索 ‖ 联网 → 生成，之后单独跑一次判定。需要 key"""
    from ecom.knowledge.core.graph import build_graph
    from ecom.knowledge.review.judge import judge
    from ecom.knowledge.settings import ENABLE_QUESTION_REVIEW

    if not QWEN_API_KEY:
        print("\n没配 QWEN_API_KEY，跳过完整链路。只跑离线那部分：")
        print("  uv run python profile.py --set")
        return []

    app = build_graph(store)
    picked = questions[:chain_n]

    # 先把 BM25 索引和客户端热起来，否则第一题会把一次性开销算进去
    retrieve(picked[0], store, top_k=top_k)

    print(f"\n跑完整链路 {len(picked)} 题（检索 ‖ 联网 → 生成"
          f"{' → 判定' if ENABLE_QUESTION_REVIEW else ''}）…")
    snapshots = []
    web_ok = 0
    for index, question in enumerate(picked, 1):
        # 每题一个独立 thread_id：否则历史会越堆越长，后面几题光 prompt 就比前面大，
        # 测出来的生成耗时是"历史长度"的函数，不是"问题难度"的函数。
        config = {"configurable": {"thread_id": f"profile-{index}"}}
        with collect() as collector:
            result = app.invoke({"question": question}, config=config)
            if ENABLE_QUESTION_REVIEW and result.get("answer"):
                # 调 judge() 而不是 record_question()：判定逻辑一样，
                # 但 record_question 会把结果**写进待审核队列**——画像不该污染业务数据。
                judge(question, result)
        snapshots.append(collector.snapshot())
        web_ok += bool(result.get("web_contexts"))

    _print_chain(snapshots, web_ok)
    return snapshots


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="性能画像：延迟分解 + 成本统计")
    parser.add_argument("--set", dest="use_set", action="store_true",
                        help="用合成评测集的题（默认用内置的 9 条）")
    parser.add_argument("--set-path", default=None, help="指定评测集文件")
    parser.add_argument("--limit", type=int, default=None, help="最多用几道题")
    parser.add_argument("--chain", type=int, default=0, metavar="N",
                        help="再跑 N 题完整链路（需要 QWEN_API_KEY）")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    if args.use_set or args.set_path:
        questions = load_eval_set(args.set_path)
        if not questions:
            print("没找到合成评测集。先运行：uv run python build_eval_set.py")
            return
        questions = [case["question"] for case in questions]
    else:
        questions = [case["question"] for case in TEST_CASES]

    if args.limit:
        questions = questions[: args.limit]

    store = get_vector_store()
    try:
        if store.count() == 0:
            print("向量库是空的，请先运行：uv run python ingest.py")
            return

        result = profile_retrieval(questions, store, top_k=args.top_k)
        _print_retrieval(result)

        if args.chain:
            profile_chain(questions, store, chain_n=args.chain, top_k=args.top_k)
    finally:
        store.close()


if __name__ == "__main__":
    main()
