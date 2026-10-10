"""
金标准评测集：生成待标注文件 + 人工逐条标注 + 出报告

**这个脚本自己不给任何标准答案。** 它只做三件事：分层抽样出题、把检索到的
候选摆到人面前、统计人确认过的那些。标注必须由人来做——模型生成标注再叫它
"金标准"，就是拿一份合成集冒充真值，评测分数越高越自欺。

与 data/eval_set.jsonl 的关系
----------------------------
那份 290 题是 `build_eval_set.py` 让模型对着每个 chunk 出的题，**题和答案同源**，
所以它的分数只能回答"检索能不能找回出题时看的那一段"，不能回答"这题出得对不对"。
这份金标准是在同一批语料上、由人挑出来的子集，用来给合成集的分数量一个**偏差**：
两者接近说明合成集可信，差得远说明合成集在报喜。

三个子命令
----------
    uv run python gold_set.py build --n 50     # 抽样 + 检索候选 → data/gold_set.jsonl
    uv run python gold_set.py label            # 逐条人审（默认只标 pending 的）
    uv run python gold_set.py report           # 只统计人确认过的条目

标注时人看到的是**盲的**（见下面两条设计），这两条是这个工具的全部价值所在。

设计一：不把合成答案抄进待标注文件
----------------------------------
出题时那份"期望段落"**不写进 gold_set.jsonl**。否则标注的人一眼看到机器给的名字，
点个头就完事——那不是标注，是复核，而且复核的是同一个模型的判断。
不给，人就必须自己读问题、自己看候选，这才有独立性。

设计二：候选给得比报告用的 Top-K 宽
----------------------------------
`build --candidates 10` 默认摆 10 条，而 `report --top-k 5` 按 5 条算分。
为什么不摆 5 条：如果只给人看 Top-5，人就只能从这 5 条里挑，那 Recall@5 必然
100%——这个数一点信息量都没有。摆得更宽，正确那条才可能落在 6-10 位，于是
"Top-5 里到底有没有"变成一个有真答案的问题。

于是标注有三种结果，第三种是**必须存在**的：

    1. 选候选中某一条         → 正确那条被召回了，记名次
    2. 选"都不对"（none）     → 前 N 条里根本没有正确的那条 → 记为**未命中**
    3. 丢弃                   → 这题本身有问题（题干不清、答案分散在多段）→ 不进统计

第 2 种若不允许，人被迫在错的中挑一个"最像对"的，报告会虚高。所以它是一等公民。
"""
import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from ecom.knowledge.cli.tune_rrf import wilson_half_width
from ecom.knowledge.retriever.retriever import retrieve
from ecom.knowledge.settings import GOLD_SET_PATH
from ecom.knowledge.utils.eval import evaluate, load_eval_set
from ecom.knowledge.vector_store import get_vector_store

# 抽样种子固定：同一个评测集每次抽出来的是同一批题。
# 换一批题要显式改这里——否则"上次 62%、这次 58%"就分不清是检索变了还是题变了。
SEED = 20260101

# 摆给人看的候选条数（比报告用的 top-k 宽，理由见文件头"设计二"）
DEFAULT_CANDIDATES = 10
CANDIDATE_TEXT_LEN = 120


# ---------------------------------------------------------------------------
# 抽样
# ---------------------------------------------------------------------------


def stratified_sample(cases: list, n: int, seed: int = SEED) -> list:
    """
    按题型分层抽样，保证改写型 / 术语型都够数。

    为什么不直接 random.sample：290 题里两类题的比例是固定的，整体随机抽 50 题，
    万一抽出来 45 题是术语型，那"改写型检索好不好"这个结论就没有样本支撑了。
    分层是先按题型分桶、再把名额摊到每个桶，**桶内**才随机。

    抽样前按 (type, id) 排序，所以**输入的条目顺序不影响结果**——
    评测集重排一次不会悄悄换掉压测的那 50 道题。
    """
    cases = list(cases)
    if n >= len(cases):
        return sorted(cases, key=lambda c: (c.get("type", ""), c["id"]))

    buckets: dict = {}
    for case in sorted(cases, key=lambda c: (c.get("type", ""), c["id"])):
        buckets.setdefault(case.get("type", "全部"), []).append(case)

    rng = random.Random(seed)
    names = sorted(buckets)
    quota = n // len(names)
    picked: list = []

    for index, name in enumerate(names):
        # 余数摊给前几个桶，保证总数正好是 n（而不是 n 向下取整后被丢掉几题）
        take = quota + (1 if index < n % len(names) else 0)
        picked.extend(rng.sample(buckets[name], min(take, len(buckets[name]))))

    # 上面让每个桶都可能超发（当某个桶题不够时），这里补足到 n
    picked_ids = {case["id"] for case in picked}
    leftovers = [case for case in cases if case["id"] not in picked_ids]
    if len(picked) < n and leftovers:
        picked.extend(rng.sample(leftovers, min(n - len(picked), len(leftovers))))

    return sorted(picked, key=lambda c: (c.get("type", ""), c["id"]))


# ---------------------------------------------------------------------------
# 读写
# ---------------------------------------------------------------------------


def gold_path(path: str = None) -> str:
    """把 None 解析成默认路径。命令行不传 --gold-path 时到处都要用这个"""
    return path or GOLD_SET_PATH


def load_gold(path: str = None) -> list:
    """读金标准文件。不存在是正常状态（还没 build 过），返回空表"""
    path = gold_path(path)
    try:
        with open(path, encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]
    except FileNotFoundError:
        return []


def save_gold(records: list, path: str = None) -> None:
    path = gold_path(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def confirmed_cases(records: list) -> tuple:
    """
    把记录分成三堆：人确认过的、还没标的、被丢弃的。

    单独抽出来是为了让"未标注的绝不进报告"这件事**可被测试**——
    它掉进报告里，金标准就退化成合成集，而且看不出来。
    """
    confirmed = [r for r in records if r.get("status") == "confirmed"]
    pending = [r for r in records if r.get("status") == "pending"]
    discarded = [r for r in records if r.get("status") == "discarded"]
    return confirmed, pending, discarded


def _labeler() -> str:
    """标注人是谁：优先环境变量，其次 git 配置——至少要能追到人"""
    return os.getenv("GOLD_LABELER") or _git_user() or "unknown"


def _git_user() -> str:
    try:
        result = subprocess.run(
            ["git", "config", "user.name"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001 —— 拿不到就不署，不该让标注流程因此中断
        return ""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def build(args) -> int:
    cases = load_eval_set(args.set_path)
    if not cases:
        print(f"评测集是空的：{args.set_path or 'data/eval_set.jsonl'}。"
              "请先跑 uv run python build_eval_set.py")
        return 1

    sample = stratified_sample(cases, args.n)
    types = {}
    for case in sample:
        types[case.get("type", "全部")] = types.get(case.get("type", "全部"), 0) + 1
    print(f"从 {len(cases)} 题里分层抽了 {len(sample)} 题：{types}")

    store = get_vector_store()
    records = []
    try:
        if store.count() == 0:
            print("向量库是空的，请先运行：uv run python ingest.py")
            return 1
        for index, case in enumerate(sample, 1):
            hits = retrieve(case["question"], store, top_k=args.candidates)
            records.append({
                "id": f"gold-{index:04d}",
                "question": case["question"],
                "type": case.get("type"),
                # 候选**存下来**：label 时人要对着当初摆出来的这几条点头，
                # 不能等标注时重新检索一遍——那可能换了配置，摆出来的就不是同一批了。
                # 但 report 仍然**现算检索**：它要回答的是"按当前这套检索配置，
                # 人认定的那条在不在 Top-K 里"，所以读的是 live 结果，不是这份快照。
                "candidates": [
                    {
                        "source": hit["source"],
                        "chunk_id": hit["chunk_id"],
                        "text": hit["text"][:CANDIDATE_TEXT_LEN],
                    }
                    for hit in hits
                ],
                # 刻意不写 expected_source / expected_chunk_id：见文件头"设计一"
                "status": "pending",
                "expected_source": None,
                "expected_chunk_id": None,
                "labeled_by": None,
                "labeled_at": None,
            })
            print(f"  [{index}/{len(sample)}] {case['question'][:40]}… "
                  f"（{len(hits)} 条候选）")
    finally:
        store.close()

    save_gold(records, args.gold_path)
    print(f"\n已写出 {len(records)} 条待标注 → {gold_path(args.gold_path)}")
    print("下一步：uv run python gold_set.py label")
    return 0


# ---------------------------------------------------------------------------
# label
# ---------------------------------------------------------------------------


def _prompt(record: dict, index: int, total: int) -> str:
    """把一条记录摆给人看，返回人的选择"""
    print("\n" + "=" * 66)
    print(f"[{index}/{total}] {record['question']}")
    print(f"题型：{record.get('type')}")
    print("-" * 66)
    for position, candidate in enumerate(record["candidates"], 1):
        head = f"{candidate['source']}#{candidate['chunk_id']}"
        print(f"  {position:>2}. {head:<28} {candidate['text']}")
    print("-" * 66)
    print("  选段号 = 正确那条就是它   n = 这些都不对   "
          "s = 跳过（保持待标注）   d = 丢弃这题   q = 存盘退出")
    return input("> ").strip().lower()


def label(args) -> int:
    records = load_gold(args.gold_path)
    if not records:
        print(f"没有待标注文件：{gold_path(args.gold_path)}。请先跑 gold_set.py build")
        return 1

    if args.only_pending:
        todo = [r for r in records if r.get("status") == "pending"]
    else:
        todo = [r for r in records if r.get("status") != "discarded"]
    if not todo:
        print("没有需要标注的条目。")
        return 0

    who = _labeler()
    print(f"待标注 {len(todo)} 条，标注人：{who}（来源可改：GOLD_LABELER 环境变量）")

    by_id = {record["id"]: record for record in records}
    for index, record in enumerate(todo, 1):
        choice = _prompt(record, index, len(todo))
        target = by_id[record["id"]]

        if choice in ("q", "quit"):
            break
        if choice in ("s", "skip", ""):
            continue
        if choice in ("d", "discard"):
            target.update(status="discarded", labeled_by=who, labeled_at=_now())
        elif choice in ("n", "none"):
            # 前 N 条里没有正确的那条。**记为 confirmed 但无期望段**——
            # report 会把它算作未命中，见文件头"设计二"。
            target.update(status="confirmed", expected_source=None,
                          expected_chunk_id=None, labeled_by=who, labeled_at=_now())
        elif choice.isdigit() and 1 <= int(choice) <= len(target["candidates"]):
            picked = target["candidates"][int(choice) - 1]
            target.update(status="confirmed", expected_source=picked["source"],
                          expected_chunk_id=picked["chunk_id"],
                          labeled_by=who, labeled_at=_now())
        else:
            print("不认识这个输入，这条留作待标注。")
            continue

        # 每标一条就落盘：标注是个慢活，中途关掉不该丢掉前面几十条
        save_gold(records, args.gold_path)

    save_gold(records, args.gold_path)
    confirmed, pending, _ = confirmed_cases(records)
    print(f"\n已存盘：确认 {len(confirmed)} 条，待标注 {len(pending)} 条")
    print(f"下一步：uv run python gold_set.py report --top-k {args.top_k}")
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def report(args) -> int:
    records = load_gold(args.gold_path)
    if not records:
        print(f"没有金标准文件：{gold_path(args.gold_path)}。请先 build 再 label")
        return 1

    confirmed, pending, discarded = confirmed_cases(records)
    labeled = [r for r in confirmed if r.get("expected_source")]
    none_of_above = [r for r in confirmed if not r.get("expected_source")]

    # 这几行必须先印：报告里若混进未标注的条目，金标准就退化成合成集了
    print("=" * 66)
    print(f"金标准集：{gold_path(args.gold_path)}")
    print(f"  已确认 {len(confirmed)} 条  |  待标注 {len(pending)} 条  |  "
          f"已丢弃 {len(discarded)} 条")
    if pending:
        print(f"  ※ 待标注的 {len(pending)} 条**不计入**下面的分数"
              f"（它们还没有人工判断，算进去就是让模型自证）")
    if none_of_above:
        print(f"  ※ 其中 {len(none_of_above)} 条标注为「前 N 条里都不对」"
              f"——按未命中计")
    print("=" * 66)

    if not confirmed:
        print("还没有确认过的条目，无法出分。先跑 gold_set.py label")
        return 1

    total = len(confirmed)
    if not labeled:
        print(f"\nRecall@{args.top_k} = 0/{total} = 0.0%（全部标注为「都不对」）")
        return 0

    cases = [
        {
            "question": r["question"],
            "expected_source": r["expected_source"],
            "expected_chunk_id": r.get("expected_chunk_id"),
            "type": r.get("type"),
        }
        for r in labeled
    ]
    verdict = evaluate(top_k=args.top_k, cases=cases, verbose=args.verbose,
                       title="金标准（人工标注）")

    # evaluate 只看到 labeled 那一部分，而"都不对"的必须算进分母——
    # 否则漏掉的那些会被无声地排除，分数虚高。
    hits = verdict["hits"]
    recall = hits / total
    mrr = verdict["mrr"] * len(labeled) / total
    half = wilson_half_width(recall, total)

    print("-" * 56)
    print(f"合计（含「都不对」的 {len(none_of_above)} 条）：")
    print(f"Recall@{args.top_k} = {hits}/{total} = {recall:.1%}  "
          f"±{half:.1%}（95% 置信区间半宽）")
    print(f"MRR        = {mrr:.3f}")
    print(f"\n置信区间比分数本身重要：{total} 条样本下 ±{half:.1%} 以内的差异"
          f"都可能是抽样噪声，别拿它去调参数。")
    return 0


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="金标准评测集工具（人工标注）")
    sub = parser.add_subparsers(dest="command")

    p_build = sub.add_parser("build", help="分层抽样 + 检索候选，生成待标注文件")
    p_build.add_argument("--n", type=int, default=50, help="抽多少条（默认 50）")
    p_build.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES,
                         help=f"每条摆出多少候选（默认 {DEFAULT_CANDIDATES}，"
                              "要比 report 的 --top-k 宽）")
    p_build.add_argument("--set-path", default=None, help="源评测集，默认 data/eval_set.jsonl")
    p_build.add_argument("--gold-path", default=None, help="输出文件，默认 data/gold_set.jsonl")

    p_label = sub.add_parser("label", help="逐条人工标注")
    p_label.add_argument("--only-pending", action="store_true",
                         help="只标 pending 的，不去动已确认的条目")
    p_label.add_argument("--gold-path", default=None)
    p_label.add_argument("--top-k", type=int, default=5, help="只为提示下一步命令用")

    p_report = sub.add_parser("report", help="统计人确认过的条目")
    p_report.add_argument("--top-k", type=int, default=5)
    p_report.add_argument("--gold-path", default=None)
    p_report.add_argument("--verbose", action="store_true", help="逐条打印，含命中的")

    args = parser.parse_args()
    if args.command == "build":
        return build(args)
    if args.command == "label":
        return label(args)
    if args.command == "report":
        return report(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
