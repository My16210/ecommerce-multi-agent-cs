"""
记忆操作台：把滑出窗口的对话蒸馏成摘要、看沉淀了哪些、试某句话会召回哪几条。

    python -m ecom.orchestrator.cli.memory distill --session <会话id>
    python -m ecom.orchestrator.cli.memory distill --all
    python -m ecom.orchestrator.cli.memory list    --session <会话id>
    python -m ecom.orchestrator.cli.memory recall  --session <会话id> "问题"

**为什么不并进 `cli/session.py`**：那个是"看会话 / 删会话"，这个是"蒸馏 / 召回记忆"。
两件事的触发者不同（一个是运维，一个是成本控制），混在一个文件里会让 `--help`
变成一锅粥。

【为什么蒸馏必须在这里、而不是问答路径上顺手做】
放问答路径上意味着**用户每问一句都可能多一次模型调用**，而且是不可控的那种
（第一次问就触发全量蒸馏）。收成一条显式命令之后：模型调用只在有人敲它的时候发生，
可以挂 cron，也可以在成本紧张时**不跑**。代价说清：**记忆不是实时的**——要有人跑
`distill` 才更新。这正是"不拖慢问答"换来的，和 `purge` 要外挂 cron 是同一个立场。

【`recall` 这条子命令是必须有的】
召回是个黑盒：`MEMORY_TOP_K` 挑出来的是哪几条、余弦各是多少，光看代码看不出来。
没有它，"线上为什么没带记忆"只能靠猜。这是这个仓库「宁可暴露」的一贯做法
（`router_source` 进 trace、`synth_source` 进 state 都是同一个理由）。
它是 **dry-run**：只读、不写、不改任何状态。
"""
import argparse
import sys

from ecom.orchestrator import memory, session
from ecom.orchestrator.utils.console import safe as _safe


def _distill_one(sid: str) -> int:
    """蒸馏一个会话。返回 0 = 正常（含"无可蒸馏"），1 = 出错了。"""
    try:
        result = memory.distill(sid)
    except Exception as exc:  # noqa: BLE001 —— 网络/embedding 失败要如实报，不能假装成功
        print(_safe(f"  {sid}：蒸馏失败（{type(exc).__name__}: {exc}）"), file=sys.stderr)
        print("  ——水位线没有推进，修好之后可以重跑这一条。", file=sys.stderr)
        return 1

    if result["created"]:
        print(_safe(f"  {sid}：新增 1 条记忆（覆盖 {result['messages']} 条消息）"))
        return 0
    if result["reason"] == "无可蒸馏":
        # **这不是错误**：会话还短，或者全都已经蒸馏过了。
        print(_safe(f"  {sid}：无可蒸馏"))
        return 0
    print(_safe(f"  {sid}：没写（{result['reason']}）"))
    return 0


def cmd_distill(args) -> int:
    if args.all:
        rows = session.list_sessions(limit=10**6)
        if not rows:
            print("还没有任何会话。")
            return 0
        print(f"共 {len(rows)} 个会话，逐个蒸馏：")
        return max((_distill_one(row["session_id"]) for row in rows), default=0)

    if not args.session:
        print("要 --session <会话id> 或 --all。", file=sys.stderr)
        return 2
    if session.get_session(args.session) is None:
        print(f"没有这个会话：{args.session}", file=sys.stderr)
        return 1
    return _distill_one(args.session)


def cmd_list(args) -> int:
    if session.get_session(args.session) is None:
        print(f"没有这个会话：{args.session}", file=sys.stderr)
        return 1

    rows = session.memories_for(args.session)
    if not rows:
        print("这个会话还没有记忆。")
        print("（滑出窗口的消息要跑 `distill` 才会被蒸馏成记忆——记忆不是实时的。）")
        return 0

    print(f"会话 {args.session} 共 {len(rows)} 条记忆：")
    for row in rows:
        print("-" * 70)
        print(f"[{row['id']}] 覆盖到消息 #{row['through_message_id']}　{row['created_at']}")
        # 摘要和索引词**都是模型产出的文本**——那是最容易夹带 GBK 打不出的字符的地方。
        print(_safe(f"     {row['summary']}"))
        if row["keywords"]:
            print(_safe(f"     索引词：{row['keywords']}"))
    return 0


def cmd_recall(args) -> int:
    """**dry-run**：这句话会召回哪几条记忆、余弦各是多少。只读。"""
    if session.get_session(args.session) is None:
        print(f"没有这个会话：{args.session}", file=sys.stderr)
        return 1

    pool = session.memories_for(args.session)
    if not pool:
        print("这个会话还没有记忆——先跑 distill，或者它本来就还没长出可蒸馏的部分。")
        return 0

    try:
        hits = memory.recall(args.session, args.question)
    except Exception as exc:  # noqa: BLE001
        print(_safe(f"召回失败（{type(exc).__name__}: {exc}）"), file=sys.stderr)
        return 1

    print(_safe(f"问题：{args.question}"))
    print(f"池子里 {len(pool)} 条，带回来 {len(hits)} 条（MEMORY_TOP_K={memory.MEMORY_TOP_K}）：")
    for hit in hits:
        print("-" * 70)
        print(f"  余弦 {hit['score']:.4f}　[记忆 #{hit['id']}]")
        # 摘要来自模型输出，见 cmd_list 那一条。
        print(_safe(f"  {hit['summary']}"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="会话记忆操作台",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    p_distill = sub.add_parser("distill", help="把滑出窗口的对话蒸馏成记忆（会调模型）")
    p_distill.add_argument("--session", help="只处理这一个会话")
    p_distill.add_argument("--all", action="store_true", help="处理所有会话")
    p_distill.set_defaults(func=cmd_distill)

    p_list = sub.add_parser("list", help="看某个会话沉淀了哪些记忆")
    p_list.add_argument("--session", required=True)
    p_list.set_defaults(func=cmd_list)

    p_recall = sub.add_parser("recall", help="试一句话会召回哪几条（dry-run，只读）")
    p_recall.add_argument("--session", required=True)
    p_recall.add_argument("question", help="要试的问题（加引号）")
    p_recall.set_defaults(func=cmd_recall)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not getattr(args, "func", None):
        parser.print_help()
        info = session.stats()
        print(f"\n记忆库：{info['db_path']}")
        print(
            f"共 {info['sessions']} 个会话 / {info['messages']} 条消息 / "
            f"{info['memories']} 条记忆"
        )
        print("记忆只在跑 `distill` 时才增长——它**不是实时的**。")
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
