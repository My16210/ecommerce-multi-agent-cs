"""
退货审核的操作台。

    python -m ecom.orchestrator.cli.returns stats
    python -m ecom.orchestrator.cli.returns list [--status 待审核] [--limit 50]
    python -m ecom.orchestrator.cli.returns show <退货单号>
    python -m ecom.orchestrator.cli.returns approve <退货单号> [--note 备注] [--reviewer 张三]
    python -m ecom.orchestrator.cli.returns reject  <退货单号> [--note 备注] [--reviewer 张三]

不带子命令时打印概览。

【为什么要一个命令行入口，明明有后台界面】
后台界面（`admin.py`）是给人**经常**用的，但要能打开它得有浏览器、有起服务、
有密码。这个 CLI 是给三种场合用的：容器里排障、写脚本批量看、以及 CI 那种
没有界面的地方。两者调的是同一份 `admin.returns` 里的函数——不是两套逻辑。

【这个文件和 `ecom/knowledge/cli/review.py` 长得像，是有意的】
那是知识库的人审队列（模型判定 → 人批准才进向量库），这里是退货的人审。
两条流程的**形状**本来就是一回事：模型只能登记，结论由人下。
两边的措辞和命令名保持一致，读到第二个的人就不用重新学一遍。

【为什么 stdout 不去 reconfigure 成 UTF-8】
理由和 `knowledge/cli/review.py` 里那段一样：GBK 控制台下靠 Python 自己的
编码错误处理降级，比强行改口径更不容易在别人的终端上炸。那条注释写在那儿，
这里不重复。
"""
import argparse
import sys

from ecom.orchestrator.admin import returns
from ecom.orchestrator.tools.order_tools import (
    REJECTED_RETURN_STATUS,
    RETURN_STATUSES,
)


def _print_reviewed(row: dict) -> str:
    """把审核痕迹拼成一句。没审过就是一句"还没有人审"。"""
    if not row.get("reviewed_at"):
        return "还没有人审"
    who = row.get("reviewer") or "（未署名）"
    note = row.get("review_note") or "（没写备注）"
    return f"{row['reviewed_at']} {who}：{row['decision']}——{note}"


def cmd_stats(args) -> int:
    counts = returns.stats()
    print("退货申请：")
    for status in RETURN_STATUSES:
        print(f"  {status}　{counts.get(status, 0)}")
    return 0


def cmd_list(args) -> int:
    rows = returns.list_returns(status=args.status, limit=args.limit)
    if not rows:
        print("没有符合条件的退货申请。" if args.status else "还没有任何退货申请。")
        return 0

    print(f"{'退货单号':<14} {'订单号':<10} {'状态':<8} {'提交时间':<20} 审核")
    print("-" * 100)
    for row in rows:
        print(
            f"{row['return_id']:<14} {row['order_id']:<10} {row['status']:<8} "
            f"{row['created_at']:<20} {_print_reviewed(row)}"
        )
    print(f"\n共 {len(rows)} 条（最多显示 {args.limit} 条）")
    return 0


def cmd_show(args) -> int:
    row = returns.get_return(args.return_id)
    if not row:
        print(f"没有这个退货申请：{args.return_id}", file=sys.stderr)
        return 1

    print(f"退货单号　{row['return_id']}")
    print(f"订单号　　{row['order_id']}")
    print(f"状态　　　{row['status']}")
    print(f"提交时间　{row['created_at']}")
    print(f"申请理由　{row['reason']}")
    print(f"审核　　　{_print_reviewed(row)}")
    return 0


def _review(args, decision: str) -> int:
    result = returns.review_return(
        args.return_id, decision, note=args.note, reviewer=args.reviewer
    )
    if not result["ok"]:
        print(result["message"], file=sys.stderr)
        return 1
    print(result["message"])
    # already 不是失败：它表示"你要的结论已经在里面了"。退出码仍然是 0，
    # 否则一个重复执行的安全脚本（幂等本来就是它的卖点）会被当成出错。
    return 0


def cmd_approve(args) -> int:
    return _review(args, returns.APPROVED_RETURN_STATUS)


def cmd_reject(args) -> int:
    return _review(args, REJECTED_RETURN_STATUS)


def _add_review_args(parser: argparse.ArgumentParser, verb: str) -> None:
    parser.add_argument("return_id", help="退货单号")
    parser.add_argument("--note", default=None, help="审核备注（会落库，人看得到）")
    parser.add_argument("--reviewer", default=None, help="审核人（不填就是未署名）")
    parser.set_defaults(func=cmd_approve if verb == "approve" else cmd_reject)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="退货审核操作台",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("stats", help="各状态各有多少条").set_defaults(func=cmd_stats)

    p_list = sub.add_parser("list", help="退货申请列表")
    p_list.add_argument(
        "--status",
        default=None,
        choices=list(RETURN_STATUSES),
        help="只看某个状态（不填 = 全部）",
    )
    p_list.add_argument("--limit", type=int, default=50)
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="看一条申请的详情（含审核痕迹）")
    p_show.add_argument("return_id")
    p_show.set_defaults(func=cmd_show)

    _add_review_args(sub.add_parser("approve", help="通过"), "approve")
    _add_review_args(sub.add_parser("reject", help="拒绝"), "reject")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not getattr(args, "func", None):
        parser.print_help()
        print()
        return cmd_stats(args)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
