"""
会话操作台：看有哪些会话、看某个会话说了什么、删掉它。

    python -m ecom.orchestrator.cli.session list [--limit 50]
    python -m ecom.orchestrator.cli.session show <会话id>
    python -m ecom.orchestrator.cli.session delete <会话id>
    python -m ecom.orchestrator.cli.session purge [--days 30]

不带子命令时打印概览。

【为什么 purge 是主角】
这个文件里只有 `purge` 是"日常该跑"的。落盘会话的代价就是**它不会自己消失**——
数据留在磁盘上，而里面是用户说过的话。所以保留期必须有东西来执行它。

**什么时候跑**：`api.py` / `app.py` 启动时会自动调一次 `purge()`（同一个函数，
不是另写一份）。但长驻进程不会自我清理，所以**建议外挂一行 cron**：

    0 4 * * *  cd /path/to/ecom_cs && python -m ecom.orchestrator.cli.session purge

刻意不做"读的时候顺手删"那种懒清理——那会让一个只读操作产生写副作用，
出问题时（"我的会话怎么没了"）极难定位。
"""
import argparse
import sys

from ecom.orchestrator import session


def cmd_list(args) -> int:
    rows = session.list_sessions(limit=args.limit)
    if not rows:
        print("还没有任何会话。")
        return 0

    print(f"{'会话ID':<34} {'轮数':>4}  {'最后活动':<20} 创建于")
    print("-" * 84)
    for row in rows:
        # messages 是"一问一答两条"，所以轮数是它的一半
        turns = row["messages"] // 2
        print(f"{row['session_id']:<34} {turns:>4}  {row['updated_at']:<20} {row['created_at']}")
    print(f"\n共 {len(rows)} 个（最多显示 {args.limit} 个）")
    return 0


def cmd_show(args) -> int:
    meta = session.get_session(args.session_id)
    if not meta:
        print(f"没有这个会话：{args.session_id}", file=sys.stderr)
        return 1

    print(f"会话 {meta['session_id']}")
    print(f"创建 {meta['created_at']}　最后活动 {meta['updated_at']}")
    print("-" * 60)

    # 看全部，不走 HISTORY_TURNS 那个截断——那个参数管的是"喂给模型多少"，
    # 这里是给人看的，人要看全的。用 `transcript()` 而不是 `history()`：
    # 后者会把人工客服（role='agent'）映射成 assistant，人的回复就看不出来了。
    messages = session.transcript(args.session_id, max_messages=10**6)
    if not messages:
        print("（这个会话还没有对话）")
        return 0

    for msg in messages:
        if msg["role"] == "user":
            who = "用户"
        elif msg["role"] == "agent":
            who = f"人工客服 {msg['author']}" if msg.get("author") else "人工客服"
        else:
            who = "机器人"
        print(f"[{who}] {msg['content']}")
    return 0


def cmd_delete(args) -> int:
    removed = session.delete_session(args.session_id)
    if removed == 0:
        print(f"没有这个会话（或它本来就没有消息）：{args.session_id}", file=sys.stderr)
        return 1
    print(f"已删除会话 {args.session_id}（{removed} 条消息）")
    return 0


def cmd_purge(args) -> int:
    days = args.days
    result = session.purge(older_than_days=days)
    # 记忆也一起报：它是这段对话的摘要，**也是用户说过的话**。只报消息数的话，
    # "记忆有没有被删掉"这件事在输出里看不见——而它恰恰是最该被看见的那部分。
    print(
        f"已清理 {result['sessions']} 个会话 / {result['messages']} 条消息 / "
        f"{result['memories']} 条记忆 / {result.get('tickets', 0)} 张工单"
        f"（{days} 天没动静的）"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="会话操作台",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="列出会话")
    p_list.add_argument("--limit", type=int, default=50)
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="看某个会话的完整对话")
    p_show.add_argument("session_id")
    p_show.set_defaults(func=cmd_show)

    p_delete = sub.add_parser("delete", help="删掉一个会话（连同它的消息）")
    p_delete.add_argument("session_id")
    p_delete.set_defaults(func=cmd_delete)

    p_purge = sub.add_parser("purge", help="清掉太久没动静的会话")
    p_purge.add_argument(
        "--days",
        type=int,
        default=None,
        help=f"保留多少天（默认 {session.SESSION_TTL_DAYS}，即 SESSION_TTL_DAYS）",
    )
    p_purge.set_defaults(func=cmd_purge)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not getattr(args, "func", None):
        parser.print_help()
        info = session.stats()
        print(f"\n会话库：{info['db_path']}")
        print(
            f"共 {info['sessions']} 个会话 / {info['messages']} 条消息 / "
            f"{info['memories']} 条记忆 / {info.get('open_tickets', 0)} 张未关闭工单，"
            f"保留 {info['ttl_days']} 天"
        )
        if info["newest"]:
            print(f"最早 {info['oldest']}　最新 {info['newest']}")
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
