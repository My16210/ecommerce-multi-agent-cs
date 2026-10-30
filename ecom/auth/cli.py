"""principal 的命令行管理。

    python -m ecom.auth.cli create --name alice --role admin [--ttl-days 30]
    python -m ecom.auth.cli set-password alice
    python -m ecom.auth.cli rotate alice [--ttl-days 30]
    python -m ecom.auth.cli list
    python -m ecom.auth.cli revoke alice
    python -m ecom.auth.cli purge-sessions
    python -m ecom.auth.cli audit --limit 20
"""
import argparse
import getpass
import json

from ecom.auth import store


def _print_json(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _print_token(created: dict) -> None:
    _print_json({k: v for k, v in created.items() if k != "token"})
    print("\nTOKEN（只显示这一次，服务端只存哈希）：")
    print(created["token"])


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ecom.auth.cli", description="身份管理")
    sub = parser.add_subparsers(dest="cmd", required=True)

    create = sub.add_parser("create", help="创建 principal，token 只显示一次")
    create.add_argument("--name", required=True)
    create.add_argument("--role", default="customer", choices=store.ROLES)
    create.add_argument("--ttl-days", type=float, default=0, help="token 有效期（天），0=不过期")

    password = sub.add_parser("set-password", help="设置/重置登录密码")
    password.add_argument("identifier")
    password.add_argument("--password", default=None, help="不给就交互输入（不回显）")

    rotate = sub.add_parser("rotate", help="轮换 API token，旧 token 和登录会话立即失效")
    rotate.add_argument("identifier")
    rotate.add_argument("--ttl-days", type=float, default=0)

    sub.add_parser("list", help="列出 principal（不含 token）")
    sub.add_parser("purge-sessions", help="清掉过期/已吊销的登录会话")

    revoke = sub.add_parser("revoke", help="按 id 或 name 吊销")
    revoke.add_argument("identifier")
    revoke.add_argument("--actor", default="cli")

    audit = sub.add_parser("audit", help="看最近的审计")
    audit.add_argument("--limit", type=int, default=20)

    args = parser.parse_args(argv)

    if args.cmd == "create":
        _print_token(store.create_principal(args.name, args.role, expires_days=args.ttl_days))
        return 0

    if args.cmd == "set-password":
        value = args.password
        if value is None:
            first = getpass.getpass("密码（至少 8 位）：")
            second = getpass.getpass("再输一次：")
            if first != second:
                print("两次输入不一致。")
                return 2
            value = first
        try:
            store.set_password(args.identifier, value)
        except ValueError as exc:
            print(exc)
            return 1
        print("密码已更新。")
        return 0

    if args.cmd == "rotate":
        try:
            rotated = store.rotate_principal(args.identifier, expires_days=args.ttl_days)
        except ValueError as exc:
            print(exc)
            return 1
        _print_token(rotated)
        return 0

    if args.cmd == "list":
        _print_json(store.list_principals())
        return 0

    if args.cmd == "purge-sessions":
        print(f"删掉 {store.purge_sessions()} 条过期/已吊销的登录会话。")
        return 0

    if args.cmd == "revoke":
        ok = store.revoke_principal(args.identifier, actor=args.actor)
        print("已吊销" if ok else "没有可吊销的 principal（不存在或已吊销）")
        return 0 if ok else 1

    if args.cmd == "audit":
        _print_json(store.recent_audit(args.limit))
        return 0

    parser.error("未知命令")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
