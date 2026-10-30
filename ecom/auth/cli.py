"""principal 的命令行管理。

    python -m ecom.auth.cli create --name alice --role admin
    python -m ecom.auth.cli list
    python -m ecom.auth.cli revoke alice
    python -m ecom.auth.cli audit --limit 20
"""
import argparse
import json

from ecom.auth import store


def _print_json(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ecom.auth.cli", description="身份管理")
    sub = parser.add_subparsers(dest="cmd", required=True)

    create = sub.add_parser("create", help="创建 principal，token 只显示一次")
    create.add_argument("--name", required=True)
    create.add_argument("--role", default="customer", choices=store.ROLES)

    sub.add_parser("list", help="列出 principal（不含 token）")

    revoke = sub.add_parser("revoke", help="按 id 或 name 吊销")
    revoke.add_argument("identifier")
    revoke.add_argument("--actor", default="cli")

    audit = sub.add_parser("audit", help="看最近的审计")
    audit.add_argument("--limit", type=int, default=20)

    args = parser.parse_args(argv)

    if args.cmd == "create":
        created = store.create_principal(args.name, args.role)
        _print_json({k: v for k, v in created.items() if k != "token"})
        print("\nTOKEN（只显示这一次，服务端只存哈希）：")
        print(created["token"])
        return 0

    if args.cmd == "list":
        _print_json(store.list_principals())
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
