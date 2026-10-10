"""
待审核队列的审核工作台

问答过程中，大模型会判断每个提问是否值得沉淀，值得的进队列等人审。
这个脚本就是那个"人"的操作台。

用法：
    uv run python review.py list              # 待审核 + 模型判为值得
    uv run python review.py list --all        # 连模型判为无价值的也列出来，用来看判定准不准
    uv run python review.py show 3
    uv run python review.py approve 3 --note "已核对，补了版本号"
    uv run python review.py reject 3 --note "一次性问题"

approve 会把内容落地成 data/documents/reviewed/ 下的一个 md 文件，
**然后需要你自己跑一次 `uv run python ingest.py` 才真正进向量库。**

为什么不在这里顺手入库：ingest.py 是向量库的唯一写入口。
多开一条写入路径，两边日后必然漂移。add_chunks 已按 source 幂等，
全量重跑的成本可以忽略，换来的是"谁写了向量库"永远只有一个答案。
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from ecom.knowledge.review import queue
from ecom.knowledge.settings import REVIEWED_DOCS_PATH

# 不要在这里 reconfigure stdout 成 UTF-8。
# Windows 控制台是 cp936，Python 默认按 GBK 输出、控制台按 GBK 解码，本来是对的；
# 强行改 UTF-8 反而会输出乱码。文件读写的编码另外显式指定（见 render_markdown 的调用处）。


def fmt_score(value) -> str:
    """把分数印成两位小数；拿不到数就留空位对齐（app.py 的界面也在用）"""
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return " -  "


def _slug(text: str, limit: int = 30) -> str:
    """把问题压成安全的文件名片段"""
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "_", (text or "").strip())
    return cleaned.strip("_")[:limit] or "question"


def render_markdown(item: dict, note: str = None) -> str:
    """把一条待审核记录渲染成入库用的 md 内容"""
    lines = [
        f"# {item['question']}",
        "",
        item.get("answer") or "（模型未给出答案）",
        "",
        "---",
        f"<!-- 来源：用户提问 #{item['id']}"
        f"｜模型判定 category={item.get('category')} score={fmt_score(item.get('score'))}"
        f"｜被问到 {item.get('ask_count', 1)} 次 -->",
        f"<!-- 知识库命中 {item.get('kb_hits')} 条"
        f"（最高相似度 {fmt_score(item.get('kb_top_score'))}），"
        f"联网命中 {item.get('web_hits')} 条 -->",
    ]
    if note:
        lines.append(f"<!-- 审核备注：{note} -->")
    lines.append("<!-- 内容为 AI 生成、经人工审核通过，不是人工撰写的原文 -->")
    return "\n".join(lines) + "\n"


def cmd_list(args) -> int:
    items = queue.list_items(
        worthy_only=not args.all, status=args.status, limit=args.limit
    )
    scope = "全部（含模型判为无价值的）" if args.all else "模型判为值得沉淀的"

    print(f"待审核队列：{scope}，状态 {args.status}，共 {len(items)} 条\n")

    if not items:
        print("没有待审核的记录。问答时如果有值得沉淀的提问，会自动进这里。")
        print(f"（队列文件：{queue.REVIEW_DB_PATH}）")
        return 0

    print(f"{'ID':>4}  {'分数':>4}  {'类别':<13}{'次数':>4}  {'状态':<8} 问题")
    print("-" * 90)
    for item in items:
        question = item["question"].replace("\n", " ")
        if len(question) > 42:
            question = question[:42] + "…"
        print(
            f"{item['id']:>4}  {fmt_score(item.get('score')):>4}  "
            f"{(item.get('category') or '-'):<13}{item.get('ask_count', 1):>4}  "
            f"{item['status']:<8} {question}"
        )

    print("-" * 90)
    print("查看详情：uv run python review.py show <ID>")
    return 0


def cmd_show(args) -> int:
    item = queue.get_item(args.id)
    if not item:
        print(f"没有 id={args.id} 的记录")
        return 1

    print("=" * 70)
    print(f"#{item['id']}  状态：{item['status']}")
    print("=" * 70)
    print(f"问题：{item['question']}\n")
    print(f"答案：\n{item.get('answer') or '（无）'}\n")
    print("-" * 70)
    print(f"模型判定：{'值得沉淀' if item['worthy'] else '不值得'}"
          f"（{item.get('category')}，分数 {fmt_score(item.get('score'))}）")
    print(f"判定理由：{item.get('reason') or '（无）'}")
    print(f"知识库命中 {item.get('kb_hits')} 条，最高相似度 {fmt_score(item.get('kb_top_score'))}")
    print(f"联网命中 {item.get('web_hits')} 条")
    print(f"被问到 {item.get('ask_count', 1)} 次｜创建 {item.get('created_at')}｜更新 {item.get('updated_at')}")

    sources = json.loads(item.get("sources_json") or "[]")
    if sources:
        print("引用来源：")
        for source in sources:
            print(f"  - {source}")

    if item.get("doc_path"):
        print(f"\n落地文件：{item['doc_path']}")
    if item.get("review_note"):
        print(f"审核备注：{item['review_note']}")
    return 0


def approve_item(item_id: int, note: str = None) -> dict:
    """
    通过一条待审核记录：渲染成 md 落到 data/documents/reviewed/，并把状态改成 approved。

    返回 {"ok", "path", "already", "message"}，**不打印**——命令行印中文提示、
    Streamlit 渲染成 st.success，展示归调用方（和 judge.record_question 一个套路）。

    放 review.py 而不是 queue.py：queue.py 开头就声明自己只管队列的存取，
    替它去落盘会破坏那条边界；而且 render_markdown / _slug 本来就在这边。

    注意这只是"落地成文件"，还没进向量库——入库的唯一入口仍是 ingest.py。
    """
    item = queue.get_item(item_id)
    if not item:
        return {"ok": False, "path": None, "already": False,
                "message": f"没有 id={item_id} 的记录"}

    # 幂等：已经通过过的直接返回原文件，不重写。
    # 重写会覆盖掉人工在那个 md 上做的修改——那份文件是"权威内容"，不能悄悄盖掉。
    if item["status"] == "approved" and item.get("doc_path"):
        return {"ok": True, "path": item["doc_path"], "already": True,
                "message": f"#{item_id} 已经审核通过了：{item['doc_path']}"}

    Path(REVIEWED_DOCS_PATH).mkdir(parents=True, exist_ok=True)
    filename = f"{datetime.now():%Y%m%d}_{_slug(item['question'])}_{item_id}.md"
    path = Path(REVIEWED_DOCS_PATH) / filename
    path.write_text(render_markdown(item, note), encoding="utf-8")

    queue.set_status(item_id, "approved", note=note, doc_path=str(path))

    return {"ok": True, "path": str(path), "already": False,
            "message": f"已通过 #{item_id}，落地到：\n  {path}"}


def cmd_approve(args) -> int:
    result = approve_item(args.id, args.note)
    print(result["message"])
    if not result["ok"]:
        return 1
    if result["already"]:
        return 0

    print("\n建议先打开这个文件读一遍、该改的地方改掉——它进库之后就是这个知识库的"
          "\n'权威内容'了。确认没问题再执行入库：\n  uv run python ingest.py")
    return 0


def cmd_reject(args) -> int:
    item = queue.get_item(args.id)
    if not item:
        print(f"没有 id={args.id} 的记录")
        return 1

    queue.set_status(args.id, "rejected", note=args.note)
    print(f"已拒绝 #{args.id}（不会进知识库）")
    if args.note:
        print(f"备注：{args.note}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="待审核队列操作台",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="列出待审核记录")
    p_list.add_argument("--all", action="store_true",
                        help="连模型判为无价值的也列出来，用来看判定准不准")
    p_list.add_argument("--status", default="pending",
                        choices=queue.STATUSES, help="按状态筛选（默认 pending）")
    p_list.add_argument("--limit", type=int, default=50)
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="看某条记录的完整现场")
    p_show.add_argument("id", type=int)
    p_show.set_defaults(func=cmd_show)

    p_approve = sub.add_parser("approve", help="通过，并落地成待入库的 md")
    p_approve.add_argument("id", type=int)
    p_approve.add_argument("--note", default=None, help="审核备注")
    p_approve.set_defaults(func=cmd_approve)

    p_reject = sub.add_parser("reject", help="拒绝")
    p_reject.add_argument("id", type=int)
    p_reject.add_argument("--note", default=None, help="拒绝原因")
    p_reject.set_defaults(func=cmd_reject)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not getattr(args, "func", None):
        parser.print_help()
        print(f"\n队列概览：{queue.stats()}")
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
