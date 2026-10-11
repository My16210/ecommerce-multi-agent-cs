"""抓取法规原文，生成知识库语料（一次性构建脚本）。

为什么是"法规原文"而不是电商平台的帮助中心
--------------------------------------------
要的是**真实的中文电商售后文档**。先试过平台帮助中心，两条路都断：

  - 淘宝 rule.taobao.com 连不上；京东/拼多多/苏宁/唯品的帮助页返回的都是
    JS 空壳（京东那篇正文页只有 1292 个中文字，文章页直接返回"页面出现异常"）。
  - 就算抓到了也不能用：商业网站的帮助中心是它的著作权内容，整站抄进一个
    公开仓库是有问题的。

法规原文两头都过得去：gov.cn / npc.gov.cn / samr.gov.cn 返回的是**静态 HTML 正文**
（不是前端渲染），而且**《著作权法》第五条**明确"法律、法规，国家机关的决议、
决定、命令和其他具有立法、行政、司法性质的文件"不适用著作权法——可以合法地
随仓库再分发。这不是"找得到就凑合用"，是这批材料本来就是公有领域。

代价写在前面：法规是**条文**，不是客服话术。用户问"我买的东西坏了怎么办"，
检索到的是"三包规定第九条"，而不是一段现成的安抚话术。对这个项目来说这反而
更真实——真实客服系统背后的知识库本来就该是规则本身。

为什么用标准库而不是 requests + BeautifulSoup
----------------------------------------------
本机 PyPI 不通（宿主装不了包），而容器里装它只为了跑一次的东西不值。
urllib + html.parser 够用：这些是静态政府页面，不需要处理 JS 渲染。
顺带一提 **curl 抓 samr.gov.cn 会段错误**，urllib 不会——所以这里没走 curl。

为什么定位正文**不靠 class 选择器，靠法条自己的结构**
------------------------------------------------------
第一版是给每个来源配一个容器 class（UCAP-CONTENT / Zoom / ...），再按标签配对
切出来。10 个源里 6 个抓回来只有几个中文字——不是网站改版了，是同一个词在页面里
出现好几次（<style> 里一次、工具栏 class 上又一次），`find` 取第一处就赌错了。
改成"每处都切开、取最长的那个"之后 9 个过了，剩下的 1 个（江西的实施条例）
div 嵌套不闭合——配对计数会在一个多余的 </div> 上提前收工。

而换成"整页抽文本，再用法条自己的结构去截"之后 10 个全过，且**每一篇都比
容器法抓得多**。所以容器那套整套删掉了：它没带来一行额外的好处，只带来了
两种失效方式。留下的是三条关于法条本身的规则（见 _trim_body），不依赖任何一个
网站的排版——网站改版不会让它们失效，换个来源也不用改代码。

抓完必须**校验**（见 _check）：中文字数和"第X条"条数都得够，不够就报错、
不写文件。抓取失败最坏的样子不是报错，是**悄悄写进去一份残缺的法条**，
然后在检索评测里表现为"召回质量下降"，查半天查不到是抓取的问题。
"""
from __future__ import annotations

import argparse
import html as html_mod
import re
import sys
import urllib.request
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

from ecom.knowledge.settings import DOCUMENTS_PATH

# 原文落这里（不进仓库，见 .gitignore）。留着是为了改截断规则时不用重新抓。
RAW_DIR = Path(DOCUMENTS_PATH).parent / "source"

# 语料子目录。和「电商客服示例」并列——那几份是场景化的问答式材料，这几份是
# **规则本身**，两者回答的问题类型不同，混在一个目录里反而看不出这个区别。
CORPUS_SUBDIR = "法规原文"

_CJK = re.compile(r"[一-鿿]")
_ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百零〇\d]+条")
_CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百零〇\d]+[章节]")

# ===========================================================================
# 来源表
#
#   min_cn / min_articles：校验门槛，按实测值留一成左右余量。它只负责拦
#       "抓回来一个错误页"和"正文被截没了"，不是精确值——写死成实测值会
#       因为网站加一行版权声明就假红。
#   note：公布与修订信息。**不抓页面上的那一段**（它排版不固定、经常和
#       主席令之类的版式混在一起），而是我核对过之后手写在这里。
# ===========================================================================
SOURCES = [
    {
        "file": "网络购买商品七日无理由退货暂行办法.md",
        "title": "网络购买商品七日无理由退货暂行办法",
        "note": "2017年1月6日国家工商行政管理总局令第90号公布，根据2020年10月23日国家市场监督管理总局令第31号修订",
        "url": "https://www.gov.cn/zhengce/zhengceku/2020-11/03/content_5557118.htm",
        "min_cn": 2800,
        "min_articles": 36,
    },
    {
        "file": "中华人民共和国电子商务法.md",
        "title": "中华人民共和国电子商务法",
        "note": "2018年8月31日第十三届全国人民代表大会常务委员会第五次会议通过，自2019年1月1日起施行",
        "url": "http://www.npc.gov.cn/npc/c2/c30834/201905/t20190521_281583.html",
        "min_cn": 8500,
        "min_articles": 80,
    },
    {
        "file": "中华人民共和国消费者权益保护法.md",
        "title": "中华人民共和国消费者权益保护法",
        "note": "1993年10月31日第八届全国人民代表大会常务委员会第四次会议通过，2013年10月25日第十二届全国人民代表大会常务委员会第五次会议修正，自2014年3月15日起施行",
        "url": "https://www.gov.cn/ldhd/2013-10/25/content_2515324.htm",
        "min_cn": 6500,
        "min_articles": 60,
    },
    {
        "file": "快递暂行条例.md",
        "title": "快递暂行条例",
        "note": "2018年3月2日国务院令第697号公布，自2018年5月1日起施行",
        "url": "https://www.gov.cn/zhengce/content/2018-03/27/content_5277801.htm",
        "min_cn": 5600,
        "min_articles": 48,
    },
    {
        "file": "网络交易监督管理办法.md",
        "title": "网络交易监督管理办法",
        "note": "2021年3月15日国家市场监督管理总局令第37号公布，自2021年5月1日起施行",
        "url": "https://www.gov.cn/zhengce/zhengceku/2021-03/16/content_5593226.htm",
        "min_cn": 6400,
        "min_articles": 50,
    },
    {
        "file": "规范促销行为暂行规定.md",
        "title": "规范促销行为暂行规定",
        "note": "2020年10月29日国家市场监督管理总局令第32号公布，自2020年12月1日起施行",
        "url": "https://www.gov.cn/zhengce/zhengceku/2020-11/06/content_5557730.htm",
        "min_cn": 2400,
        "min_articles": 30,
    },
    {
        "file": "部分商品修理更换退货责任规定.md",
        "title": "部分商品修理更换退货责任规定（三包规定）",
        "note": "1995年8月25日国家经贸委、国家技术监督局、国家工商局、财政部发布，国经贸质〔1995〕458号",
        "url": "https://www.samr.gov.cn/zw/zfxxgk/zc/xzgfxwj/art/2019/art_f943e66de125437d92df701e1654c47c.html",
        "min_cn": 1900,
        "min_articles": 22,
    },
    {
        "file": "中华人民共和国个人信息保护法.md",
        "title": "中华人民共和国个人信息保护法",
        "note": "2021年8月20日第十三届全国人民代表大会常务委员会第三十次会议通过，自2021年11月1日起施行",
        "url": "http://www.npc.gov.cn/npc/c2/c30834/202108/t20210820_313088.html",
        "min_cn": 7500,
        "min_articles": 70,
    },
    {
        "file": "中华人民共和国发票管理办法.md",
        "title": "中华人民共和国发票管理办法",
        "note": "1993年12月23日财政部令第6号发布，2023年7月20日国务院令第764号第三次修订",
        "url": "https://liaoning.chinatax.gov.cn/art/2023/7/20/art_5869_7634.html",
        "min_cn": 3500,
        "min_articles": 38,
    },
    {
        "file": "中华人民共和国消费者权益保护法实施条例.md",
        "title": "中华人民共和国消费者权益保护法实施条例",
        "note": "2024年3月15日国务院令第778号公布，自2024年7月1日起施行",
        "url": "https://amr.jiangxi.gov.cn/amr/zcwj129/pc/content/content_1902972746679164928.html",
        "min_cn": 5500,
        "min_articles": 50,
    },
]


class _TextExtractor(HTMLParser):
    """把 HTML 转成按段落分行、不留标签的纯文本。

    只看块级标签决定换行：行内标签（a/span/strong）不该断行，否则一个句子会被
    切得七零八落——而切分器是按行/句号切的，行被切碎会直接毁掉 chunk 质量。
    """

    _SKIP = {"script", "style", "noscript"}
    _BLOCK = {
        "p", "div", "br", "li", "tr", "td", "table", "section", "article",
        "h1", "h2", "h3", "h4", "h5", "h6", "blockquote",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skipping = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skipping += 1
        if tag in self._BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skipping:
            self._skipping -= 1
        if tag in self._BLOCK:
            self._parts.append("\n")

    def handle_data(self, data):
        if not self._skipping:
            self._parts.append(data)

    def text(self) -> str:
        raw = html_mod.unescape("".join(self._parts))
        raw = re.sub(r"[ \t　\xa0\r]+", " ", raw)
        lines = [line.strip() for line in raw.split("\n")]
        return "\n".join(line for line in lines if line)


def html_to_text(page_html: str) -> str:
    parser = _TextExtractor()
    parser.feed(page_html)
    return parser.text()


def _first_match_line(lines: list[str], pattern: re.Pattern) -> int | None:
    for index, line in enumerate(lines):
        if pattern.match(line):
            return index
    return None


def _trim_body(text: str) -> str:
    """从整页文本里切出法条正文。三条规则，都只依赖法条自己的结构。

    一、**头**：切到第一个「第X条」——注意是条不是章。有的法条页在正文之前
        先放一份"章目录"（只有第一章、第二章……没有条文），拿章当锚点会把整个
        目录当成正文开头；拿条当锚点自然就跳过去了。实在没有条（不成文的文件）
        才退回用章。标题、导航、发布日期都在这之前，不用另外清。
        开头那个「第X条」也就是下面去重用的锚点。

    二、**重**：如果开头那条（"第一条"）在正文里又出现了一次，就在第二次出现处
        切断。很多政府页面除了正文还嵌一份"打印/分享"用的副本，全文照抄一遍——
        网络购买商品七日无理由退货暂行办法和网络交易监督管理办法都是这样，
        不切的话**每一条都在语料里存了两份**（前者的中文字数会从 3220 变成 6495，
        后者的条文数直接翻倍）。判据是法条本身的性质："第一条"在一部法里
        只会出现一次，出现两次一定是页面重复了内容。

        （先试过靠 class 容器躲开重复，结果容器法在别的站上更差——见模块开头。
        现在这条重复规则反而比容器更准，因为它读的是内容，不是排版。）

    三、**尾**：从后往前找到最后一段"以句号收尾"的正文，它之后的全部丢掉。
        站点家具（相关链接、"责任编辑"、推荐阅读标题）不以句号收尾，正文
        段落则是。用"句号"当边界不用维护站点清单，换个来源也不用改。
        它的短板是**清不掉夹在正文和结尾之间的**家具：少数页面会把页脚放在
        正文前面（如快递暂行条例那页），末段之后又跟着一段公布令，中间那行
        ICP 备案号就会被留下。一行，认了——为它加规则属于按站点打补丁。
    """
    lines = text.split("\n")

    start = _first_match_line(lines, _ARTICLE_RE)
    if start is None:
        start = _first_match_line(lines, _CHAPTER_RE)
    body = lines[start:] if start is not None else lines

    if body:
        opener = _ARTICLE_RE.match(body[0]) or _CHAPTER_RE.match(body[0])
        if opener:
            token = opener.group(0)
            for index in range(1, len(body)):
                if body[index].startswith(token):
                    body = body[:index]
                    break

    end = len(body)
    while end > 0 and not body[end - 1].rstrip().endswith("。"):
        end -= 1
    if end == 0:  # 整篇都不以句号结尾（截断规则失效），原样返回让校验去拦
        return "\n".join(body)
    return "\n".join(body[:end])


def _check(body: str, spec: dict) -> list[str]:
    """校验。返回问题列表，空表示通过。"""
    problems = []
    chinese = len(_CJK.findall(body))
    articles = len(_ARTICLE_RE.findall(body))
    if chinese < spec["min_cn"]:
        problems.append(f"中文字只有 {chinese}，低于门槛 {spec['min_cn']}")
    if articles < spec["min_articles"]:
        problems.append(f"只有 {articles} 个「第X条」，低于门槛 {spec['min_articles']}")
    if "<" in body and ">" in body:
        problems.append("正文里还有尖括号，可能没洗干净 HTML")
    if not body.startswith("第"):
        problems.append(f"正文没有从条文开头开始，而是：{body[:40]!r}")
    if not body.rstrip().endswith(("。", "）", "；")):
        problems.append(f"正文结尾不像一个完整的条文末句：{body[-30:]!r}")
    return problems


def _front_matter(spec: dict) -> str:
    """给每份语料写清出处。**这是能不能合法再分发的依据**，不是装饰。"""
    return (
        f"# {spec['title']}\n\n"
        f"> **来源**：{spec['url']}\n"
        f"> **公布/修订**：{spec['note']}\n"
        f"> **抓取日期**：{date.today():%Y-%m-%d}\n"
        f"> **可再分发性**：本文为法规原文，依《中华人民共和国著作权法》第五条\n"
        f"> （法律、法规，国家机关的决议、决定、命令和其他具有立法、行政、司法性质\n"
        f"> 的文件，不适用著作权法）属公有领域，可随本仓库再分发。\n"
        f"> **说明**：正文由 `ecom/knowledge/cli/fetch_corpus.py` 抓取并去除了站点导航，\n"
        f"> 未经人工校对；以官方发布版本为准。\n\n"
        f"---\n\n"
    )


def fetch(url: str, timeout: int = 45) -> str:
    request = urllib.request.Request(
        url,
        headers={
            # 不带 UA 会被部分政府站点拒绝。这里不伪装浏览器指纹，只表明是一个
            # 普通的抓取客户端——冒充浏览器骗过反爬不是这个脚本该做的事。
            "User-Agent": "Mozilla/5.0 (compatible; corpus-fetch/1.0)",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, "replace")


def build(only: str = "", dry_run: bool = False, inspect: bool = False) -> int:
    corpus_dir = Path(DOCUMENTS_PATH) / CORPUS_SUBDIR
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        corpus_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for spec in SOURCES:
        if only and only not in spec["file"]:
            continue

        raw_path = RAW_DIR / (Path(spec["file"]).stem + ".html")
        try:
            if raw_path.exists():
                page_html = raw_path.read_text(encoding="utf-8", errors="replace")
                origin = "本地缓存"
            else:
                page_html = fetch(spec["url"])
                raw_path.write_text(page_html, encoding="utf-8")
                origin = "已抓取"
        except Exception as exc:  # noqa: BLE001 —— 一个源挂了不该让整批停摆
            print(f"  ✗ {spec['title']}：抓取失败 {type(exc).__name__}: {exc}")
            failed.append(spec["file"])
            continue

        body = _trim_body(html_to_text(page_html))
        chinese = len(_CJK.findall(body))
        articles = len(_ARTICLE_RE.findall(body))

        if inspect:
            print(f"\n=== {spec['title']}  [{chinese} 中文字，{articles} 条] ===")
            for line in body.split("\n")[:2]:
                print(f"  头 | {line[:110]}")
            for line in body.split("\n")[-3:]:
                print(f"  尾 | {line[:110]}")
            continue

        problems = _check(body, spec)
        if problems:
            print(f"  ✗ {spec['title']}：{'；'.join(problems)}")
            failed.append(spec["file"])
            continue

        if dry_run:
            print(f"  · {spec['title']}（{origin}，{chinese} 中文字，{articles} 条）→ 未写入")
        else:
            target = corpus_dir / spec["file"]
            target.write_text(_front_matter(spec) + body + "\n", encoding="utf-8")
            print(f"  ✓ {spec['title']}（{origin}，{chinese} 中文字，{articles} 条）")

    if failed:
        print(f"\n{len(failed)} 个来源失败：{'、'.join(failed)}")
        print("**没有写入的源不算语料**——不要用半份法条去跑评测。")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="抓取法规原文生成知识库语料")
    parser.add_argument("--only", default="", help="只处理文件名里含这段文字的源")
    parser.add_argument("--dry-run", action="store_true", help="只抓取和校验，不写文件")
    parser.add_argument("--inspect", action="store_true", help="打印每篇的开头结尾，人工核对截断")
    args = parser.parse_args()

    print(f"语料目录：{Path(DOCUMENTS_PATH) / CORPUS_SUBDIR}")
    print(f"原文缓存：{RAW_DIR}")
    code = build(only=args.only, dry_run=args.dry_run, inspect=args.inspect)
    if code == 0 and not args.dry_run and not args.inspect:
        print("\n下一步：重跑入库（语料变了，向量库必须重建）")
        print("    python -m ecom.knowledge.cli.ingest")
    return code


if __name__ == "__main__":
    sys.exit(main())
