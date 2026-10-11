"""
合成 chunk 级评测集（生成脚本）

为什么要它
----------
原来 `ecom/knowledge/utils/eval.py` 里的评测集是 **source 级**的：问「期望的那份**文档**
有没有被召回」。期望答案只有十几个候选文档，随便召回 5 条都能"沾到"正确的那份
——所以 Recall@5 恒等于 100%，量不出任何东西。

问题**不在语料太大还是太小，在评测粒度太粗**。改成 chunk 级：期望答案从十几个
文档变成 184 个候选段落，同一套检索系统立刻分得开好坏，而且**不需要新语料**。
这就是这个脚本存在的理由。

题目怎么来
----------
把 chunk 喂给大模型，让它为每一段出两道题，两类都必须有——只有一类的话，
这个评测集**在原理上就测不出另一半能力**：

  - **改写型（paraphrase）**：模拟不懂行话的用户，完全用自己的措辞提问。
    考语义理解，向量路的强项。
  - **术语型（term）**：把这段里罕见的专有名词原样放进问题。
    考精确词面匹配，BM25 的强项。

不合格的题**直接丢**，理由是
----------------------------
大模型一定会偷懒照抄原文，而"照抄出来的改写型题目"是**最危险**的一种坏数据：
照抄意味着词面完全对上，BM25 必然赢，向量路被系统性低估——A/B 出来的结论
是错的，而且是往"混合检索更好"的方向错。所以生成的题要过一遍机器校验
（见 `reject_reason`），宁可少收，不能收脏的。

`reject_reason` 拦得住"照抄"和"改写型里混术语"，**拦不住**"这题其实另一段答得更好"
和"这题看着像样其实没意义"——那两种只能靠人看。所以每次生成完要跑一次抽检：

    uv run python build_eval_set.py --review 12

用法
----
    uv run python build_eval_set.py --dry-run      # 只看会抽到哪些 chunk，不调模型、不花钱
    uv run python build_eval_set.py                # 每份文档抽 40 段（默认）
    uv run python build_eval_set.py --limit 0      # 全量 253 段
    uv run python build_eval_set.py --review 12    # 抽检已生成的结果

产物 `data/knowledge/eval_set.jsonl` **提交进仓库**：它是评测基准，不是运行产物。
换语料或改了 prompt 之后要重新生成，并把新旧对比写进文档。
"""
import argparse
import json
import random
import re
from collections import Counter

from ecom.knowledge.retriever.bm25 import tokenize
from ecom.knowledge.settings import EVAL_SET_PATH
from ecom.knowledge.utils.json_utils import strip_to_json
from ecom.knowledge.utils.llm import get_llm
from ecom.knowledge.utils.logger import setup_logger
from ecom.knowledge.vector_store import get_vector_store

logger = setup_logger(__name__)

# --------------------------------------------------------------------------
# 参数
# --------------------------------------------------------------------------

BATCH_SIZE = 4                 # 一次喂几段给模型：太小贵、太大模型会漏段
DEFAULT_PER_SOURCE_LIMIT = 40  # 每份文档抽样数；0 表示全量
SEED = 20261010                # 固定种子，抽样可复现
GENERATION_TEMPERATURE = 0.5   # 要一点多样性，但别高到不守"别照抄"的规矩
GENERATION_TIMEOUT = 120

MIN_CHUNK_CHARS = 60      # 太短的块（标题、目录残片）出不了有区分度的题

# 长度上下限只用来挡"畸形输出"，**不是**用来规定问题该多长。
# 上限从 60 放宽到 120 是实测调出来的：中文里一个真实的用户提问动辄七八十字
# （"我装好了工具，想确认它在后台跑着，该敲哪条命令"这种），卡在 60 会把
# 一大堆**好题**当成坏数据丢掉，而且丢得毫无规律——越像真人话的越容易被丢。
# 现在这个上限只拦"模型把一整段话当问题交上来"这一类。
MIN_QUESTION_CHARS = 6
MAX_QUESTION_CHARS = 120
MIN_TERM_CHARS = 2
# 术语是**词汇**，不是从正文里剪下来的一段代码。
# 抽检时抓到过 term = `filter="id in [463480757150366907, 463480757150366908]"`
# 这种：把文档里一整行表达式连同具体 ID 原样当术语。它的问题在于——
# 拿一串 50 字的字面量去检索必然词面全中，测出来的不是"术语命中"，
# 是"字符串相等"，而且真人绝不会这么提问。
# 两个特征能干净地把它挡掉：真术语不含空格，也不会长到这个地步。
MAX_TERM_CHARS = 30
MAX_TERM_DF = 5           # 术语在整个语料里最多出现在几个 chunk 里，才算"罕见"
# 问题里连续这么多字与原文重合，就算照抄。
#
# 取 6 而不是更大：中文里 6 个字连在一起基本就是一个**短语**了（"检索增强生成"
# 正好 6 个），而短语级的重合恰恰是最常见的泄漏形态——模型不会整句照抄，
# 它会照抄短语。实测放到 8，这条规则在 30 多条改写型里一条都没拦下来，
# 等于没写。再小又会误伤真话（"需要注意的是"这种恰好撞上），6 是分界。
VERBATIM_N = 6

KIND_PARAPHRASE = "paraphrase"
KIND_TERM = "term"

SYSTEM_PROMPT = """你是检索系统的评测集出题人。用户会给你若干段技术文档，每段带一个编号。

给**每一段**各出 1 道「改写型」和 1 道「术语型」问题：

【改写型】模拟一个真实用户**带着需求来提问**：他心里有个问题要解决，但不知道这段资料里的专业说法。
  - **像真人一样直接问，别绕着弯描述概念**。把答案的构成一样样描述一遍、再问"这叫什么"，
    是坏题——真人不会这么说话，而且这种问法把答案的线索全泄在问题里了。
  - 句式短，尽量 40 字以内。不要出现这段资料里的任何专业术语，也别照抄原文的任何一句话。
  - 目的是考「语义理解」：问法和原文措辞完全不一样，只有真的理解了意思才可能召回得到。
  - 好例子：「我买的东西坏了，多久之内可以要求退换？」
  - 坏例子：「那个自开具发票之日起计算、用来办理修理换货退货的期限，叫什么？」

【术语型】从这段资料里挑一个**罕见的专有名词**（例如 个人信息处理者、快递运单、
  有奖销售 这类），把它**原样**放进问题里。
  - 要挑"罕见"的：像 消费者、平台、退货 这种整篇文档到处都是的词不算，起不到区分作用。
  - **术语是一个词**（个人信息处理者、快递运单、有奖销售 这样），
    不要把整句话、带标点的片段剪下来当术语——那是照抄，不是出题。
  - 这段里实在找不到合适的术语，term 和 term_question 都填 null，不要硬凑。

两类问题都必须满足：
1. 只靠**这一段**就能回答。靠常识、或靠别的段也能答出来的问题没有区分度，不要出。
2. 问题里不许出现定位信息（"上面这段""第 3 段""标题为 X 的章节"），那是作弊。

只输出一个 JSON 数组，不要 markdown 代码块，不要任何解释：
[{"index": 1, "paraphrase": "改写型问题", "term": "选中的术语原文或 null", "term_question": "术语型问题或 null"}, ...]
"""

# "第 3 段"这类定位信息：模型偶尔会把喂给它的编号抄进问题里
_POSITION_HINT = re.compile(r"第\s*[一二三四五六七八九十\d]+\s*段|上面那?段|上段|这段|上文|前述|上述")


# --------------------------------------------------------------------------
# 校验（纯函数，见 tests/test_build_eval_set.py）
# --------------------------------------------------------------------------


def normalize(text: str) -> str:
    """去掉空白和标点、转小写。

    做重合比对必须先归一化：原文"检索增强生成（RAG）"和问题"检索增强生成"，
    不归一化就只重合 6 个字、归一化后是 6 个字+括号，判断结果会随标点漂移。
    """
    return re.sub(r"[^\w]+", "", (text or "").lower())


def has_verbatim_run(question: str, text: str, n: int = VERBATIM_N) -> bool:
    """问题里有没有连续 n 个字原样出现在资料里

    问题本身比 n 还短时，把 n 降到问题长度——否则短问题**完全照抄**也判不出来
    （"什么是RAG"归一化才 6 个字，按 n=8 根本无从比对）。
    """
    question_norm = normalize(question)
    if len(question_norm) < 4:
        return False
    n = min(n, len(question_norm))
    text_norm = normalize(text)
    return any(
        question_norm[start:start + n] in text_norm
        for start in range(len(question_norm) - n + 1)
    )


class CorpusTerms:
    """词 → 它出现在多少个 chunk 里（document frequency）

    "罕见"只能相对整份语料说。在单个 chunk 内部统计没有意义：任何词在它自己
    那一段里 df 都是 1，全是"罕见词"。

    比对用**子串**而不是分词结果：jieba 很可能把 个人信息处理者 拆成"个人信息"和
    "处理者"两截，那样按整体查 df 会得到 0，把真正的罕见术语误判成"不存在"。
    """

    def __init__(self, texts):
        self._docs = [normalize(text) for text in texts]

    def df(self, term: str) -> int:
        needle = normalize(term)
        if not needle:
            return 0
        return sum(1 for doc in self._docs if needle in doc)


def leaked_terms(question: str, chunk_text: str, corpus: CorpusTerms) -> list:
    """改写型问题里混进来的罕见术语

    这是"别照抄"这条规矩的**第二条防线**。第一条是按连续 n 字比对，但它漏得掉
    2~7 字的术语（"向量库""重排序"这种），而中文里这类词恰恰是 BM25 最吃香的：
    换汤不换药地问"向量库是怎么把两路结果合起来的"，改写的努力就白费了。

    判定要求同时满足三条，才算了泄漏，避免误伤：
      1. 这个词出现在**这一段**里（否则跟本题无关）
      2. 它在整份语料里罕见（df <= MAX_TERM_DF）
      3. 它进了问题
    """
    chunk_tokens = set(tokenize(chunk_text))
    return sorted({
        token for token in set(tokenize(question))
        if len(token) >= MIN_TERM_CHARS
        and token in chunk_tokens
        and corpus.df(token) <= MAX_TERM_DF
    })


def reject_reason(question, kind, term, chunk, corpus):
    """返回 (原因类别, 细节)；合格返回 None

    分成"类别 + 细节"两截，是为了让汇总表按类别聚合。类别必须是**有限几个
    固定字符串**——否则每条都带一个术语名，统计出来就是一堆各出现一次的
    单例，"丢弃原因分布"这个表就没有信息量了。

    检查顺序是刻意的：先挡结构性问题（空、长度、定位词），再挡质量性问题。
    一个空字符串会同时违反好几条规则，顺序不定的话它落在哪个桶里全看运气。
    """
    if not isinstance(question, str) or not question.strip():
        return ("问题为空", "")
    question = question.strip()

    if not (MIN_QUESTION_CHARS <= len(question) <= MAX_QUESTION_CHARS):
        return ("问题长度不合适", f"{len(question)} 字")

    if _POSITION_HINT.search(question):
        return ("含定位信息", question[:20])

    chunk_text = chunk["text"]

    if kind == KIND_PARAPHRASE:
        if has_verbatim_run(question, chunk_text):
            return ("照抄原文", f"连续 {VERBATIM_N} 字重合")
        leaked = leaked_terms(question, chunk_text, corpus)
        if leaked:
            return ("改写型混进了术语", "/".join(leaked[:3]))
        return None

    if kind == KIND_TERM:
        if not isinstance(term, str) or not term.strip():
            return ("术语型没给术语", question[:20])
        term = term.strip()
        if len(term) < MIN_TERM_CHARS:
            return ("术语太短", term)
        if len(term) > MAX_TERM_CHARS or any(ch.isspace() for ch in term):
            return ("术语是一段代码而不是词", term)
        if normalize(term) not in normalize(chunk_text):
            return ("术语不在本段里", term)
        if normalize(term) not in normalize(question):
            return ("术语没写进问题", term)
        if corpus.df(term) > MAX_TERM_DF:
            return ("术语不够罕见", f"{term}（出现在 {corpus.df(term)} 段）")
        return None

    return ("未知题型", str(kind))


# --------------------------------------------------------------------------
# 生成
# --------------------------------------------------------------------------


def filter_by_sources(all_chunks, sources):
    """
    只保留文件名含指定子串的 chunk（sources 为逗号分隔的字符串，空则全要）。

    为什么需要它：加了示例语料之后，全库出题会把"教程"和"电商客服"混在一份评测集里。
    两套语料的检索难度不一样，混在一起的总分会互相抵消，看不出任何一边的变化。
    按来源切开，才能给新语料单独出一份 `data/knowledge/eval_set_business.jsonl`。

    用子串而不是精确文件名：`--sources 售后` 能一次圈住「售后政策.md」这类命名，
    不必把文件名一个个抄全；写全名也是一种子串，两种用法都成立。
    """
    if not sources:
        return list(all_chunks)
    needles = [item.strip() for item in sources.split(",") if item.strip()]
    if not needles:
        return list(all_chunks)
    return [chunk for chunk in all_chunks if any(n in chunk["source"] for n in needles)]


def sample_chunks(all_chunks, limit, seed=SEED):
    """按文档分层抽样，每份文档最多 limit 段（limit=0 表示全量）

    分层而不是全库随机抽：这十篇法规长短差得远，长的（网络交易监督管理办法）二十多段，
    短的（发票管理办法）只有 3 段——全库随机抽会让最短的那篇几乎抽不到题。分完层再抽，
    每篇都能被代表到。
    """
    by_source = {}
    for chunk in all_chunks:
        if len(chunk["text"]) < MIN_CHUNK_CHARS:
            continue
        by_source.setdefault(chunk["source"], []).append(chunk)

    rng = random.Random(seed)
    sampled = []
    for source in sorted(by_source):
        items = sorted(by_source[source], key=lambda c: c["chunk_id"])
        if limit and len(items) > limit:
            items = rng.sample(items, limit)
            items.sort(key=lambda c: c["chunk_id"])   # 排回去，让产物顺序稳定可 diff
        sampled.extend(items)
    return sampled


def build_batches(chunks, size=BATCH_SIZE):
    """按文档分批：不把两份文档的段塞进同一个 prompt，免得模型串了上下文"""
    batches = []
    for source in dict.fromkeys(chunk["source"] for chunk in chunks):
        group = [chunk for chunk in chunks if chunk["source"] == source]
        for start in range(0, len(group), size):
            batches.append(group[start:start + size])
    return batches


def build_prompt(chunks) -> str:
    # 编号写成 【1】 而不是 【第 1 段】：带上"段"字，模型会把编号原样抄进问题，
    # 于是整批被判成"含定位信息"丢掉。实测这么一改，那一类丢弃基本消失。
    blocks = [f"【{i}】\n{chunk['text']}" for i, chunk in enumerate(chunks, 1)]
    return f"下面是 {len(chunks)} 段资料：\n\n" + "\n\n".join(blocks)


def parse_batch(raw, chunks):
    """把一次模型输出摊平成 [{chunk, kind, question, term}]

    只做**结构**解析，不做质量判定——判定全在 reject_reason 里，这样
    "模型输出坏成什么样"和"题目本身不合格"两件事的排查是分开的。
    """
    text = strip_to_json(raw)
    if text is None:
        raise ValueError("模型输出里找不到 JSON")
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"模型输出的 JSON 不是数组，是 {type(data).__name__}")

    items = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        try:
            chunk = chunks[int(entry.get("index")) - 1]
        except (TypeError, ValueError, IndexError):
            # 模型编了个不存在的编号，或 index 不是数字——整条丢掉
            continue
        items.append({
            "chunk": chunk,
            "kind": KIND_PARAPHRASE,
            "question": entry.get("paraphrase"),
            "term": None,
        })
        items.append({
            "chunk": chunk,
            "kind": KIND_TERM,
            "question": entry.get("term_question"),
            "term": entry.get("term"),
        })
    return items


def generate(chunks, corpus):
    """跑完所有批次，返回 (合格题目, 丢弃原因计数)"""
    batches = build_batches(chunks)
    accepted, rejected = [], Counter()
    seen_questions = set()

    for number, batch in enumerate(batches, 1):
        try:
            response = get_llm(GENERATION_TEMPERATURE, GENERATION_TIMEOUT).invoke(
                [("system", SYSTEM_PROMPT), ("human", build_prompt(batch))]
            )
            items = parse_batch(response.content, batch)
        except Exception as exc:
            # 一批失败不影响其它批：重跑一次的成本远高于丢掉这 4 段
            logger.warning("第 %d 批生成失败，跳过：%s", number, exc)
            rejected["整批失败"] += len(batch)
            print(f"[{number}/{len(batches)}] 整批失败：{exc}")
            continue

        for item in items:
            verdict = reject_reason(
                item["question"], item["kind"], item["term"], item["chunk"], corpus
            )
            if verdict is None and normalize(item["question"]) in seen_questions:
                verdict = ("与已有题目重复", "")
            if verdict:
                category, detail = verdict
                rejected[category] += 1
                if detail:
                    logger.info("丢弃 [%s] %s：%s", category, detail, item["question"])
                continue

            chunk = item["chunk"]
            seen_questions.add(normalize(item["question"]))
            accepted.append({
                "id": f"{chunk['source'].removesuffix('.md')}-{chunk['chunk_id']}"
                      f"-{'p' if item['kind'] == KIND_PARAPHRASE else 't'}",
                "question": item["question"].strip(),
                "type": item["kind"],
                "term": item["term"],
                "expected_source": chunk["source"],
                "expected_chunk_id": chunk["chunk_id"],
                "chunk_head": chunk["text"][:60].replace("\n", " "),
            })

        print(f"[{number}/{len(batches)}] 已收 {len(accepted)} 条，丢弃 {sum(rejected.values())} 条")

    return accepted, rejected


# --------------------------------------------------------------------------
# 读 / 写 / 抽检
# --------------------------------------------------------------------------


def write_eval_set(cases, path):
    with open(path, "w", encoding="utf-8") as file:
        for case in cases:
            file.write(json.dumps(case, ensure_ascii=False) + "\n")


def load_eval_set(path):
    cases = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def review(path, count, seed=SEED):
    """抽检：把题目和它**来源的那段资料**并排打出来

    这是这个流程里唯一能发现"这题其实另一段答得更好""这题看着像样其实没意义"
    的环节。机器校验只能证明题目没有明显作弊，证明不了题目有意义。
    """
    cases = load_eval_set(path)
    if not cases:
        print(f"{path} 是空的，先跑一次生成。")
        return
    picked = random.Random(seed).sample(cases, min(count, len(cases)))
    for index, case in enumerate(picked, 1):
        label = "改写型" if case["type"] == "paraphrase" else f"术语型（{case['term']}）"
        print(f"\n{'=' * 70}")
        print(f"[{index}] {label}  ← {case['expected_source']} 第 {case['expected_chunk_id']} 段")
        print(f"问题：{case['question']}")
        print(f"该段开头：{case['chunk_head']}…")
    print(f"\n{'=' * 70}")
    print("逐条自问：这题是不是**只有这一段**答得了？靠常识能答吗？问法像真人吗？")


def summarize(cases, rejected):
    print(f"\n{'=' * 70}")
    print(f"合计 {len(cases)} 条题")
    for kind, label in ((KIND_PARAPHRASE, "改写型"), (KIND_TERM, "术语型")):
        group = [case for case in cases if case["type"] == kind]
        if not group:
            print(f"  {label}：0 条")
            continue
        lengths = sorted(len(case["question"]) for case in group)
        # 问题长度中位数是"这题像不像真人问的"的代理指标：真实提问一般四五十字，
        # 中位数窜到八九十就是模型又在绕圈子描述概念了，该去改 prompt 而不是调阈值。
        print(f"  {label}：{len(group)} 条，问题长度中位数 {lengths[len(lengths) // 2]} 字")
    sources = Counter(case["expected_source"] for case in cases)
    for source, count in sorted(sources.items()):
        print(f"  {source}：{count} 条")

    if rejected:
        print(f"\n丢弃 {sum(rejected.values())} 条，按原因：")
        for reason, count in rejected.most_common():
            print(f"  {count:>5}  {reason}")
        print(
            "\n提示：「改写型混进了术语」占比高是正常的——模型天生爱用术语。\n"
            "      但若高到改写型几乎不剩，说明 MAX_TERM_DF 太严，可放宽到 10。"
        )


def main():
    parser = argparse.ArgumentParser(description="合成 chunk 级评测集")
    parser.add_argument("--limit", type=int, default=DEFAULT_PER_SOURCE_LIMIT,
                        help="每份文档抽多少段，0 表示全量（默认 40）")
    parser.add_argument("--out", default=EVAL_SET_PATH, help="输出路径")
    parser.add_argument("--seed", type=int, default=SEED, help="抽样种子")
    parser.add_argument("--sources", default=None,
                        help="只对文件名含这些子串的文档出题，逗号分隔。"
                             "例：--sources 售后政策,退换货流程,商品FAQ,物流与发票 "
                             "（为示例语料单独出一份题时用）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印会抽到哪些段和首个 prompt，不调模型")
    parser.add_argument("--review", type=int, metavar="N",
                        help="抽检已生成的结果 N 条，不生成")
    args = parser.parse_args()

    if args.review:
        review(args.out, args.review, args.seed)
        return

    store = get_vector_store()
    try:
        all_chunks = store.all_chunks()
    finally:
        store.close()

    # 术语表仍按**全库**建：判断"这个词罕不罕见"要看它在整个知识库里有多常见，
    # 只用被选中的那几段去建，会让"只在示例语料里出现一次"的词也被当成罕见词。
    corpus = CorpusTerms([chunk["text"] for chunk in all_chunks])

    picked = filter_by_sources(all_chunks, args.sources)
    if args.sources:
        print(f"--sources {args.sources!r} 命中 {len(picked)}/{len(all_chunks)} 段")
    chunks = sample_chunks(picked, args.limit, args.seed)

    print(f"语料共 {len(picked)} 段，本次抽样 {len(chunks)} 段：")
    for source, count in sorted(Counter(c["source"] for c in chunks).items()):
        print(f"  {source}：{count} 段")

    batches = build_batches(chunks)
    print(f"将分成 {len(batches)} 批（每批 {BATCH_SIZE} 段），约 {len(batches)} 次模型调用")

    if args.dry_run:
        print("\n" + "=" * 70)
        print("第一个批次的完整 prompt：\n")
        print("【system】")
        print(SYSTEM_PROMPT)
        print("【human】")
        print(build_prompt(batches[0]))
        return

    cases, rejected = generate(chunks, corpus)
    write_eval_set(cases, args.out)
    summarize(cases, rejected)
    print(f"\n已写入 {args.out}")
    print("接着跑一次抽检：uv run python build_eval_set.py --review 12")


if __name__ == "__main__":
    main()
