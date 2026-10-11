"""
答案质量评测：忠实性 / 相关性 / 弃答

**为什么要有这个东西。** 在这之前，这个项目里所有指标（Recall@5、MRR、精度）
量的全是**检索**——「有没有找回正确的段落」。但用户拿到的是**答案**，中间还隔着
一步「模型有没有忠实使用这些段落」。召回率再高也推不出「答案是对的」。
这个模块补的就是这条腿。

三个维度，都不需要参考答案（所以能直接跑在现有题库上，不用再开一个标注活）：

    忠实性  answer 里的每条事实性声明，能不能在检索到的资料里找到依据
    相关性  answer 有没有切题回答 question
    弃答    语料里根本没有答案时，该不该承认"不知道"而不是编一个

**判定者就是被测者，这一点必须先说清楚**：judge 和 generator 都是 qwen-plus。
所以报告头部第一行会写明「这些分数是模型自评」。三道防线：
`selftest` 用金丝雀验判定者本身、`calibrate` 人工盲判算一致率、以及把判定
做成**清单式**（拆声明逐条核验）而不是整体打分——后者难糊弄得多。

三步：build（生成答案快照）→ grade（判定）→ report（出数）。分开是为了可复现：
判定提示词改了可以只重判不重生成，报告永远读同一份快照。

**为什么绕开 LangGraph 图，直接调 retrieve() + generate_answer()**：
弃答要可测，就必须关掉联网——图里 web_search_node 和检索并行，语料里没有答案的
问题会被联网**合法地**救回来，那是正确行为不是失败。直接传 web_contexts=[]
等于"模拟联网返回空"，两个维度才测得干净。另外这样是单轮（不传 chat_history）、
也无副作用（不走 answer_node，不触发 record_question 写队列）。

用法：
    uv run python answer_eval.py selftest              # 先验判定者本身，不花生成的钱
    uv run python answer_eval.py build --n 30          # 生成答案快照
    uv run python answer_eval.py grade                 # 判定
    uv run python answer_eval.py report                # 出数
    uv run python answer_eval.py calibrate --n 20      # 人工盲判，算一致率
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from ecom.knowledge.cli.gold_set import stratified_sample
from ecom.knowledge.cli.tune_rrf import wilson_half_width
from ecom.knowledge.core.graph import ANSWER_FALLBACK
from ecom.knowledge.generator.answer import generate_answer
from ecom.knowledge.retriever.retriever import retrieve
from ecom.knowledge.settings import (
    ANSWER_EVAL_PATH,
    QWEN_PRICE_INPUT,
    QWEN_PRICE_OUTPUT,
    TOP_K,
    UNANSWERABLE_PATH,
)
from ecom.knowledge.utils.eval import load_eval_set
from ecom.knowledge.utils.json_utils import strip_to_json
from ecom.knowledge.utils.llm import get_llm
from ecom.knowledge.utils.timing import metric, stage
from ecom.knowledge.vector_store import get_vector_store

# 和 gold_set 的种子刻意错开：两个工具抽到同一批题没有额外价值，错开反而多覆盖一点。
# 固定住是为了让「上次 88%、这次 85%」能分清是答案变了还是题换了。
SEED = 20260202

# 判定用 temperature=0：这是打分，不是创作，同一份输入必须给同一个结论。
TEMPERATURE = 0
TIMEOUT = 60

# 判定输出解析不出来时重试几次。**temperature=0 不等于确定性**：实测同一个
# 答案重跑一遍就正常返回了（MoE 路由、批处理等都会引入抖动）。所以把解析失败
# 当偶发故障重试，比降低解析标准强——放宽解析会把真坏的输出也放进来。
PARSE_ATTEMPTS = 3

# 「不可答题」的**复核提示线**：检索 top 余弦相似度超过它就提示人工看一眼
# 这题是不是其实答得出来。注意它是提示，不是判据——见 build 里那段说明。
#
# 为什么是 0.65 而不是拍脑袋的 0.75：
#   第一版写的是 0.75，而且比的是 RRF 融合分。RRF 分的量纲是 1/(K+rank)，
#   实测恒在 0.031～0.033 之间，**永远够不到 0.75**——那条自检是死的，
#   它一声不吭地放过了两条标错的题（unans-05「服务器配置」、unans-07「积分提现」，
#   语料里其实写得明明白白）。改成比余弦之后才有意义。
#   0.65 这个分界是实测标出来的：那两条判错的余弦是 0.662 / 0.706，
#   而 6 条真查不到的落在 0.530～0.642。**只有 8 个样本标定**，样本一多就该重估。
#
# 已知会误报，而且已经实测到一次：换成余弦自检后跑的第一遍，
# 「开具发票的税率是多少？」以 0.661 越过提示线，但 grep 语料发现
# 物流与发票.md 通篇讲发票类型/开票时间/红冲，**从头到尾没有税率**——
# 它在讲同一件事，但没有这个事实。余弦分不出这两者。
# 这正是"提示不是判据"的意思：越过线只表示"该去 grep 一下"，不表示"标错了"。
UNANSWERABLE_COSINE_REVIEW = 0.65

FAITHFULNESS_LABELS = ("supported", "unsupported", "contradicted")
RELEVANCE_LABELS = ("answered", "partial", "irrelevant", "refused")

FAITHFULNESS_PROMPT = """你是 RAG 系统的「忠实性」评审员。给你【检索到的资料】和【助手答案】，
判断答案里的每一句话有没有资料的支撑。

做法：
1. 把答案拆成若干条**独立的事实性声明**。一句话里含两个事实就拆成两条。
   下面这些**都不算声明**，一律不要列进去：
   - 客套话、纯过渡句（"好的""希望对你有帮助"）
   - 表示"资料里没有相关内容"的说明（"提供的资料中没有相关内容，无法回答"）
     ——它不是在陈述世界上的事实，而是在说明资料的状态。
   整段答案都是这类说明时，claims 就该是**空数组**，不要硬凑一条出来。
2. 逐条核对，label 三选一：
   - supported    资料里能直接找到依据
   - unsupported  资料里没有任何依据（编造，或超出资料的推断）
   - contradicted 与资料的说法矛盾
3. evidence 填资料里**能支撑该条的原句片段**；unsupported / contradicted 填空串。

判定要**严**：只要资料里找不到明确依据就是 unsupported，不要因为"这话听起来对"、
"一般常识如此"、"应该是这样"就放它过。这一条是本次评审的全部意义所在。

只输出一个 JSON 对象，不要任何其他文字，不要用 markdown 代码块包裹：
{"claims": [{"claim": "声明原文", "label": "supported|unsupported|contradicted", "evidence": "资料原句或空串"}], "score": 0 到 1 之间的小数}
"""

RELEVANCE_PROMPT = """你是 RAG 系统的「相关性」评审员。给你【用户问题】和【助手答案】，
判断答案有没有切题回答这个问题。

只选一个 label：
- answered   正面回答了问题
- partial    只答了一部分，或者答得含糊、绕
- irrelevant 跑题；或者只是把问题换句话复述了一遍
- refused    明确表示"资料里没有相关内容"，没有给出实质答案

注意：
- 「答得对不对」**不是**你要管的（那是忠实性的事），你只管**切不切题**。
- 资料里确实没有答案时，refused 是**正确**的取舍，不是缺点——照样标 refused。
- 答案里既有实质内容又有免责声明时，按有没有实质内容判，别只看那句话。

只输出一个 JSON 对象，不要任何其他文字，不要用 markdown 代码块包裹：
{"label": "answered|partial|irrelevant|refused", "reason": "一句话理由"}
"""


# ---------------------------------------------------------------------------
# 读写
# ---------------------------------------------------------------------------


def load_jsonl(path: str) -> list:
    """读 JSONL。文件不存在是正常状态（还没 build 过），返回空表"""
    try:
        with open(path, encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]
    except FileNotFoundError:
        return []


def save_jsonl(records: list, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 解析判定输出（守卫写法对齐 ecom/knowledge/review/judge.py：非 dict 拦截、枚举兜底、数值 clamp）
# ---------------------------------------------------------------------------


def parse_faithfulness(raw: str) -> dict:
    """
    把模型输出解析成 {"claims": [...], "model_score": float}。

    解析失败**不抛异常**，返回空 claims——调用方据此得到 score=None（未定义），
    而不是一个假的 1.0。判定是辅助环节，不该把整轮评测带崩；
    但也不能假装"没判出问题"就是"没问题"。
    """
    text = strip_to_json(raw)
    try:
        data = json.loads(text) if text else None
    except (json.JSONDecodeError, TypeError):
        data = None

    if not isinstance(data, dict):
        return {"claims": [], "model_score": None, "parse_error": True}

    raw_claims = data.get("claims")
    claims = []
    if isinstance(raw_claims, list):
        for item in raw_claims:
            if not isinstance(item, dict):
                continue
            claim = str(item.get("claim") or "").strip()
            if not claim:
                continue
            label = str(item.get("label") or "").strip().lower()
            if label not in FAITHFULNESS_LABELS:
                # 认不出的标签按**最严**处理：宁可算它没被支持，
                # 也不要把一个来路不明的判定当成 supported 混进分子。
                label = "unsupported"
            claims.append({
                "claim": claim,
                "label": label,
                "evidence": str(item.get("evidence") or "").strip(),
            })

    try:
        model_score = min(max(float(data.get("score")), 0.0), 1.0)
    except (TypeError, ValueError):
        model_score = None

    return {"claims": claims, "model_score": model_score, "parse_error": False}


def faithfulness_score(claims: list) -> float | None:
    """
    忠实性 = 被支持的声明数 / 总声明数。

    **零声明返回 None，不是 1.0。** 答案里没有事实性声明（纯客套、纯拒答）时
    忠实性是**未定义**：算 1.0 会把拒答的答案捧成满分，算 0.0 又会把它踩到底，
    两种都会让总分静默失真。返回 None，由汇总层单独计数、排除出均值。

    contradicted 按**未支持**计：和资料打架比"资料没提"更严重，但两者都是
    "这一条没有依据"。
    """
    if not claims:
        return None
    supported = sum(1 for claim in claims if claim["label"] == "supported")
    return supported / len(claims)


def parse_relevance(raw: str) -> dict:
    """解析相关性判定。认不出的 label 落到 refused——不能默认成 answered"""
    text = strip_to_json(raw)
    try:
        data = json.loads(text) if text else None
    except (json.JSONDecodeError, TypeError):
        data = None

    if not isinstance(data, dict):
        return {"label": "refused", "reason": "", "parse_error": True}

    label = str(data.get("label") or "").strip().lower()
    if label not in RELEVANCE_LABELS:
        label = "refused"
    return {
        "label": label,
        "reason": str(data.get("reason") or "").strip(),
        "parse_error": False,
    }


def is_abstention(record: dict) -> bool:
    """
    这条记录算不算"承认了不知道"。

    **不能拿 ANSWER_FALLBACK 当判据。** 那是**服务故障**文案（ecom/knowledge/core/graph.py），
    不是"资料里没答案"文案——后者是模型按 system prompt 第 1 条自己说的，
    措辞由模型定，只能靠判定者的 refused 标签认。
    fallback 只作为兜底一起看（正常情况下恒为 False，见 _snapshot 的说明）。
    """
    if record.get("fallback"):
        return True
    return (record.get("relevance") or {}).get("label") == "refused"


# ---------------------------------------------------------------------------
# 判定（两次 LLM 调用）
# ---------------------------------------------------------------------------


def _format_contexts(contexts: list) -> str:
    if not contexts:
        return "（没有检索到任何资料）"
    lines = []
    for index, chunk in enumerate(contexts, start=1):
        lines.append(
            f"[{index}]（来源：{chunk.get('source', '未知')} 第 {chunk.get('chunk_id', '?')} 段）\n"
            f"{chunk.get('text', '')}"
        )
    return "\n\n".join(lines)


def _call(stage_name: str, system: str, human: str) -> tuple:
    """
    一次判定调用。计时 + token 记账都收在这一处——判定函数自己不去碰这两件事。

    返回 (正文, usage)。usage 里 in/out 两个数由调用方累加：
    `metric()` 只在 collect() 里生效，而这个模块是独立跑的命令行工具，
    外面没有 collector，所以真实成本必须自己记账（见 grade 末尾那行）。
    """
    with stage(stage_name):
        response = get_llm(TEMPERATURE, TIMEOUT).invoke(
            [("system", system), ("human", human)]
        )
    usage = getattr(response, "usage_metadata", None) or {}
    inputs = usage.get("input_tokens", 0)
    outputs = usage.get("output_tokens", 0)
    metric(f"{stage_name}_input_tokens", inputs)
    metric(f"{stage_name}_output_tokens", outputs)
    return response.content, {"in": inputs, "out": outputs}


def grade_faithfulness(record: dict) -> dict:
    """
    判定忠实性。返回 {"score", "claims", "model_score", "parse_error", "raw", "usage"}。
    score 可能为 None——**注意 None 有两个来源，别混为一谈**：
      - 答案真的没有事实性声明（拒答/纯客套）→ parsed["parse_error"] is False
      - 判定输出没解析出来 → parsed["parse_error"] is True，是**判定的故障**，不是答案的性质
    第一版把两者都塞进 claims=[]，汇总里也就分不出来，结果 4 条"零声明"里
    有 3 条其实是解析失败——它们被当成"拒答"静静地踢出了分母。所以 raw 要留下来。
    """
    prompt = (
        f"【检索到的资料】\n{_format_contexts(record.get('contexts') or [])}\n\n"
        f"【助手答案】\n{record.get('answer') or '（空）'}"
    )
    total = {"in": 0, "out": 0}
    parsed = None
    for _ in range(PARSE_ATTEMPTS):
        # 解析失败重试一次：temperature=0 也不保证同输入同输出（实测同一个答案
        # 判定重跑就正常了），把偶发当缺陷处理比调低标准强。usage 要**累加**，
        # 否则重试那次的 token 就漏记了。
        content, usage = _call("grade_faithfulness", FAITHFULNESS_PROMPT, prompt)
        total["in"] += usage["in"]
        total["out"] += usage["out"]
        parsed = parse_faithfulness(content)
        if not parsed["parse_error"]:
            break
    return {
        # 分数**在代码里**由标签算，不采信模型给的那个数——这样每一分都能在
        # --verbose 里追到是哪条声明没被支持，是个可审计的产物而不是黑箱。
        "score": faithfulness_score(parsed["claims"]),
        "claims": parsed["claims"],
        "model_score": parsed["model_score"],
        "parse_error": parsed["parse_error"],
        # 留着原始输出的头一段：解析失败的记录不留证据就查不下去，
        # 而"查不下去"正是它上次能悄悄混进"零声明"里的原因。
        "raw": content[:400] if parsed["parse_error"] else None,
        "usage": total,
    }


def grade_relevance(record: dict) -> dict:
    """判定相关性。返回 {"label", "reason", "parse_error", "raw", "usage"}"""
    prompt = f"【用户问题】\n{record.get('question')}\n\n【助手答案】\n{record.get('answer') or '（空）'}"
    total = {"in": 0, "out": 0}
    verdict = None
    for _ in range(PARSE_ATTEMPTS):
        content, usage = _call("grade_relevance", RELEVANCE_PROMPT, prompt)
        total["in"] += usage["in"]
        total["out"] += usage["out"]
        verdict = parse_relevance(content)
        if not verdict["parse_error"]:
            break
    verdict["raw"] = content[:400] if verdict["parse_error"] else None
    verdict["usage"] = total
    return verdict


# ---------------------------------------------------------------------------
# 汇总（纯函数，不打印——打印是 report 的事）
# ---------------------------------------------------------------------------


def summarize(records: list) -> dict:
    """
    把记录折成一组数。**分母全部写清楚**，因为这里最容易出的错是
    "把不该算进来的算进来了"——比如把零声明的答案当满分、把不可答题算进忠实性均值。
    """
    graded = [r for r in records if r.get("status") == "graded"]
    answerable = [r for r in graded if r.get("answerable")]
    unanswerable = [r for r in graded if not r.get("answerable")]

    scored = [r for r in answerable if (r.get("faithfulness") or {}).get("score") is not None]
    # score is None 的两条来路必须分开数，否则判定故障会被当成"答案在拒答"：
    #   - 真·零声明：答案里确实没有事实性声明（拒答/纯客套）——答案的性质
    #   - 解析失败：判定输出没解出来——**判定的故障**，是要去修的东西
    # 两者都进不了忠实性分母，但报告里必须能看出有几条是故障。
    parse_failed = [
        r for r in answerable
        if (r.get("faithfulness") or {}).get("score") is None
        and (r.get("faithfulness") or {}).get("parse_error")
    ]
    no_claims = [
        r for r in answerable
        if (r.get("faithfulness") or {}).get("score") is None
        and not (r.get("faithfulness") or {}).get("parse_error")
    ]

    total_claims = sum(len(r["faithfulness"]["claims"]) for r in scored)
    supported_claims = sum(
        1 for r in scored for claim in r["faithfulness"]["claims"]
        if claim["label"] == "supported"
    )
    # "与资料矛盾"单独拎出来：它比"资料没提"更严重，混在 unsupported 里看不出来
    contradicted = sum(
        1 for r in scored for claim in r["faithfulness"]["claims"]
        if claim["label"] == "contradicted"
    )

    relevance: dict = {}
    for record in answerable:
        label = (record.get("relevance") or {}).get("label", "refused")
        relevance[label] = relevance.get(label, 0) + 1

    abstained = [r for r in unanswerable if is_abstention(r)]
    # 「误答」：库里根本没有答案，系统却给了实质回答。这是本模块里**最贵的一类错误**
    wrong = [r for r in unanswerable if not is_abstention(r)]

    return {
        "graded": len(graded),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "faithful_scored": len(scored),
        "faithful_no_claims": len(no_claims),
        "faithful_parse_failed": len(parse_failed),
        "claim_total": total_claims,
        "claim_supported": supported_claims,
        "claim_contradicted": contradicted,
        "faithfulness": (supported_claims / total_claims) if total_claims else None,
        "relevance": relevance,
        "abstained": len(abstained),
        "wrong_answers": [r.get("question") for r in wrong],
        "mean_faithfulness": (
            sum(r["faithfulness"]["score"] for r in scored) / len(scored) if scored else None
        ),
    }


def agreement(records: list, dimension: str) -> tuple:
    """
    判定者与人的一致率。只统计两个标签都有的记录。

    返回 (一致条数, 可比条数)。分母是**人真的标过的**那些，不是全部记录——
    拿没标的当"不一致"会把一致率低估，当"一致"会高估，两种都是编数。
    """
    key = f"human_{dimension}"
    judge_key = dimension
    pairs = []
    for record in records:
        human = record.get(key)
        judge = record.get(judge_key)
        if not human or not judge:
            continue
        if dimension == "faithfulness":
            # 忠实性两边都是 float：判"一致"用 0.5 的粗档，因为 0.83 vs 0.86
            # 这种差异本来就没人分得清，硬比小数会把一致率压得毫无意义
            judge_value = judge.get("score")
            if judge_value is None:
                continue
            pairs.append(abs(judge_value - float(human)) < 0.5)
        else:
            pairs.append(judge.get("label") == human)
    return sum(pairs), len(pairs)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def _snapshot(question: str, qtype: str, answerable: bool, store, top_k: int) -> dict:
    """
    跑一次「检索 → 生成」，把结果冻成一条记录。

    生成失败**直接抛出去**，不吞：和 ecom/knowledge/utils/eval.py 里 embedding 失败必须
    暴露是同一条理由——一条含着兜底文案的快照看起来"跑完了"，实际整批作废。
    """
    contexts = retrieve(question, store, top_k=top_k)
    answer = generate_answer(question, contexts, [])  # 联网故意不传，见文件头
    return {
        "question": question,
        "type": qtype,
        "answerable": answerable,
        "contexts": [
            {
                "source": hit["source"],
                "chunk_id": hit["chunk_id"],
                "text": hit["text"],
                # score 是 RRF 融合分，只有 0.03 上下，用来排序可以、用来判"像不像"不行。
                # 判"库里有没有答案"要看 vector_score（向量路的余弦相似度）——
                # 两个都留下，免得以后又拿融合分去比相似度阈值。
                "score": hit.get("score"),
                "vector_score": hit.get("vector_score"),
            }
            for hit in contexts
        ],
        "answer": answer,
        # 正常情况下恒为 False：生成失败在上面就抛出去了，走不到这一行。
        # 留着是为了万一答出的正文恰好等于兜底文案时，弃答统计还能认出来。
        "fallback": answer == ANSWER_FALLBACK,
        "status": "built",
        "faithfulness": None,
        "relevance": None,
        "human_faithfulness": None,
        "human_relevance": None,
        "human_by": None,
        "human_at": None,
    }


def _top_score(record: dict) -> float:
    """
    这条记录检索到的**最像**的那一段，有多像——取向量路的余弦相似度。

    刻意不取 `score`：那是 RRF 融合分，量纲是 1/(K+rank)，恒在 0.03 附近，
    拿它比任何相似度阈值都恒为假。这个坑踩过一次（见 UNANSWERABLE_COSINE_REVIEW
    上面那段），所以这里只认 `vector_score`。

    没有向量分时（只召回 BM25 的 chunk）退回 0.0 而不是退回融合分——
    融合分会让"最像的一段其实只有 0.02"看起来像 0.02 就是个正常的相似度，
    而 0.0 至少诚实地表达"这条路没有给出相似度"。
    """
    scores = [c.get("vector_score") or 0.0 for c in (record.get("contexts") or [])]
    return max(scores) if scores else 0.0


def build(args) -> int:
    records = load_jsonl(args.out)
    if args.resume:
        if not records:
            print(f"没有可续跑的快照：{args.out}。直接 build 即可")
            return 1
        print(f"续跑：已有 {len(records)} 条，只补没跑过的题")
    else:
        records = []

    done = {record["question"] for record in records}

    cases = load_eval_set(args.set_path)
    if not cases:
        print("评测集是空的。请先跑 uv run python build_eval_set.py")
        return 1

    sample = stratified_sample(cases, args.n, seed=SEED)
    if not args.resume:
        types: dict = {}
        for case in sample:
            types[case.get("type", "全部")] = types.get(case.get("type", "全部"), 0) + 1
        print(f"从 {len(cases)} 题里分层抽了 {len(sample)} 题：{types}")

    unanswerable = load_jsonl(UNANSWERABLE_PATH)
    if not unanswerable:
        print(f"※ 没找到不可答题：{UNANSWERABLE_PATH}——弃答这一维将没有样本")
    else:
        print(f"不可答题 {len(unanswerable)} 条（跑完会按检索相似度排一遍，见末尾自检表）")

    todo = [
        (case["question"], case.get("type"), True)
        for case in sample if case["question"] not in done
    ] + [
        (item["question"], "unanswerable", False)
        for item in unanswerable if item["question"] not in done
    ]

    store = get_vector_store()
    try:
        if store.count() == 0:
            print("向量库是空的，请先运行：uv run python ingest.py")
            return 1
        next_index = len(records) + 1
        for index, (question, qtype, answerable) in enumerate(todo, 1):
            record = _snapshot(question, qtype, answerable, store, args.top_k)
            record["id"] = f"ans-{next_index:04d}"
            next_index += 1

            records.append(record)
            print(f"  [{index}/{len(todo)}] {question[:38]}… 答案 {len(record['answer'])} 字")
    finally:
        store.close()

    save_jsonl(records, args.out)
    print(f"\n已写出 {len(records)} 条快照 → {args.out}")
    _audit_unanswerable(records)
    print("下一步：uv run python answer_eval.py grade")
    return 0


def _audit_unanswerable(records: list) -> None:
    """
    把不可答题按"检索到的最像程度"从高到低排出来，给人过一眼。

    **这不是判据，是提示。** 我本来想让它自动判定"这题其实答得出来"，
    实测之后放弃了：真查不到的题余弦落在 0.53～0.64，而标错的题落在 0.66～0.71——
    两堆只差 0.02，还有重叠（"这题在讲什么"和"库里有没有这句话的答案"本来就是两件事，
    余弦只测得出前者）。拿一个手调阈值去切，切出来的确定性是假的。

    所以这里退一步做**分流**：超过提示线的排在最上面、打上标记，剩下的照样列出来。
    真正判定"答不答得出来"靠人去语料里 grep 一遍——就靠看这张表知道该去 grep 哪几行。
    上一版拿 RRF 融合分比 0.75 阈值，永远不触发，反而给人一种"自检过了"的错觉；
    宁可要一张会让人动手的表，也不要一个永远说 OK 的绿灯。
    """
    marked = [r for r in records if not r.get("answerable")]
    if not marked:
        return

    ranked = sorted(marked, key=_top_score, reverse=True)
    flagged = [r for r in ranked if _top_score(r) >= UNANSWERABLE_COSINE_REVIEW]

    print(f"\n【不可答题自检】{len(marked)} 条，按检索到的最像程度排序"
          f"（提示线 {UNANSWERABLE_COSINE_REVIEW}，仅供参考不是判据）")
    for record in ranked:
        top = _top_score(record)
        hit = max(record.get("contexts") or [{}],
                  key=lambda c: c.get("vector_score") or 0.0)
        flag = "※ 建议人工复核" if top >= UNANSWERABLE_COSINE_REVIEW else ""
        where = f"{hit.get('source')}#{hit.get('chunk_id')}" if hit else "-"
        print(f"  {top:.3f}  {record['question'][:30]:32} 最像 {where}  {flag}")

    if flagged:
        print(f"  ※ 有 {len(flagged)} 条疑似语料里其实答得出来——去 {UNANSWERABLE_PATH} "
              f"把它们换成真查不到的题，或者在语料里 grep 确认")
    else:
        print("  ※ 没有越过提示线的。但这只说明没有**明显**标错的，不等于都验证过了")


# ---------------------------------------------------------------------------
# grade
# ---------------------------------------------------------------------------


def grade(args) -> int:
    records = load_jsonl(args.path)
    if not records:
        print(f"没有快照：{args.path}。请先跑 answer_eval.py build")
        return 1

    if args.only_pending:
        todo = [r for r in records if r.get("status") != "graded"]
    else:
        todo = list(records)
    if not todo:
        print("没有需要判定的条目。")
        return 0

    print(f"待判定 {len(todo)} 条（每条 2 次调用：忠实性 + 相关性）")
    tokens_in = tokens_out = 0
    failures = 0
    for index, record in enumerate(todo, 1):
        try:
            record["faithfulness"] = grade_faithfulness(record)
            record["relevance"] = grade_relevance(record)
            record["status"] = "graded"
        except Exception as exc:  # noqa: BLE001 —— 一条判废不该让整批停下
            failures += 1
            print(f"  ※ 判定失败，跳过这一条：{exc}")
            continue

        for part in ("faithfulness", "relevance"):
            usage = record[part].pop("usage", {})
            tokens_in += usage.get("in", 0)
            tokens_out += usage.get("out", 0)
        score = record["faithfulness"]["score"]
        if score is None:
            # 这两行长得一样但含义相反，进度里就得分开——否则"解析失败"
            # 混在一串"未定义"里，跑完没人会注意到判定其实坏了几条。
            shown = ("※ 解析失败" if record["faithfulness"].get("parse_error")
                     else "未定义(零声明)")
        else:
            shown = f"{score:.0%}"
        print(f"  [{index}/{len(todo)}] 忠实性 {shown:<12} "
              f"相关性 {record['relevance']['label']}")
        save_jsonl(records, args.path)  # 每判一条就落盘：判得慢，中途关掉不该白判

    save_jsonl(records, args.path)
    if failures:
        print(f"\n※ {failures} 条判定失败，仍是未判定状态，可重跑 grade --only-pending")
    print(f"\n已判定 {len(todo) - failures} 条 → {args.path}")
    print(_cost_line(tokens_in, tokens_out, len(todo) - failures))
    print("下一步：uv run python answer_eval.py report")
    return 0


def _cost_line(tokens_in: int, tokens_out: int, count: int) -> str:
    if not count:
        return "（没有成功的判定，无法算成本）"
    cost = (tokens_in / 1e6) * QWEN_PRICE_INPUT + (tokens_out / 1e6) * QWEN_PRICE_OUTPUT
    return (f"判定成本约 ￥{cost:.4f}（{tokens_in} in + {tokens_out} out tok，"
            f"单价见 QWEN_PRICE_*，只算判定、不含生成）")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _relevance_lines(relevance: dict, total: int) -> list:
    labels = [
        ("answered", "答上了"),
        ("partial", "只答了一部分"),
        ("irrelevant", "跑题/复述问题"),
        ("refused", "说资料里没有 ← 可答题却拒答，是检索/生成没撑住"),
    ]
    return [
        f"  {key:<10} {relevance.get(key, 0):>3}  {note}"
        for key, note in labels if relevance.get(key)
    ]


def report(args) -> int:
    records = load_jsonl(args.path)
    if not records:
        print(f"没有快照：{args.path}。请先 build 再 grade")
        return 1

    stats = summarize(records)
    if not stats["graded"]:
        print("还没有判定过的条目，无法出数。先跑 uv run python answer_eval.py grade")
        return 1

    print("=" * 66)
    print(f"答案质量评测：{args.path}")
    print("※ 判定者与被测答案是同一个模型（qwen-plus）——未经人工校准前，"
          "下面每一行都是模型自评")
    print(f"  已判定 {stats['graded']} 条：可答 {stats['answerable']} 条 + "
          f"不可答 {stats['unanswerable']} 条")
    print("=" * 66)

    # 误答顶在最前面：它是最贵的一类错误——答案读起来很顺，整个是编的，
    # 而用户没有任何办法察觉。整个项目的"宁可暴露也不掩盖"就是为了防它。
    if stats["unanswerable"]:
        print(f"\n【最贵的一类错误：误答】不可答的 {stats['unanswerable']} 条里，"
              f"有 {len(stats['wrong_answers'])} 条给出了实质回答")
        for question in stats["wrong_answers"]:
            print(f"    × {question}")
        print("  （本该说「资料里没有」，却答了——用户看不出这是编的）")
        half = wilson_half_width(stats["abstained"] / stats["unanswerable"],
                                 stats["unanswerable"])
        print(f"  弃答正确率 = {stats['abstained']}/{stats['unanswerable']} = "
              f"{stats['abstained'] / stats['unanswerable']:.1%}  "
              f"±{half:.1%}（95% 置信区间半宽）")
        # 满分时要额外说一句：威尔逊区间在 p=1 处会塌成 0，那个 ±0.0% 不是
        # "算得很准"，而是"这个公式在边界上给不出信息"。真正的不确定性此刻
        # 挪到了别处——这套不可答题是手写的，它的可信度就是整个数的上限。
        if stats["abstained"] == stats["unanswerable"]:
            print("  ※ 满分不是「测得准」，是这个区间在边界上塌成了 0。"
                  "真正的误差在题本身：这 10 条不可答题是手写的、没有独立来源，"
                  "标错一条这个数就掉 10 个点。别把 100% 当成系统的属性。")

    print(f"\n【忠实性】只统计可答题里**有事实性声明**的 {stats['faithful_scored']} 条")
    if stats["faithfulness"] is not None:
        print(f"  被支持的声明 {stats['claim_supported']}/{stats['claim_total']} = "
              f"{stats['faithfulness']:.1%}")
        print(f"  逐条取平均 = {stats['mean_faithfulness']:.1%}"
              f"（与上面的合并不完全相等：合并按声明数加权，这里每条一票）")
        if stats["claim_contradicted"]:
            print(f"  ※ 其中有 {stats['claim_contradicted']} 条与资料**矛盾**"
                  f"——比「资料没提」更严重，已单独列出")
    else:
        print("  （没有任何可统计的声明）")
    if stats["faithful_no_claims"]:
        print(f"  已排除 {stats['faithful_no_claims']} 条零声明的答案（拒答/纯客套）"
              f"——算 1.0 会虚高，算 0.0 会虚低，两者都是编数")
    if stats["faithful_parse_failed"]:
        # 这一条要显眼：它说明上面的忠实性是**在这几条缺席的情况下**算出来的，
        # 不是"这几条没声明"。上一版把它并进"零声明"里，等于把判定的故障
        # 说成了答案的性质——报告读起来一切正常，坏掉的那部分却看不见。
        print(f"  ※ 另有 {stats['faithful_parse_failed']} 条**判定输出没解析出来**"
              f"（重试 {PARSE_ATTEMPTS} 次仍失败）——它们既不进分子也不进分母，"
              f"也就是说忠实性是在这几条缺席时算出来的，不是它们没有声明")

    print(f"\n【相关性】可答题 {stats['answerable']} 条")
    for line in _relevance_lines(stats["relevance"], stats["answerable"]):
        print(line)

    covered = stats["answerable"] + stats["unanswerable"]
    calib_f, calib_f_total = agreement(records, "faithfulness")
    calib_r, calib_r_total = agreement(records, "relevance")
    print("\n【人工校准】")
    if not calib_r_total:
        print("  还没做过人工校准——跑 uv run python answer_eval.py calibrate")
        print("  ※ 没有这个数，上面的分数没有任何外部参照，只能当模型自评看")
    else:
        rate = calib_r / calib_r_total
        print(f"  相关性  判定者与人一致 {calib_r}/{calib_r_total} = {rate:.1%}  "
              f"±{wilson_half_width(rate, calib_r_total):.1%}")
        if calib_f_total:
            rate_f = calib_f / calib_f_total
            print(f"  忠实性  判定者与人一致 {calib_f}/{calib_f_total} = {rate_f:.1%}  "
                  f"±{wilson_half_width(rate_f, calib_f_total):.1%}")
        print(f"  ※ 校准样本只有 {calib_r_total} 条（总 {covered} 条），"
              f"它只能挡住「判得离谱」，挡不住几个百分点的偏差")

    if args.verbose:
        print("\n" + "-" * 56)
        print("逐条明细（只列有问题的）：")
        for record in records:
            faith = record.get("faithfulness") or {}
            bad = [c for c in (faith.get("claims") or []) if c["label"] != "supported"]
            label = (record.get("relevance") or {}).get("label")
            if not bad and label in ("answered", "refused", None):
                continue
            print(f"\n[{record.get('id')}] {record.get('question')}")
            print(f"  答案：{(record.get('answer') or '')[:150]}")
            print(f"  相关性：{label}")
            if faith.get("score") is not None:
                print(f"  忠实性：{faith['score']:.0%}")
            for claim in bad:
                print(f"    × [{claim['label']}] {claim['claim'][:70]}")
    return 0


# ---------------------------------------------------------------------------
# calibrate：人工盲判
# ---------------------------------------------------------------------------


def calibrate(args) -> int:
    """
    人工抽检，算判定者与人的一致率。

    **盲判**：先只给（问题, 资料, 答案），人判完才揭晓判定者的结论。
    理由和 gold_set.py 的设计一完全一样——先看到判定者说什么，人会被锚定，
    一致率就虚高了，那叫"复核"，不叫"校准"。
    """
    records = load_jsonl(args.path)
    graded = [r for r in records if r.get("status") == "graded"]
    if not graded:
        print("还没有判定过的条目。先跑 uv run python answer_eval.py grade")
        return 1

    todo = [r for r in graded if r.get("human_relevance") is None]
    if args.n:
        todo = todo[:args.n]
    if not todo:
        print("已经全部校准过了。")
        return 0

    who = _labeler()
    print(f"待校准 {len(todo)} 条，标注人：{who}（GOLD_LABELER 环境变量可改）")
    print("**盲判**：先给你问题/资料/答案，你判完才揭晓判定者的结论——")
    print("先看到判定者说什么就会被锚定，一致率会虚高。\n")

    by_id = {record["id"]: record for record in records}
    for index, record in enumerate(todo, 1):
        print("=" * 66)
        print(f"[{index}/{len(todo)}] {record['question']}")
        print("-" * 66)
        for position, chunk in enumerate(record.get("contexts") or [], 1):
            text = (chunk.get("text") or "").replace("\n", " ")[:110]
            print(f"  [{position}] {chunk['source']}#{chunk['chunk_id']}  {text}")
        print("-" * 66)
        print(f"答案：{record['answer']}")
        print("-" * 66)
        print("  相关性   a=答上了  p=只答了一部分  i=跑题  r=说资料里没有  s=跳过  q=存盘退出")
        relevance = input("  相关性 > ").strip().lower()
        if relevance in ("q", "quit"):
            break
        if relevance in ("s", "skip", ""):
            continue
        relevance = {"a": "answered", "p": "partial", "i": "irrelevant",
                     "r": "refused"}.get(relevance, relevance)
        if relevance not in RELEVANCE_LABELS:
            print("  不认识这个输入，这条留作未校准。")
            continue

        print("  忠实性   n=答案里没有资料支撑不了的内容  y=有（编造/与资料矛盾）")
        faithful = input("  忠实性 > ").strip().lower()

        target = by_id[record["id"]]
        target.update(
            human_relevance=relevance,
            # 忠实性在人工这边只能给粗档（1.0 / 0.0）：让一个人逐条去数声明
            # 不现实，而一致率本来就只在 0.5 这个粒度上有意义（见 agreement）。
            human_faithfulness=1.0 if faithful in ("n", "no", "1") else 0.0,
            human_by=who,
            human_at=_now(),
        )

        judge_faith = (record.get("faithfulness") or {}).get("score")
        judge_rel = (record.get("relevance") or {}).get("label")
        shown = "未定义" if judge_faith is None else f"{judge_faith:.0%}"
        mark = "一致" if judge_rel == relevance else "**不一致**"
        print(f"  → 判定者说：相关性 {judge_rel}，忠实性 {shown}   ({mark})")

        save_jsonl(records, args.path)  # 每标一条就落盘

    save_jsonl(records, args.path)
    done_r, total_r = agreement(records, "relevance")
    print(f"\n已存盘。相关性一致 {done_r}/{total_r}"
          + (f" = {done_r / total_r:.1%}" if total_r else ""))
    print("下一步：uv run python answer_eval.py report")
    return 0


# ---------------------------------------------------------------------------
# selftest：金丝雀，验判定者本身
# ---------------------------------------------------------------------------


SELFTEST_CASES = [
    {
        "name": "忠实：答案完全来自资料",
        "question": "退货要在几天内提出？",
        "contexts": [{
            "source": "售后政策.md", "chunk_id": 1,
            "text": "自签收之日起 7 天内可申请无理由退货，退货运费由买家承担。",
        }],
        "answer": "自签收之日起 7 天内可以申请无理由退货，退货运费由买家承担。[1]",
        "expect_faithful": True,
        "expect_relevance": "answered",
    },
    {
        "name": "编造：数字资料里根本没有",
        "question": "退货要在几天内提出？",
        "contexts": [{
            "source": "售后政策.md", "chunk_id": 1,
            "text": "自签收之日起 7 天内可申请无理由退货，退货运费由买家承担。",
        }],
        "answer": "自签收之日起 30 天内可以申请无理由退货，退货运费由卖家承担。[1]",
        "expect_faithful": False,
        "expect_relevance": "answered",
    },
    {
        "name": "与资料矛盾：金额对不上",
        "question": "会员积分怎么抵扣？",
        "contexts": [{
            "source": "会员与积分.md", "chunk_id": 2,
            "text": "100 积分可抵扣 1 元，单笔订单最多抵扣订单金额的 50%。",
        }],
        "answer": "100 积分可以抵扣 10 元，单笔订单最多抵扣订单金额的 50%。[1]",
        "expect_faithful": False,
        "expect_relevance": "answered",
    },
    {
        "name": "跑题：答的是另一个话题",
        "question": "退货要在几天内提出？",
        "contexts": [{
            "source": "售后政策.md", "chunk_id": 1,
            "text": "自签收之日起 7 天内可申请无理由退货，退货运费由买家承担。",
        }],
        "answer": "发票可以在订单完成后 30 天内申请开具，支持电子普票和专用发票。[1]",
        "expect_faithful": None,
        "expect_relevance": "irrelevant",
    },
    {
        "name": "明确拒答：资料里没有",
        "question": "你们公司的客服电话是多少？",
        "contexts": [{
            "source": "售后政策.md", "chunk_id": 3,
            "text": "退货申请提交后，审核一般在 1 个工作日内完成。",
        }],
        "answer": "提供的资料中没有相关内容，无法回答这个问题。",
        # 这一条钉的是"零声明"：拒答句**不是**事实性声明，claims 必须为空，
        # 于是忠实性为 None（未定义）而不是 0%。判成 0% 会把忠实性均值往下拽
        # ——第一次跑金丝雀时它就真的这么判了，所以这里必须钉住。
        "expect_zero_claims": True,
        "expect_relevance": "refused",
    },
]


def selftest(args) -> int:
    """
    金丝雀：拿几组**已知正确答案**的（资料, 答案）喂给判定者，看它抓不抓得住。

    这是本模块最重要的一道防线，因为它**不需要人工标注**就能回答
    「这个判定者可信吗」。抓不住编造的那两条，后面所有忠实性分数都没有意义。
    """
    print("判定者自检（金丝雀）——它抓不住下面这些，后面所有分数都不用看了")
    print("=" * 66)
    passed = 0
    for case in SELFTEST_CASES:
        record = {"question": case["question"], "contexts": case["contexts"],
                  "answer": case["answer"]}
        faith = grade_faithfulness(record)
        relevance = grade_relevance(record)

        ok = True
        if case["expect_relevance"] != relevance["label"]:
            ok = False
        expected_faithful = case.get("expect_faithful")
        if case.get("expect_zero_claims"):
            # 拒答/纯客套：必须一条声明都拆不出来，忠实性**未定义**
            if faith["claims"] or faith["score"] is not None:
                ok = False
        elif expected_faithful is True and faith["score"] != 1.0:
            ok = False
        elif expected_faithful is False and not (faith["score"] is not None
                                                 and faith["score"] < 1.0):
            ok = False
        passed += ok

        if case.get("expect_zero_claims"):
            score = "未定义" if faith["score"] is None else f"{faith['score']:.0%}"
            score += f"（{len(faith['claims'])} 条声明）"
        else:
            score = "未定义" if faith["score"] is None else f"{faith['score']:.0%}"
        print(f"  {'√' if ok else '×'} {case['name']:<20} "
              f"忠实性 {score:<14} 相关性 {relevance['label']}")
        if not ok:
            print(f"      期望 忠实性={expected_faithful} "
                  f"零声明={bool(case.get('expect_zero_claims'))} "
                  f"相关性={case['expect_relevance']}")
        bad = [c for c in faith["claims"] if c["label"] != "supported"]
        if bad:
            print(f"      它标出的可疑声明：{[c['label'] for c in bad]}")

    print("=" * 66)
    total = len(SELFTEST_CASES)
    print(f"通过 {passed}/{total}")
    if passed < total:
        print("※ 判定者抓不住已知的错误答案——先修提示词，再看它给的任何分数")
        return 1
    print("判定者能抓住编造/矛盾/跑题——它的分数才有参考价值。")
    print("注意这**不**等于它的分是对的，只是「至少不是瞎判」；校准还得靠 calibrate。")
    return 0


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _labeler() -> str:
    """标注人是谁：优先环境变量，其次 git 配置——至少要能追到人（与 gold_set 一致）"""
    who = os.getenv("GOLD_LABELER")
    if who:
        return who
    try:
        result = subprocess.run(
            ["git", "config", "user.name"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return result.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 —— 拿不到就不署，不该让标注流程因此中断
        return "unknown"


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="答案质量评测：忠实性 / 相关性 / 弃答")
    sub = parser.add_subparsers(dest="command")

    p_build = sub.add_parser("build", help="跑检索+生成，写出答案快照")
    p_build.add_argument("--n", type=int, default=30, help="可答题抽多少条（默认 30）")
    p_build.add_argument("--top-k", type=int, default=TOP_K)
    p_build.add_argument("--set-path", default=None, help="源评测集，默认 data/knowledge/eval_set.jsonl")
    p_build.add_argument("--out", default=ANSWER_EVAL_PATH)
    p_build.add_argument("--resume", action="store_true", help="在已有快照上续跑，只补没跑过的题")

    p_grade = sub.add_parser("grade", help="判定忠实性与相关性")
    p_grade.add_argument("--path", default=ANSWER_EVAL_PATH)
    p_grade.add_argument("--only-pending", action="store_true",
                         help="只判还没判过的（默认全部重判）")

    p_report = sub.add_parser("report", help="出数")
    p_report.add_argument("--path", default=ANSWER_EVAL_PATH)
    p_report.add_argument("--verbose", action="store_true", help="逐条列出有问题的")

    p_calib = sub.add_parser("calibrate", help="人工盲判，算判定者与人的一致率")
    p_calib.add_argument("--path", default=ANSWER_EVAL_PATH)
    p_calib.add_argument("--n", type=int, default=20, help="抽多少条（默认 20）")

    sub.add_parser("selftest", help="金丝雀：验判定者本身，不花生成的钱")

    args = parser.parse_args()
    if args.command == "build":
        return build(args)
    if args.command == "grade":
        return grade(args)
    if args.command == "report":
        return report(args)
    if args.command == "calibrate":
        return calibrate(args)
    if args.command == "selftest":
        return selftest(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
