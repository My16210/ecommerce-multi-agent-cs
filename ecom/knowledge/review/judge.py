"""
提问价值判定
功能：让大模型判断用户这次的提问是否值得沉淀进知识库

判定发生在**一轮问答结束之后**（调用点在 main.py），不进 LangGraph 图：
图节点会被 app.invoke() 同步等待，判定多一次 LLM 往返，用户就得晚几秒才看到答案；
LangGraph 里也没有真正的 fire-and-forget 尾节点。

判定结果只进待审核队列，**绝不写向量库**——详见 queue.py 里"知识库自污染"那段说明。
"""
import json

# 兜底文案从 graph.py 引进来，不在这里另写一份字面量：
# 两处一旦不同步，生成失败的那一轮就会被误判成"知识库缺口"进队列。
# graph.py 不反向依赖 ecom.knowledge.review，所以这个方向不会成环。
from ecom.knowledge.core.graph import ANSWER_FALLBACK
from ecom.knowledge.review.queue import enqueue
from ecom.knowledge.settings import JUDGE_TIMEOUT
from ecom.knowledge.utils.json_utils import strip_to_json
from ecom.knowledge.utils.llm import get_llm
from ecom.knowledge.utils.logger import setup_logger
from ecom.knowledge.utils.timing import metric, stage

logger = setup_logger(__name__)

# 判定标准分两条：补缺口（kb_gap）和沉淀 FAQ（reusable_faq）
SYSTEM_PROMPT = """你是知识库运营助手。你的任务：判断用户这次的提问是否值得沉淀进知识库，供以后的人复用。

你会拿到：用户问题、助手给出的答案、知识库检索的命中情况、联网搜索的命中情况。

判为「值得」的两种情形：
- kb_gap（知识库缺口）：知识库命中很少或相似度很低，答案主要靠联网搜索撑起来。
  这说明知识库里缺这块内容，沉淀价值最高。
- reusable_faq（可复用领域问答）：问题带明确的领域术语，答案是通用的、
  后来的人会重复问到，像一条 FAQ。

判为「不值得」（category 填 neither）的情形：
- 寒暄、闲聊、一次性的操作请求（如"帮我把这句话翻译一下"）
- 与知识库领域无关的通用常识（如"今天天气怎么样"）
- 含个人隐私、或答案有时效性的内容
- 问题本身过于模糊，答案没有复用价值

只输出一个 JSON 对象，不要任何其他文字，不要用 markdown 代码块包裹：
{"worthy": true 或 false, "category": "kb_gap" 或 "reusable_faq" 或 "neither", "score": 0 到 1 之间的小数, "reason": "一句话理由"}
"""

VALID_CATEGORIES = ("kb_gap", "reusable_faq", "neither")

# temperature=0：判定结果要稳定、可复现，同一份现场每次判成一样
TEMPERATURE = 0


def _format_signals(question: str, result: dict) -> str:
    """把判定要用到的现场信号拼成一段文本"""
    contexts = [c for c in (result.get("contexts") or []) if isinstance(c, dict)]
    # 混合检索开启后 contexts 的 score 是 RRF 融合分（量级 1/61），**不是相似度**。
    # 判定要的是"知识库答得怎么样"这个信号，所以优先取向量的原始余弦分；
    # 只有纯向量检索、或该条没被向量路召回时，才退回 score。
    scores = [c.get("vector_score", c.get("score", 0.0)) for c in contexts]
    top_score = max(scores) if scores else 0.0

    return (
        f"【用户问题】\n{question}\n\n"
        f"【助手答案】\n{result.get('answer') or '（无）'}\n\n"
        f"【知识库检索】命中 {len(contexts)} 条，最高相似度 {top_score:.3f}\n"
        f"【联网搜索】命中 {len(result.get('web_contexts') or [])} 条"
    )


def _parse(raw: str, question: str) -> dict:
    """
    容错解析模型的输出。

    不依赖 response_format 严格 JSON（compatible-mode 是否支持没验证过），
    所以这里要处理：被 ``` 围栏包住、前后有多余文字、字段类型不对、直接不是 JSON。
    任何解析不了的输出都按「不值得」处理，宁可漏收也不能污染队列。

    剥围栏、找 JSON 那段和出题脚本共用 ecom/knowledge/utils/json_utils.py——两边各写一份
    的话，脏输出处理得不干净的症状都是**静默丢数据**，谁也不报错。
    """
    text = strip_to_json(raw)

    try:
        data = json.loads(text) if text else None
    except (json.JSONDecodeError, TypeError):
        data = None

    # 除了"根本不是 JSON"，还要挡一种情况：模型给的是一个数组或字符串。
    # 后面要按 dict 取字段，不挡就会在 data.get 上抛 AttributeError，
    # 从"这条判成不值得"升级成"整轮判定失败"。
    if not isinstance(data, dict):
        logger.warning("判定输出不是合法 JSON 对象，按「不值得」处理：%r", (raw or "")[:200])
        return {
            "worthy": False,
            "category": "neither",
            "score": 0.0,
            "reason": "模型输出无法解析为 JSON",
        }

    category = str(data.get("category") or "neither").strip()
    if category not in VALID_CATEGORIES:
        category = "neither"

    try:
        score = min(max(float(data.get("score", 0.0)), 0.0), 1.0)
    except (TypeError, ValueError):
        score = 0.0

    # 模型可能把布尔写成字符串 "false"，直接 bool("false") 会得到 True
    worthy_raw = data.get("worthy")
    if isinstance(worthy_raw, str):
        worthy = worthy_raw.strip().lower() in ("true", "yes", "1")
    else:
        worthy = bool(worthy_raw)

    return {
        # category 和 worthy 冲突时以 category 为准
        "worthy": worthy and category != "neither",
        "category": category,
        "score": score,
        "reason": str(data.get("reason") or "").strip(),
    }


def judge(question: str, result: dict) -> dict:
    """判定单次提问，返回 {"worthy", "category", "score", "reason"}"""
    with stage("judge"):
        response = get_llm(TEMPERATURE, JUDGE_TIMEOUT).invoke(
            [("system", SYSTEM_PROMPT), ("human", _format_signals(question, result))]
        )
    usage = getattr(response, "usage_metadata", None) or {}
    metric("judge_input_tokens", usage.get("input_tokens", 0))
    metric("judge_output_tokens", usage.get("output_tokens", 0))
    verdict = _parse(response.content, question)
    logger.info(
        "提问判定：worthy=%s category=%s score=%.2f reason=%s",
        verdict["worthy"], verdict["category"], verdict["score"], verdict["reason"],
    )
    return verdict


def record_question(question: str, result: dict) -> dict:
    """
    一轮问答的收尾：判定这次提问值不值得沉淀，值得就写进待审核队列。

    成功返回 {"verdict": {...}, "row_id": int, "is_new": bool}；
    **跳过或失败一律返回 None**，调用方只需判空。

    这里把守卫一并吃掉（判空 + 整段兜底），而不是留给调用方：
    main.py 和 app.py 都要走这一步，各写一遍 try/except 和 ANSWER_FALLBACK
    判断的话，两边日后必然漂移——UI 那边漏判一次，服务抖动就会被当成
    知识库缺口写进队列，而队列里的脏数据是会被人 approve 进知识库的。
    所以守卫只能有一份。

    **不打印**：命令行要印中文提示、Streamlit 要 st.success，展示归调用方。
    """
    if result.get("answer") == ANSWER_FALLBACK:
        # 这一轮生成失败了。判定看到的现场是"知识库没命中、答案也没内容"，
        # 很容易被判成 kb_gap，把一次服务故障当作知识库缺口写进队列。
        logger.info("本轮答案生成失败，跳过提问判定")
        return None

    try:
        verdict = judge(question, result)
        row_id, is_new = enqueue(question, result, verdict)
    except Exception as exc:
        # 判定只是锦上添花，绝不能因为它失败就让用户拿不到答案。
        logger.warning("提问判定失败，已跳过：%s", exc)
        return None

    return {"verdict": verdict, "row_id": row_id, "is_new": is_new}


if __name__ == "__main__":
    # 造一份"知识库没答好、主要靠联网"的现场，应该被判为 kb_gap
    demo = {
        "answer": "七天无理由退货的期限，自消费者签收商品之日起计算。",
        "contexts": [{"score": 0.031, "source": "售后政策.md", "chunk_id": 3}],
        "web_contexts": [{"url": "https://example.com"}, {"url": "https://example.org"}],
        "sources": ["售后政策.md", "https://example.com"],
    }
    print(json.dumps(judge("七天无理由退货的七天从哪一天开始算？", demo),
                     ensure_ascii=False, indent=2))
