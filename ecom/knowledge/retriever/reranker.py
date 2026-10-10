"""
Rerank重排序模块
功能：对检索结果重排序，取Top3最相关的

用硅基流动的 BGE-Reranker。三级降级，任何一级失败都不会让整条链路中断：
  1. 硅基流动 rerank 接口
  2. 本地 CrossEncoder（需自己装 sentence-transformers）
  3. 按向量检索分数截断（等于不做重排）
"""
import requests

from ecom.knowledge.settings import (
    RERANK_MODEL,
    RERANK_TIMEOUT,
    RERANK_TOP_K,
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
)
from ecom.knowledge.utils.logger import setup_logger
from ecom.knowledge.utils.timing import stage

logger = setup_logger(__name__)

RERANK_URL = f"{SILICONFLOW_BASE_URL}/rerank"

_cross_encoder = None
_cross_encoder_checked = False


def _doc_text(item) -> str:
    """兼容 list[str] 和 list[dict] 两种入参"""
    return item.get("text", "") if isinstance(item, dict) else str(item)


def _doc_score(item) -> float:
    """取已有的向量检索分数，用于最后一级降级"""
    return item.get("score", 0.0) if isinstance(item, dict) else 0.0


def _attach_score(item, score: float):
    """把 rerank 分数写回 dict，方便调用方查看"""
    if isinstance(item, dict):
        item["rerank_score"] = score
    return item


def _rerank_by_api(question: str, texts: list):
    """调硅基流动 rerank 接口，返回 [{"index","score"}]；不可用时返回 None"""
    if not SILICONFLOW_API_KEY:
        logger.info("未配置 SILICONFLOW_API_KEY，跳过接口 rerank")
        return None

    try:
        response = requests.post(
            RERANK_URL,
            headers={
                "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": RERANK_MODEL,
                "query": question,
                "documents": texts,
                "top_n": len(texts),
                "return_documents": False,
            },
            timeout=RERANK_TIMEOUT,
        )
    except Exception as exc:
        logger.warning("rerank 接口调用失败：%s", exc)
        return None

    if response.status_code != 200:
        # 账号未实名认证会返回 403
        logger.warning("rerank 接口返回 %s：%s", response.status_code, response.text[:200])
        return None

    try:
        return [
            {"index": item["index"], "score": item["relevance_score"]}
            for item in response.json().get("results", [])
        ]
    except Exception as exc:
        logger.warning("rerank 响应解析失败：%s", exc)
        return None


def _rerank_by_local_model(question: str, texts: list):
    """本地 CrossEncoder 兜底；没装 sentence-transformers 就返回 None"""
    global _cross_encoder, _cross_encoder_checked
    if not _cross_encoder_checked:
        _cross_encoder_checked = True
        try:
            from sentence_transformers import CrossEncoder

            _cross_encoder = CrossEncoder(RERANK_MODEL)
            logger.info("已加载本地 CrossEncoder：%s", RERANK_MODEL)
        except Exception as exc:
            logger.info("本地 CrossEncoder 不可用（%s），跳过这一级降级", exc)
            _cross_encoder = None

    if _cross_encoder is None:
        return None

    try:
        scores = _cross_encoder.predict([[question, text] for text in texts])
        return [{"index": index, "score": float(score)} for index, score in enumerate(scores)]
    except Exception as exc:
        logger.warning("本地 CrossEncoder 打分失败：%s", exc)
        return None


def rerank(question: str, contexts: list, top_k: int = RERANK_TOP_K) -> list:
    """
    对检索结果重排序，取 Top-k

    contexts 可以是 [{"text","source",...}, ...] 或纯字符串列表
    返回：重排后的 Top-k（dict 会带上 rerank_score 字段）
    """
    if not contexts:
        return []

    # 整个函数体算一个阶段（含三级降级的全部耗时）。画像里关心的是
    # "rerank 这一步吃掉多少"——降级到第几级是原因，不是要分开计量的事。
    with stage("rerank"):
        texts = [_doc_text(item) for item in contexts]

        ranked = _rerank_by_api(question, texts)
        if ranked is None:
            ranked = _rerank_by_local_model(question, texts)
        if ranked is None:
            logger.warning("rerank 全部降级失败，退化为按向量分数截断")
            ranked = [{"index": i, "score": _doc_score(item)} for i, item in enumerate(contexts)]

        # index 是外部给的，不能直接拿去下标：
        #   越界正数 -> IndexError，把整条问答链路打断
        #   负数     -> **静默**取到尾部 chunk，结果看着正常但是错的（最隐蔽）
        # bool 是 int 的子类，JSON 里的 true 能骗过 isinstance(index, int)，一并排除。
        valid = [
            item for item in ranked
            if isinstance(item.get("index"), int)
            and not isinstance(item.get("index"), bool)
            and 0 <= item["index"] < len(contexts)
        ]
        if len(valid) != len(ranked):
            logger.warning("rerank 返回 %d 条非法 index，已丢弃", len(ranked) - len(valid))
        if not valid:
            logger.warning("rerank 结果全部非法，退化为按向量分数截断")
            valid = [
                {"index": index, "score": _doc_score(item)}
                for index, item in enumerate(contexts)
            ]

        valid.sort(key=lambda item: item["score"], reverse=True)
        top = valid[:top_k]
        logger.info("rerank 完成：%d → %d", len(contexts), len(top))

        return [_attach_score(contexts[item["index"]], item["score"]) for item in top]


if __name__ == "__main__":
    contexts = ["退货政策：7天无理由", "安装指南：第一步", "运费：质量问题商家承担"]
    result = rerank("退货政策是什么", contexts)
    print(f"重排序后Top{len(result)}: {result}")
