"""
状态定义
功能：定义RAG工作流中数据怎么在节点间传递

并行两路各写自己的字段（contexts / web_contexts），
每个字段在同一超步只有一个写入者，所以用默认的覆盖语义就够，不需要 reducer。
"""
from typing import TypedDict


class RAGState(TypedDict):
    question: str             # 用户问题，由外部 invoke 传入
    # 知识库检索+rerank结果。混合检索开启时 score 是 RRF 融合分（不是相似度），
    # 两路的原始分和名次各自带在 vector_score / bm25_score / vector_rank / bm25_rank 上，
    # 没被某一路召回的 chunk 就没有那一路的字段（见 ecom/knowledge/retriever/retriever.py）。
    contexts: list[dict]      # {text, source, chunk_id, score, rerank_score, ...融合明细}
    web_contexts: list[dict]  # 联网搜索结果 {text, title, url, source}
    answer: str               # 生成的答案
    sources: list[str]        # 引用来源（去重）
    chat_history: list[dict]  # 多轮对话历史 [{"role","content"}]，由 answer_node 追加后整体写回
