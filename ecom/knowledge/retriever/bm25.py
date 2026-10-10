"""
BM25 关键词检索（混合检索的第二路）

为什么手写而不引 rank_bm25
--------------------------
BM25 的公式只有四五个符号，抄一遍就能看懂每一项在做什么；引一个库把它变成黑盒，
反而丢掉了这个项目最想讲清楚的东西。代价是约 40 行代码和一份要自己维护的测试。

为什么放在应用层、而不是让向量库去做
------------------------------------
两个后端的原生能力并不对等：Milvus 2.5+ 有内置的 BM25 Function，而 chromadb 1.5
的本地版**完全没有**（`SparseVectorIndexConfig(bm25=True)` 在本地直接报
"Sparse vector indexing is not enabled in local"，真正的 `collection.search()`
也只在托管版可用）。如果图省事只在 Milvus 上用原生 BM25，两个后端的检索能力就
不对称了——而 ecom/knowledge/vector_store/base.py 和 FLOWCHART 设计点 5 承诺的正是
"换后端只改配置，检索/rerank/测评的代码一行不动"。那条契约是这块最值钱的东西，
不能为了一个后端的新特性把它变成半真半假。

所以这里只依赖契约里现成的 `store.all_chunks()`，不关心底下是 Chroma 还是 Milvus。

索引的生命周期
--------------
懒加载 + 进程内缓存，缓存 key 是 `store.count()`。Streamlit 是长驻进程，
而 ingest.py 在**另一个进程**里跑，所以缓存不会自己感知到语料变化；按 count 失效
能覆盖"新增了 reviewed 文档、条数变了"这个主路径。
**已知盲区**：同一个 source 被等量改写时 count 不变、缓存不刷新——语料 253 条、
单人使用，接受这个边界；UI 上留了一个"重建索引"按钮可以手动调 reset_index()。
"""
import logging
import math
import threading

from ecom.knowledge.settings import BM25_B, BM25_K1
from ecom.knowledge.utils.logger import setup_logger

logger = setup_logger(__name__)

_index_cache = {"key": None, "index": None}
_index_lock = threading.Lock()

_jieba = None


def _get_jieba():
    """
    延迟导入 jieba。

    不放在模块顶层：HYBRID_SEARCH=false 时不该白白加载一份词典，而且
    "只 import 这个模块"（比如跑单测）也不该被一个可选的运行时依赖卡住。
    """
    global _jieba
    if _jieba is None:
        try:
            import jieba
        except ImportError as exc:
            # 默认的 ModuleNotFoundError 只说"没有这个模块"，不说怎么办。
            raise RuntimeError(
                "混合检索需要 jieba 分词。请执行：uv sync，"
                "或把 .env 里的 HYBRID_SEARCH 设为 false 退回纯向量检索"
            ) from exc

        # 首次分词会打印 "Building prefix dict ..."，压到 WARNING 以上
        jieba.setLogLevel(logging.WARNING)
        _jieba = jieba
    return _jieba


def tokenize(text: str) -> list:
    """
    中文分词 + 归一化，BM25 的输入。

    两个容易被忽略的细节：
      - **转小写**：jieba 会把 "SqliteSaver"、"RAGFlow" 这类拉丁词整体保留，
        它们的大小写在正文里并不统一，不转小写就会出现"提问写 SQLiteSaver、
        正文写 SqliteSaver"这种匹配不上的情况。
      - **丢掉纯标点/空白**：jieba 会把空格也当成一个 token 吐出来，标点同理。
        它们对打分毫无意义，却会一起被算进文档长度 dl，把长度归一化带偏。
    """
    tokens = _get_jieba().lcut(text or "")
    return [token.lower() for token in tokens if any(ch.isalnum() for ch in token)]


def idf(n_docs: int, df: int) -> float:
    """
    逆文档频率：一个词越罕见，命中它越说明问题。

    用 ln(1 + (N - n + 0.5)/(n + 0.5)) 这一支，**不是** ln((N - n + 0.5)/(n + 0.5))。
    差别在于加的那个 1：不加时，当某个词出现在超过一半的文档里，括号里的值小于 1、
    对数为负，这个词就会把文档的分数**往下拉**。中文里"的 / 是 / 在"这类词几乎
    必然触发——我们希望它们贡献接近 0，而不是贡献负数。
    """
    return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))


def term_score(tf: int, dl: int, avgdl: float, idf_value: float, k1: float, b: float) -> float:
    """
    单个查询词对单篇文档的贡献：

        idf * tf*(k1+1) / (tf + k1*(1 - b + b*dl/avgdl))

    分子里的 tf 让词出现得越多越相关，但除以分母后是**饱和**的——tf 从 1 涨到 3
    会加分，但绝不会有 3 倍，因为一个词在一段话里重复出现并不代表它重要三倍。
    k1 决定饱和的快慢。

    分母里的 `b*dl/avgdl` 是长度归一化：长文档天然更容易"撞上"查询词，
    不做惩罚的话长 chunk 会系统性地占便宜。b=0 关掉它，b=1 完全按长度折。
    """
    if avgdl <= 0:
        return 0.0
    denominator = tf + k1 * (1 - b + b * dl / avgdl)
    if denominator <= 0:
        return 0.0
    return idf_value * tf * (k1 + 1) / denominator


class BM25Index:
    """
    一份只读的 BM25 索引。建好之后查询是纯计算，没有 IO。

    docs 就是 `store.all_chunks()` 的返回值：[{"text","source","chunk_id"}, ...]。
    它**不保证顺序**（契约条款 5），所以这里自己按 (source, chunk_id) 排一遍——
    否则破平的时候就不知道是按什么在破，eval 的命中位次会跟着随机抖动。
    """

    def __init__(self, docs: list, k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b

        ordered = sorted(
            docs or [], key=lambda doc: (doc.get("source", ""), doc.get("chunk_id", -1))
        )

        self.chunks = []
        self.term_freqs = []
        self.doc_lengths = []
        self.doc_freq = {}

        for doc in ordered:
            text = doc.get("text") or ""
            tokens = tokenize(text)

            freq = {}
            for token in tokens:
                freq[token] = freq.get(token, 0) + 1

            self.chunks.append(
                {
                    "text": text,
                    "source": doc.get("source", ""),
                    "chunk_id": doc.get("chunk_id", -1),
                }
            )
            self.term_freqs.append(freq)
            self.doc_lengths.append(len(tokens))
            for token in freq:
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

        self.avgdl = (sum(self.doc_lengths) / len(self.doc_lengths)) if self.doc_lengths else 0.0

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, top_k: int) -> list:
        """
        返回 [{"text","source","chunk_id","score"}, ...]，按 score 降序。

        空索引、空查询、一个词都没匹配上都返回 []，不抛异常。
        """
        if top_k <= 0 or not self.chunks:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = [0.0] * len(self.chunks)

        # 查询词先去重：同一个词在问题里出现两次，不该把它的权重翻倍
        # （这正是"词频饱和"想避免的事，查询侧同样适用）。
        for token in set(query_tokens):
            df = self.doc_freq.get(token)
            if not df:
                continue
            weight = idf(len(self.chunks), df)
            for position, freq in enumerate(self.term_freqs):
                tf = freq.get(token)
                if tf:
                    scores[position] += term_score(
                        tf, self.doc_lengths[position], self.avgdl, weight, self.k1, self.b
                    )

        # score > 0 才算命中。不过滤的话，一个词都没匹配上的查询会返回一批
        # 分数为 0 的 chunk，看着像"召回了"，其实全是噪声。
        order = sorted(
            (i for i, score in enumerate(scores) if score > 0),
            key=lambda i: (-scores[i], self.chunks[i]["source"], self.chunks[i]["chunk_id"]),
        )[:top_k]

        return [{**self.chunks[i], "score": scores[i]} for i in order]


def get_index(store) -> BM25Index:
    """
    取当前库的 BM25 索引，没有或已过期就重建。

    过期判据是 `store.count()` 变了。加锁是因为 Streamlit 下可能有多个会话
    同时第一次提问，不锁会重复建索引（不至于出错，但白烧 CPU）。
    """
    key = (type(store).__name__, store.count())

    with _index_lock:
        if _index_cache["index"] is not None and _index_cache["key"] == key:
            return _index_cache["index"]

        index = BM25Index(store.all_chunks())
        logger.info("BM25 索引已建立：%d 个 chunk，平均长度 %.1f 词", len(index), index.avgdl)
        _index_cache["key"] = key
        _index_cache["index"] = index
        return index


def reset_index() -> None:
    """显式丢弃缓存。测试用；UI 上那个"重建索引"按钮也调它。"""
    with _index_lock:
        _index_cache["key"] = None
        _index_cache["index"] = None


def bm25_search(store, question: str, top_k: int) -> list:
    """混合检索里"关键词"这一路。失败由调用方（retrieve）兜底。"""
    return get_index(store).search(question, top_k)


if __name__ == "__main__":
    from ecom.knowledge.vector_store import get_vector_store

    demo_store = get_vector_store()
    try:
        for hit in bm25_search(demo_store, "LangGraph 的 Checkpointer 怎么用", 3):
            print(f"[{hit['score']:.3f}] {hit['source']}#{hit['chunk_id']}: {hit['text'][:40]}...")
    finally:
        demo_store.close()
