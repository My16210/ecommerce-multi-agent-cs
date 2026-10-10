"""
知识库子系统的配置。

**这个模块只放知识库自己的旋钮。** 密钥、模型名、仓库根、数据目录都在
`ecom/settings.py`——那是全项目唯一读 `.env`、唯一算根路径的地方。

原来这里第一行是 `PROJECT_ROOT = Path(__file__).resolve().parent.parent`，
因为这份文件当时在 `config/` 下、往上两级正好是仓库根。现在它躺在
`ecom/knowledge/` 里，同样的算法会算出 `ecom/`——不报错，只是此后所有
`data/`、`logs/` 全部指错地方。所以那个算法在这里被删掉、不再复制一份。
"""
import os

from ecom.settings import (
    KNOWLEDGE_DATA_DIR,
    LOGS_DIR,
    QWEN_API_KEY,
    QWEN_BASE_URL,
    QWEN_MODEL,
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
    TAVILY_API_KEY,
)

# 下面这些常量历史上从别的模块被 import 过（`from ecom.knowledge.settings import QWEN_MODEL`
# 之类），所以即使这里只是转手，名字也必须继续存在——改名会让 25 处 import 白改。
__all__ = [
    "QWEN_API_KEY",
    "QWEN_BASE_URL",
    "QWEN_MODEL",
    "SILICONFLOW_API_KEY",
    "SILICONFLOW_BASE_URL",
    "TAVILY_API_KEY",
]

# Embedding配置（BGE-M3，用硅基流动API）
BGE_M3_MODEL = "BAAI/bge-m3"

# Rerank配置（同样走硅基流动，与 Embedding 共用一个 key）
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_TIMEOUT = 30

# 联网搜索**整条这一路**的时间预算（秒），Tavily 与 ddgs 共用。
# 为什么需要一个总预算：ddgs 自己的 timeout 是"每一批"的超时，它会按引擎分批
# 发请求、批与批之间重新计时，所以总耗时可以是那个值的若干倍——画像里量到的
# 5.6 秒主要就是这么来的。联网这一路上限保住，答案才不会被它拖着不出来。
WEB_SEARCH_TIMEOUT = float(os.getenv("WEB_SEARCH_TIMEOUT", "8"))

# RAG配置
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5
RERANK_TOP_K = 3
WEB_SEARCH_MAX_RESULTS = 5
EMBED_BATCH_SIZE = 32          # 硅基流动单次 embedding 的批大小

# 问题级向量缓存条数（精确重复的问题才命中）。设 0 等于关掉缓存。
# 缓存的是"问题 → 向量"，不是"问题 → 答案"——语义答案缓存明确不做，
# 理由见 ecom/knowledge/embedding/embedder.py 里 `get_query_embedding` 上方那段。
QUERY_EMBED_CACHE_SIZE = int(os.getenv("QUERY_EMBED_CACHE_SIZE", "256"))

# 混合检索（向量 + BM25，RRF 融合）
#
# 实现放在应用层（ecom/knowledge/retriever/），与向量库后端无关——这样 Chroma 和 Milvus
# 的检索能力保持对称，"换后端只改配置"那条契约才成立。详见 FLOWCHART.md
# 「混合检索：为什么在应用层」。
HYBRID_SEARCH = os.getenv("HYBRID_SEARCH", "true").lower() in ("1", "true", "yes")

# BM25 的两个超参，值是 Robertson/TREC 的经典默认。
# k1 管词频饱和：某词出现 3 次应该比 1 次相关，但不该是 3 倍。
# b  管长度归一化：0 = 完全不管长度，1 = 完全按长度折；512 字的 chunk 长短有差异，取 0.75。
# 两个都放出来是为了能亲手调、再看 eval 的命中位次怎么变。
BM25_K1 = float(os.getenv("BM25_K1", "1.5"))
BM25_B = float(os.getenv("BM25_B", "0.75"))

# RRF（倒数排名融合）的平滑常数。K 越大，名次靠前的几条之间的差距被压得越平。
# 60 是原论文（Cormack et al. 2009）给的经典值，实践中很少需要调。
RRF_K = int(os.getenv("RRF_K", "60"))

# 两路权重。**默认都是 1.0，也就是不加权。**
#
# 注意：等权 RRF 里"调 K"和"调权重"是两件不同的事——任一路第 1 名的得分恒为
# w/(K+1)，所以 K 动不了两路的话语权，只有权重能。想改两路的相对强弱，改这里。
#
# 那为什么默认不加权？因为 tune_rrf.py 在留出集上没调出稳定收益（见 README）。
# 保留这两个参数是为了让那个结论**可复现**：跑 tune_rrf.py 能复现出同样的结果，
# 而不是只能信文档里一句话。
RRF_WEIGHT_VECTOR = float(os.getenv("RRF_WEIGHT_VECTOR", "1.0"))
RRF_WEIGHT_BM25 = float(os.getenv("RRF_WEIGHT_BM25", "1.0"))

# 两路各自先召回多少条再融合。融合完会截断回 TOP_K。
# 取 20（约全库 253 条 chunk 的 8%）是为了给融合足够的候选——
# 混合检索的价值就在于把"BM25 命中、向量漏了"的那条挤进最终的 Top-5。
VECTOR_RECALL_K = int(os.getenv("VECTOR_RECALL_K", "20"))
BM25_RECALL_K = int(os.getenv("BM25_RECALL_K", "20"))

# 向量库后端：chroma（默认，纯本地零外部依赖）/ milvus（需要 Docker）
# 两个后端实现同一套契约，切换不影响检索、rerank、测评的逻辑
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "chroma").strip().lower()
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")     # 本地 Standalone 无鉴权，留空即可
EMBED_DIM = 1024                                  # BGE-M3 固定 1024 维，Milvus 建表要用

# 提问价值判定 + 待审核队列
# 判定结果只进待审核队列，绝不自动写入向量库（避免知识库自污染）
ENABLE_QUESTION_REVIEW = os.getenv("ENABLE_QUESTION_REVIEW", "true").lower() in ("1", "true", "yes")
REVIEW_DB_PATH = str(KNOWLEDGE_DATA_DIR / "review_queue.db")
REVIEWED_DOCS_PATH = str(KNOWLEDGE_DATA_DIR / "documents" / "reviewed")
JUDGE_TIMEOUT = 30

# 成本估算单价（每百万 tokens，人民币），给 profile.py 用。
#
# **这几个数会变**，写在配置里而不是抄进文档，就是为了只有一处会过期。
# 检索那一路（BGE-M3 / BGE-Reranker-v2-m3）在硅基流动上属于免费档，
# 所以整个检索链路是 ￥0——真正花钱的只有下面这个对话模型。
# 按 2025 年百炼（DashScope）qwen-plus 非思考模式、输入 ≤128K 的档位填的，
# 换模型或官方调价后改这里，或者用 .env 覆盖。
QWEN_PRICE_INPUT = float(os.getenv("QWEN_PRICE_INPUT", "0.8"))
QWEN_PRICE_OUTPUT = float(os.getenv("QWEN_PRICE_OUTPUT", "2.0"))

# 合成评测集（chunk 级）：由 build_eval_set.py 生成，**提交进仓库**——
# 它是评测基准，不是运行产物；换了语料或改了出题 prompt 才需要重新生成。
EVAL_SET_PATH = str(KNOWLEDGE_DATA_DIR / "eval_set.jsonl")

# 金标准集（chunk 级，**人工标注**）：由 gold_set.py 生成待标注文件、人工逐条确认。
# 与上面那份合成集的关键区别：那份的"标准答案"是模型自己出的题自己给的，
# 这份只有人点过头的条目才算数。同样提交进仓库——它是评测基准，不是运行产物。
GOLD_SET_PATH = str(KNOWLEDGE_DATA_DIR / "gold_set.jsonl")

# 答案质量评测快照：由 answer_eval.py 生成，**提交进仓库**。
# 存的是"当时检索到了什么、模型答了什么、判定给了什么"——留着它，
# 报告才是在读一份不变的被测对象，而不是每次重跑都换一批答案。
ANSWER_EVAL_PATH = str(KNOWLEDGE_DATA_DIR / "answer_eval.jsonl")

# 语料里查不到答案的问题（**手写**，不是生成物）。
# 它测的是另一件事：库里没有答案时，系统该说"资料里没有"，而不是编一个出来。
# 每条都带 why 说明为什么查不到，且 build 时会用检索自检一遍。
UNANSWERABLE_PATH = str(KNOWLEDGE_DATA_DIR / "unanswerable_questions.jsonl")

# 存储路径（绝对路径，避免依赖当前工作目录）
VECTOR_DB_PATH = str(KNOWLEDGE_DATA_DIR / "vector_store")
DOCUMENTS_PATH = str(KNOWLEDGE_DATA_DIR / "documents")
CHAT_DB_PATH = str(KNOWLEDGE_DATA_DIR / "chat_history.db")
LOGS_PATH = str(LOGS_DIR)
COLLECTION_NAME = "knowledge_base"   # 两个后端共用同一个逻辑集合名
