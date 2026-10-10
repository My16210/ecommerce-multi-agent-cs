"""
知识库的命令行入口。

这些脚本原本躺在 `rag_knowledge_base` 的仓库根，靠"当前工作目录就是仓库根"
来互相 import（`from gold_set import ...`）。名字又都太通用（`main.py`、
`review.py`），留在新仓库根会和 `main.py`（编排层 CLI）打架——所以收进包里，
调用方式从 `python ingest.py` 变成 `python -m ecom.knowledge.cli.ingest`。

改名时被当模块 import 的只有两个：`gold_set`（被 answer_eval 和测试用）
和 `tune_rrf`（被 answer_eval / gold_set 用），它们现在从 `ecom.knowledge.cli`
下面拿。

`ask.py` 是从 `main.py` 改过来的：仓库根已经有 `main.py` 了，两个 `main.py`
会让"改哪个 main"变成每次都要想一下的事。这个文件做的是"提问一次"，
名字照着它做的事叫。

各脚本的职责：
    ask.py            问一句，走完整张图（带会话持久化）
    ingest.py         把 documents/ 切块、向量化、入库（**向量库唯一写入口**）
    review.py         人审队列：看待审问题、批准/驳回
    build_eval_set.py 用模型生成 chunk 级合成评测集
    gold_set.py       生成待人工标注的金标准候选，人工逐条勾
    answer_eval.py    答案质量评测：忠实性 / 相关性 / 弃答
    tune_rrf.py       在留出集上扫 RRF 的权重与 K
    profile.py        逐阶段耗时剖析
    recover_corpus.py 从向量库反推原文语料（应急）
"""
