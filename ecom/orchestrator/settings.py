"""
编排层的配置：路由、汇总、worker 超时、回环上限、订单库。

密钥、模型名、仓库根、数据目录**不在这里**——那些在 `ecom/settings.py`。
这里只放"编排自己"的旋钮。这个文件里出现的 QWEN_* 只是从那边转手过来，
好让原来那几处 `from ... import QWEN_MODEL` 不用改（为改而改不值得）。

【这个文件原来叫 `cs/settings.py`，而那个名字是被一次事故逼出来的】

它原先确实叫 `config/settings.py`，和 rag_knowledge_base 保持一致。第一次跑通
知识库那一路时直接炸了：

    ImportError: cannot import name 'LOGS_PATH' from 'config.settings'

知识库 worker 是**进程内** import sibling 仓库的，而 rag 有 16 个模块写着
`from config.settings import ...`。我们自己先 import 了自己的 `config`，
它就占住了 `sys.modules["config"]`；等 rag 的模块再 import 时拿到的还是
**我们这一份**。于是它要的 `LOGS_PATH` 找不到——而更坏的情况是**它悄悄找到了
一个同名但含义不同的常量**：不报错，只是行为不对，这种最难查。

**合并之后这段历史的意义变了。** 那时候的结论是"本项目必须起一个足够独特的
顶层包名，别和 sibling 撞"；现在 sibling 就是自己，没有"别人的包"了，
所以当初为绕开它写的东西——这个独特命名、`workers/knowledge.py` 里的
`sys.path` 挂载、`_assert_imports_came_from()` 那道运行时闸门、以及它的回归
测试——全部作废，一并删掉。

只留下一件事没作废：**顶层包名就是全局命名空间，一个 `ecom` 从构造上就撞不了。**
这正是合并真正买到的那个东西。

【超时和上限这些"实验装置"写死在这里，不跟着 .env 走】

允许环境变量悄悄换掉它们，等于允许"跑的不是同一套东西"发生。
密钥走 .env（它必须可变），行为参数改代码（它不该悄悄变）。
"""
import os
from pathlib import Path

# 唯一一份读 .env、唯一一份算根路径的地方。见 ecom/settings.py。
from ecom.settings import (  # noqa: F401  —— QWEN_* 是转手给别人 import 的
    DATA_DIR,
    LOGS_DIR,
    QWEN_API_KEY,
    QWEN_BASE_URL,
    QWEN_MODEL,
)

# ---------------------------------------------------------------------------
# 模型：路由和汇总用 qwen-plus（和知识库 worker 生成答案同一个模型，一处配好）
# ---------------------------------------------------------------------------
# 路由用 temperature=0：这是一个分类任务，要的是稳定，不要"有创意"。
# 但**别把 0 当成确定性**——同一个输入重跑仍可能不同（MoE 路由、批处理），
# 所以下游解析失败要重试，不能假定"上次这样这次还这样"。
ROUTER_TEMPERATURE = float(os.getenv("ROUTER_TEMPERATURE", "0"))
SYNTH_TEMPERATURE = float(os.getenv("SYNTH_TEMPERATURE", "0.3"))

# 路由单次调用的超时。**必须短**：路由是一个几十 token 的分类任务，8 秒还没回来
# 就说明模型这条路当下不可用，"继续等"只是在让用户干等。
ROUTER_TIMEOUT = int(os.getenv("ROUTER_TIMEOUT", "8"))
# 路由重试的**总预算**（秒）。只有单次超时是不够的：3 次重试 × 8s 看起来没事，
# 但客户端 SDK 自己还带一层重试，两层是**乘法**——实测一次请求的路由环节
# 从 8 秒拖成了 25 秒。有总预算，最坏情况才有界。
ROUTER_BUDGET = int(os.getenv("ROUTER_BUDGET", "20"))
# 汇总的调用比路由长：它要读完几段结论再写一段话。
SYNTH_TIMEOUT = int(os.getenv("SYNTH_TIMEOUT", "60"))

# ---------------------------------------------------------------------------
# worker 的边界参数（这几条是"防无限循环/防拖死"的全部家当，改之前先想清楚）
# ---------------------------------------------------------------------------
# 单个 worker 的超时（秒）。超时按**失败**处理并走兜底，不是无限等。
# 数据分析那条最慢：生成代码 + 执行 + 可能审查 + 可能纠错，给足。
WORKER_TIMEOUT = {
    "knowledge": int(os.getenv("WORKER_TIMEOUT_KNOWLEDGE", "90")),
    "order": int(os.getenv("WORKER_TIMEOUT_ORDER", "20")),
    "analytics": int(os.getenv("WORKER_TIMEOUT_ANALYTICS", "180")),
}
WORKER_TIMEOUT_DEFAULT = int(os.getenv("WORKER_TIMEOUT_DEFAULT", "60"))

# Supervisor 的回环上限：**一次提问最多跑几轮**（含首轮）。
# 有它，最坏情况才有界——否则"某个 worker 说没答上来 → 再路由 → 还是它"
# 就是一个死循环。设 2 表示：首轮 + 最多一次重路由。
MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", "2"))
# 路由的重试次数（解析失败时）。见 ROUTER_TEMPERATURE 那段。
ROUTER_ATTEMPTS = int(os.getenv("ROUTER_ATTEMPTS", "3"))

# ---------------------------------------------------------------------------
# 订单库：默认 SQLite（零外部依赖，clone 下来就能跑）
# ---------------------------------------------------------------------------
ORDER_DB_PATH = Path(os.getenv("ORDER_DB_PATH", str(DATA_DIR / "orders.sqlite")))
# 给了 DSN 就走 MySQL（需要 uv sync --extra mysql）。两种后端跑的是同一份 schema。
ORDER_DB_DSN = os.getenv("ORDER_DB_DSN", "")

# 订单种子数据（派生产物，**进仓库**）。`derive_seed.py` 写它、`seed.py` 读它。
# 它和 orders.sqlite 不是一回事：sqlite 是跑一次 seed 就有的运行物（已 gitignore），
# 这两份 CSV 才是"订单数据长什么样"的唯一来源，丢了就得重新下 96MB 才能造出来。
ORDER_SEED_DIR = DATA_DIR / "orchestrator" / "seed"

# ---------------------------------------------------------------------------
# 对外请求体上限，防止一句 10 万字把路由 prompt 撑爆
# ---------------------------------------------------------------------------
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "2000"))

# ---------------------------------------------------------------------------
# 会话存储（多轮记忆）
# ---------------------------------------------------------------------------
# **和订单库分开一个文件**：两者的生命周期不一样。orders.sqlite 是"灌一次就
# 不动"的派生数据，删了能重灌；sessions.db 是"每天都在长"的运行数据，而且
# 里面是用户说过的话。它们不该共用一份备份/清理策略，也不该共用一条
# "删库重灌"的操作。
#
# 它同时**不是** knowledge 那边那个 chat_history.db：那个存的是知识库图的
# LangGraph checkpoint，只有 `ask.py` 用。为什么 supervisor 这边不复用它，
# 理由写在 ecom/orchestrator/session.py 的模块 docstring 里。
SESSION_DB_PATH = Path(os.getenv("SESSION_DB_PATH", str(DATA_DIR / "orchestrator" / "sessions.db")))

# 会话保留天数。**按时间清，不按条数清**——条数上限解决的是"上下文太长"
# （那是 HISTORY_TURNS 的事），时间上限解决的是"数据留太久"，两个问题别混。
SESSION_TTL_DAYS = int(os.getenv("SESSION_TTL_DAYS", "30"))

# 每次提问**回读几轮**历史喂给汇总。这是成本参数：轮数越多 prompt 越大、
# token 越贵。它和上面的保留期是两件事，所以是两个独立配置。
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "6"))
