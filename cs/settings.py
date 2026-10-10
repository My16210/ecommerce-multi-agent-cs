"""
配置中心。所有"会变的数"收在这里，代码里不散落魔法值。

【为什么这个文件叫 cs/settings.py，而不是和 sibling 仓库一样的 config/settings.py】
这不是洁癖，是**踩出来的**。原先它确实叫 `config/settings.py`，和 rag_knowledge_base
保持一致。第一次跑通知识库那一路时直接炸了：

    ImportError: cannot import name 'LOGS_PATH' from 'config.settings'

知识库 worker 是**进程内** import sibling 仓库的（见 workers/knowledge.py），
而 rag 有 16 个模块写着 `from config.settings import ...`。我们自己先 import 了
自己的 `config`，它就占住了 `sys.modules["config"]`；等 rag 的模块再 import 时，
拿到的还是**我们这一份**。于是它要的 `LOGS_PATH` 找不到——而更坏的情况是
**它悄悄找到了一个同名但含义不同的常量**：不报错，只是行为不对，这种最难查。

同一个进程里，顶层包名就是全局命名空间。谁先占住谁说了算，而且**失败时不会告诉你
是撞名了**。所以这个仓库的顶层包只留一个 `cs`——一个足够独特、永远不会撞的名字。
测试里那条 `test_package_names_do_not_collide_with_siblings` 就是在守这件事。

（routing 那一路本来就没这个问题：它不 import rag。）

【sibling 仓库的定位】
本项目**不复制**那两个仓库的代码，而是在运行时引用它们：
  - rag_knowledge_base：知识库 worker **进程内** import（热路径，省一次进程启动）
  - sales_agent：数据分析 worker 走**子进程**（它自己也有一个顶层 `src` 包，
    而我们不能再指望"名字不撞"这种事；而且它是唯一带纠错循环、可能跑很久、
    还会执行模型生成的代码的 worker，给一个真进程边界和真超时反而更对）
路径默认按兄弟目录推，可用 .env 覆盖——因为面试官 clone 的目录结构和本机不同。

密钥只从 .env 读，不进代码。运行参数可以走 .env；"实验装置"类的常量——sibling
仓库在哪、worker 的超时——写死在这里，改代码才能改。理由和 rag 项目里把评测路径
锚死一样：让环境变量悄悄换掉它们，等于允许"跑的不是同一套东西"发生。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------------------
# 模型：路由和汇总用 qwen-plus（和知识库 worker 生成答案同一个模型，一处配好）
# ---------------------------------------------------------------------------
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")

# 路由用 temperature=0：这是一个分类任务，要的是稳定，不要"有创意"。
# 但**别把 0 当成确定性**——同一个输入重跑仍可能不同（MoE 路由、批处理），
# 所以下游解析失败要重试，不能假定"上次这样这次还这样"。
ROUTER_TEMPERATURE = float(os.getenv("ROUTER_TEMPERATURE", "0"))
SYNTH_TEMPERATURE = float(os.getenv("SYNTH_TEMPERATURE", "0.3"))

# 路由单次调用的超时。**必须短**：路由是一个几十 token 的分类任务，8 秒还没回来
# 就说明模型这条路当下不可用，"继续等"只是在让用户干等。
ROUTER_TIMEOUT = int(os.getenv("ROUTER_TIMEOUT", "8"))
# 路由重试的**总预算**（秒）。只有单次超时是不够的：3 次重试 × 60s = 180s，
# 实测真的发生过（网络正好不通时），一次演示就这么没了。有总预算，
# 最坏情况才有界——20 秒之内一定落到规则兜底上去。
ROUTER_BUDGET = int(os.getenv("ROUTER_BUDGET", "20"))
# 汇总的调用比路由长：它要读完几段结论再写一段话。
SYNTH_TIMEOUT = int(os.getenv("SYNTH_TIMEOUT", "60"))

# ---------------------------------------------------------------------------
# sibling 仓库路径
# ---------------------------------------------------------------------------
WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", str(ROOT.parent)))
RAG_KB_ROOT = Path(os.getenv("RAG_KB_ROOT", str(WORKSPACE_ROOT / "rag_knowledge_base")))
SALES_AGENT_ROOT = Path(os.getenv("SALES_AGENT_ROOT", str(WORKSPACE_ROOT / "sales_agent")))

# 数据分析 worker 用哪个解释器起子进程。
# sales_agent 是 requirements.txt 那套（不是 uv），所以默认直接 python；
# 本机用 uv 管环境的话，把它设成 "uv run --project <sales_agent> python"。
SALES_AGENT_PYTHON = os.getenv("SALES_AGENT_PYTHON", "python")

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
DATA_DIR = ROOT / "data"
ORDER_DB_PATH = Path(os.getenv("ORDER_DB_PATH", str(DATA_DIR / "orders.sqlite")))
# 给了 DSN 就走 MySQL（需要 uv sync --extra mysql）。两种后端跑的是同一份 schema。
ORDER_DB_DSN = os.getenv("ORDER_DB_DSN", "")

# ---------------------------------------------------------------------------
# 可观测：每次请求一条 JSONL
# ---------------------------------------------------------------------------
LOGS_DIR = ROOT / "logs"

# ---------------------------------------------------------------------------
# 对外请求体上限，防止一句 10 万字把路由 prompt 撑爆
# ---------------------------------------------------------------------------
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "2000"))
