"""
整个项目共享的一层配置：**仓库根、数据目录、日志目录、以及所有密钥和模型名。**

【为什么这个文件存在，而且只有它做这两件事】

合并进来之前，三个仓库各有一份 `settings.py`，各自用
`Path(__file__).resolve().parent.parent` 往上数两级来定位"仓库根"。
那在两个仓库各自的目录树里都是对的。搬进 `ecom/knowledge/`、
`ecom/analytics/` 之后，同样的算术会算出 `ecom/`——**不是报错，是算出一个
合法的、错误的目录**：`data/` 找不到、日志写到 `ecom/logs/`。

这类错误没有异常、没有堆栈，只会表现为"文件不存在"或"东西写丢了"。所以
它不能靠"记得改"，只能靠**结构上只有一处能算**：根在这里算一次，
`load_dotenv` 在这里调一次，其他地方一律 `from ecom.settings import ...`。

（`ecom/orchestrator/settings.py` 是第三个这么数目录的地方，它数的是三级，
同一个坑，一并收到这里。）

【这里放什么、不放什么】

放：**跨子系统共享、且改错了会互相影响的东西**——密钥、模型名、根路径。
不放：各子系统自己的旋钮。chunk 大小是知识库的事、worker 超时是编排层的事，
它们各自留在 `ecom/<子系统>/settings.py` 里。

理由：一个 200 行的"巨无霸 settings"读起来整齐，但改一个检索参数要动一个
三个子系统共用的文件，谁也不敢动——**整齐是给读者看的，风险是自己的**。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# ——————————————————————————————————————————————————————————————
# 路径：全项目唯一一次"往上数目录"
# ——————————————————————————————————————————————————————————————
# 本文件在 <仓库根>/ecom/settings.py，所以上两级就是仓库根。
ROOT = Path(__file__).resolve().parent.parent

# .env 只在这里加载一次。默认 override=False：真实环境变量优先于文件里的，
# 这样容器/CI 里 `QWEN_API_KEY=xxx python ...` 能盖掉 .env，不用改文件。
load_dotenv(ROOT / ".env")

# 数据不是代码，不进包目录——否则 `find ecom -name "*.py"` 这种日常动作
# 都要绕开 14M 语料。按子系统分目录，互相不干扰。
DATA_DIR = ROOT / "data"
KNOWLEDGE_DATA_DIR = DATA_DIR / "knowledge"
ANALYTICS_DATA_DIR = DATA_DIR / "analytics"
LOGS_DIR = ROOT / "logs"

# ——————————————————————————————————————————————————————————————
# 对话模型（通义千问）：路由 + 汇总，知识库的生成也用同一个
# ——————————————————————————————————————————————————————————————
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")

# ——————————————————————————————————————————————————————————————
# 数据分析用的是另一家（DeepSeek），单独一套
# ——————————————————————————————————————————————————————————————
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# 数据分析那边有一张"供应商 → 参数"的表，里面还留着 OpenAI 这一档（没人用，
# 但它是那张表"加一行就多一个供应商"这个设计的演示）。密钥放在这里而不是
# 那张表里，是为了守住"密钥只有一处"——表只负责挑，不负责存。
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ——————————————————————————————————————————————————————————————
# 检索那一路（BGE-M3 embedding / rerank），走硅基流动
# ——————————————————————————————————————————————————————————————
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

# 联网搜索（可选）。配了 Tavily key 就优先用它，否则退回免 key 的 ddgs。
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
