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

# ——————————————————————————————————————————————————————————————
# 图片理解（多模态）：走硅基流动的视觉模型
# ——————————————————————————————————————————————————————————————
# 默认复用上面的 SILICONFLOW_API_KEY / BASE_URL；想换供应商或模型时只改
# 这几个变量，不动代码。模型名会随平台上下架变化，跑不通时先看控制台的
# 模型列表——这类错误在运行时表现为一次可读的识别失败，不会崩。
VISION_API_KEY = os.getenv("VISION_API_KEY", SILICONFLOW_API_KEY)
VISION_BASE_URL = os.getenv("VISION_BASE_URL", SILICONFLOW_BASE_URL)
VISION_MODEL = os.getenv("VISION_MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct")

# 单次识别的超时。图片理解在用户点发送之后、进入主链路之前，它慢只会让
# 这一步多转一会儿；但**不能无限等**——超时就降级成"请手动输入"。
VISION_TIMEOUT = int(os.getenv("VISION_TIMEOUT", "30"))

# 图片大小上限（MB）。超过就拒绝，不发出去：既保护成本，也避免把一张
# 几十 MB 的图 base64 之后撑爆请求体。5MB 对截图/发票/面单足够。
VISION_MAX_IMAGE_MB = float(os.getenv("VISION_MAX_IMAGE_MB", "5"))

# 联网搜索（可选）。配了 Tavily key 就优先用它，否则退回免 key 的 ddgs。
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ——————————————————————————————————————————————————————————————
# 后台管理端的口令（admin.py）
# ——————————————————————————————————————————————————————————————
# 它放在这里而不是 orchestrator/settings.py，依据是本文件开头那条规则：
# **密钥只有一处**。这个口令是唯一一个和模型无关的密钥，但它仍然是个密钥。
#
# ⚠️ 它是**演示级的鉴权**，不是生产级的：一个明文口令，没有用户、没有角色、
# 没有审计（"谁批的"是人手填的一个字符串，填不填都行）。别把它说成
# "做了权限系统"——它的真实作用是让"生产该怎么接"这件事在代码里有个落点。
#
# 和 api.py 的 API_KEY **故意相反**：那边不配就放行（读为主，面向"clone 下来
# 就能调"），这边不配就**拒绝启动**。理由见 admin.py 的 docstring——
# 读可以默认开放，写必须默认关闭。
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
