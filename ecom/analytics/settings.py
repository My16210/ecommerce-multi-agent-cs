"""
数据分析子系统的配置。

【保留"类 + 单例"这个写法】

另外两个子系统（knowledge / orchestrator）是一排扁平模块常量，这里是
`Settings` 类 + 全局 `settings` 对象。合并时**没有强行统一**——统一成一种
风格只需要动这个文件，但要连带改掉 6 个 agent 节点里的
`from ... import settings` 用法。收益是"看起来一致"，代价是让一次搬家
顺手变成一次重构，而重构是可以不做的。所以风格照旧。

【改了哪两处】

1. `_ROOT`（原来 `Path(__file__).resolve().parent.parent`）**删掉**。
   这个文件原来在 `config/` 下，往上两级是仓库根；现在它在
   `ecom/analytics/` 里，同样的算法会算出 `ecom/`——不报错，只是
   `data/raw/sales.csv` 从此找不到。路径改从 `ecom.settings` 拿。
2. 密钥不再自己 `os.getenv`。原来这个类和 `ecom/settings.py` 会各读一遍
   `DEEPSEEK_API_KEY`——同一个 key 两处配、改一处忘一处的经典开局。
   现在这里只负责"挑哪个供应商"，密钥怎么来不归它管。

【原来的用法说明照抄在这里，只改了模块路径】

    from ecom.analytics.settings import settings
    llm = ChatOpenAI(**settings.get_llm_kwargs("deepseek"))  # 指定模型
    llm = ChatOpenAI(**settings.get_llm_kwargs())             # 默认 deepseek
"""
import os

from ecom.settings import (
    ANALYTICS_DATA_DIR,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    QWEN_API_KEY,
    QWEN_BASE_URL,
    QWEN_MODEL,
)


class Settings:
    """所有配置集中管理，方便全局调用"""

    # ===== 多模型配置（每个模型提供商一段） =====
    # 以后加新模型：在 .env 里加几行、在 ecom/settings.py 里读出来、在这里加一段。
    # 三步比原来多一步——这是"密钥只有一处"要付的钱，值得：原来那一步
    # （在本文件里再 os.getenv 一遍）才是两份真相的来源。
    #
    # temperature 是本子系统自己的旋钮（换个场景可能想换），所以留在本地读，
    # 不进 ecom/settings.py。
    LLM_CONFIGS: dict = {
        "deepseek": {
            "model": DEEPSEEK_MODEL,
            "api_key": DEEPSEEK_API_KEY,
            "base_url": DEEPSEEK_BASE_URL,
            "temperature": float(os.getenv("DEEPSEEK_TEMPERATURE", "0.3")),
        },
        "openai": {
            "model": OPENAI_MODEL,
            "api_key": OPENAI_API_KEY,
            "base_url": OPENAI_BASE_URL,
            "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
        },
        "qwen": {
            "model": QWEN_MODEL,
            "api_key": QWEN_API_KEY,
            "base_url": QWEN_BASE_URL,
            "temperature": float(os.getenv("QWEN_TEMPERATURE", "0.3")),
        },
        # 以后加新模型，复制上面一段改名字就行
    }

    # ===== 数据路径 =====
    # 集中管理，避免各节点硬编码。数据在 <仓库根>/data/analytics/ 下，不进包目录——
    # raw/ 是进仓库的输入、output/ 是跑出来的图，两个性质不同，所以分两级而不是平铺。
    DATA_DIR = ANALYTICS_DATA_DIR
    RAW_CSV_PATH = ANALYTICS_DATA_DIR / "raw" / "sales.csv"
    OUTPUT_DIR = ANALYTICS_DATA_DIR / "output"

    # ===== MySQL（Text-SQL 用，可选） =====
    # 默认 3306 是 MySQL 的标准端口。本机 Docker 里那个 mysql8 在 3308，
    # 真要用时在 .env 里写 MYSQL_PORT=3308——**默认值不跟着本机改**：
    # 一台机器上的端口不该成为这个项目的默认值。
    MYSQL_HOST: str = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT: int = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER: str = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DATABASE: str = os.getenv("MYSQL_DATABASE", "")

    def get_mysql_url(self) -> str:
        """生成 SQLAlchemy 连接字符串"""
        return (
            f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
        )

    def get_llm_kwargs(self, provider: str = "deepseek") -> dict:
        """
        获取 ChatOpenAI 初始化参数
        :param provider: 模型提供商，默认 deepseek，可选：deepseek/openai/qwen
        """
        if provider not in self.LLM_CONFIGS:
            available = list(self.LLM_CONFIGS.keys())
            raise ValueError(f"不支持的模型提供商: {provider}，可选: {available}")
        kwargs = self.LLM_CONFIGS[provider].copy()
        if not kwargs.get("api_key"):
            raise ValueError(
                f"未配置 {provider} 的 API Key。请复制 .env.example 为 .env 并填写对应变量。"
            )
        return kwargs

    def list_providers(self) -> list:
        """列出所有已配置的模型提供商"""
        return list(self.LLM_CONFIGS.keys())


# 全局单例，其他模块直接 import 使用
settings = Settings()
