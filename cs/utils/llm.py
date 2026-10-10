"""
取 qwen 客户端（按参数缓存）。缺 key 时报错要能照着改。

【为什么 max_retries=0：重试只能有一层】
openai 客户端默认自带 2 次重试。而外面（router）也有一套重试 + 超时预算——
两层叠在一起，**总耗时是乘法不是加法**：单次超时 8 秒 × 客户端自己试 3 次
= 一次 invoke 就 24 秒，外面那个"20 秒总预算"直接失效。实测真发生了：路由本该
8 秒放弃，实际拖了 25 秒才开始走规则兜底。

所以这里关掉 SDK 自带的重试，把重试权收回到我们的代码里。理由是**我们比 SDK
多知道两件事**：这次调用值不值得重试（路由是幂等的分类，值得；写退货不是，
绝不能重试），以及整件事的总预算是多少。这两件 SDK 都不知道。
"""
from langchain_openai import ChatOpenAI

from cs.settings import QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL

_clients: dict = {}


def get_llm(temperature: float = 0.0, timeout: int = 60) -> ChatOpenAI:
    key = (temperature, timeout)
    if key not in _clients:
        if not QWEN_API_KEY:
            raise RuntimeError(
                "缺少 QWEN_API_KEY。请在项目根目录建 .env（复制 .env.example），"
                "填入 QWEN_API_KEY / QWEN_BASE_URL。"
            )
        _clients[key] = ChatOpenAI(
            model=QWEN_MODEL,
            api_key=QWEN_API_KEY,
            base_url=QWEN_BASE_URL,
            temperature=temperature,
            timeout=timeout,
            max_retries=0,  # 见模块开头：重试只能有一层，那一层是我们
        )
    return _clients[key]
