"""
千问客户端的**唯一**构造点。

answer.py（生成答案，temperature=0.3）和 judge.py（判定，temperature=0）原本
各写了一份 `_get_llm()`，字段几乎一样。build_eval_set.py 会是第三个调用方，
三份里任何一份漏改（比如超时），症状都是"某个链路偶尔卡住"——这种 bug 很难
定位到"三处配置不同步"上面。所以集中到这里。

**调用方不持有 client**：每次 `get_llm()` 拿到的都是缓存的同一个实例，
不要 close、不要改它的属性（改了全局都跟着变）。
"""
from langchain_openai import ChatOpenAI

from ecom.knowledge.settings import QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL

# 按 (温度, 超时) 缓存。这两个值决定 client 的行为，同值就必须复用同一个连接池。
# 线程安全不额外加锁：最坏情况是两个线程各建一次，多一个 client 而已，不影响正确性。
_clients: dict[tuple[float, int], ChatOpenAI] = {}


def get_llm(temperature: float = 0.3, timeout: int = 60) -> ChatOpenAI:
    """取千问客户端（按参数缓存）

    缺 key 时抛的错要能照着改，所以错误信息里写清去哪配——这个异常会一路
    冒到命令行，是使用者最可能撞上的第一个错误。
    """
    cache_key = (temperature, timeout)
    if cache_key not in _clients:
        if not QWEN_API_KEY:
            raise RuntimeError("缺少 QWEN_API_KEY，请在项目根目录的 .env 中配置")
        _clients[cache_key] = ChatOpenAI(
            model=QWEN_MODEL,
            api_key=QWEN_API_KEY,
            base_url=QWEN_BASE_URL,
            temperature=temperature,
            timeout=timeout,
            # 流式模式下**默认拿不到 usage**：OpenAI 兼容接口只在
            # `stream_options={"include_usage": true}` 时才在最后一个 chunk 带上用量，
            # langchain 用这个开关控制它。不打开的话，界面照常逐字出答案，
            # 但 profile.py 的成本表会变成 ￥0.0000/题——而且它打印得毫无异常，
            # 是最典型的那种"静默错"。所以默认打开。
            # 非流式调用不受影响（本来就有 usage）。
            stream_usage=True,
        )
    return _clients[cache_key]
