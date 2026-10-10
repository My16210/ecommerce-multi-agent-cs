"""
可选的阶段计时器：性能画像用。生产路径上一直挂着它也不花钱。

为什么挂在生产代码里，而不是在 profile.py 里重拼一条链路
--------------------------------------------------------
在 profile 脚本里重新拼一遍"embed → search → bm25 → fuse"很诱人，但那样测的是
**脚本拼出来的那条路**，不是线上真正跑的那条。两份编排一旦有一处不一样
（比如哪天 rerank 挪了个位置、或者某一路加了缓存），画像就开始骗人，而且
骗得悄无声息——数字照样好看。计时器挂在真正的执行点上，谁调都一样。

为什么用 ContextVar 而不是把计时器当参数往下传
----------------------------------------------
  - `retrieve` / `rerank` / `generate_answer` 的签名被测试和调用方钉住了，
    加一个 `collector` 参数是破坏性的，还会一路传染到所有调用点；
  - 计时是**横切**关注点，塞进业务函数的参数表会污染每一个函数；
  - ContextVar 跟着"当前执行上下文"走：Streamlit 的多线程 rerun、并发评测
    之间不会串味（普通模块级全局变量会）。

关闭时的开销
------------
`stage()` 进入时只做一次 `ContextVar.get()`，拿到 None 就直奔 yield——
不读时钟、不写字典。所以生产代码里挂满 `with stage(...)` 也测不出成本，
不需要"要画像时才临时打开"的开关。

`metric()` 同理，用来记 token 这类非时长量。
"""
import time
from contextlib import contextmanager
from contextvars import ContextVar


class Collector:
    """
    一次采集会话的容器。

    - `durations`：阶段名 → 累计秒数
    - `counts`：阶段名 → 被进入的次数。同一阶段在一题里进入多次会累加，
      这一列专门用来暴露"这里其实循环调了 N 次"——只看总时长看不出来。
    - `metrics`：非时长量（token 数等）。和 durations 分开，因为聚合方式不同：
      时长要看分位数，用量只需求和。
    """

    __slots__ = ("durations", "counts", "metrics")

    def __init__(self) -> None:
        self.durations: dict = {}
        self.counts: dict = {}
        self.metrics: dict = {}

    def add_metric(self, name: str, value) -> None:
        if not value:
            return
        self.metrics[name] = self.metrics.get(name, 0) + value

    def snapshot(self) -> dict:
        """拷一份出来。用于"每题一档"再聚合，避免被后续阶段改写"""
        return {
            "durations": dict(self.durations),
            "counts": dict(self.counts),
            "metrics": dict(self.metrics),
        }


_collector: ContextVar = ContextVar("timing_collector", default=None)


@contextmanager
def collect():
    """开始采集，yield 出一个 Collector。嵌套时内层覆盖外层，退出后外层恢复"""
    collector = Collector()
    token = _collector.set(collector)
    try:
        yield collector
    finally:
        _collector.reset(token)


def active() -> bool:
    """当前是否在采集。给"要不要额外算点东西"的分支用"""
    return _collector.get() is not None


@contextmanager
def stage(name: str):
    """给一段代码计时。没在采集时是空操作"""
    collector = _collector.get()
    if collector is None:
        yield
        return

    started = time.perf_counter()
    try:
        yield
    finally:
        # 记在 finally 里而不是正常路径上：这段抛异常时，已经花掉的时间也要留下。
        # 否则"失败的那几次特别慢"——画像里最该被看见的信号——恰好被吞掉。
        elapsed = time.perf_counter() - started
        collector.durations[name] = collector.durations.get(name, 0.0) + elapsed
        collector.counts[name] = collector.counts.get(name, 0) + 1


def metric(name: str, value) -> None:
    """记一个非时长量（token 数等）。没在采集时是空操作"""
    collector = _collector.get()
    if collector is not None:
        collector.add_metric(name, value)
