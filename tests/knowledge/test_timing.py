"""
阶段计时器的行为。钉住两件事：**开着的时候数得准**，**关着的时候真的没有副作用**。

后者是它敢挂在生产代码（retrieve / rerank / generate）里的全部前提。如果关掉时
还会读时钟、写全局字典，那它就是往线上链路里偷偷塞了一笔开销，而且是最难被发现
的那种：没有报错、没有输出，只是每个阶段都慢一点点。所以"关闭即空操作"要有用例守着。

另外钉住 ContextVar 的隔离性——docstring 里承诺了"Streamlit 多线程 rerun 不会串味"，
这个承诺不能只写在注释里。
"""
import threading
import time

from ecom.knowledge.utils.timing import active, collect, metric, stage

# --------------------------------------------------------------------------
# 开着的时候：数得准
# --------------------------------------------------------------------------


def test_stage_records_duration_and_count():
    with collect() as collector:
        with stage("x"):
            time.sleep(0.01)

    assert collector.durations["x"] >= 0.01
    assert collector.counts["x"] == 1


def test_repeated_stages_accumulate():
    """同一阶段进入多次要累加

    只报总时长看不出"这里其实循环调了 N 次"，counts 那一列就是为它留的。
    """
    with collect() as collector:
        for _ in range(3):
            with stage("loop"):
                pass

    assert collector.counts["loop"] == 3


def test_failed_stage_still_records():
    """抛异常的那一次也要留下耗时

    计时写在 finally 里就是为了这个：画家里最该被看见的信号恰恰是
    "失败的那几次特别慢"（超时重试），漏掉它等于把最贵的 bug 藏起来。
    """
    with collect() as collector:
        try:
            with stage("boom"):
                raise RuntimeError("模拟失败")
        except RuntimeError:
            pass

    assert collector.counts["boom"] == 1
    assert "boom" in collector.durations


def test_metric_sums_and_ignores_falsy():
    with collect() as collector:
        metric("tokens", 10)
        metric("tokens", 5)
        metric("tokens", 0)      # 0 不该凭空造出一个键
        metric("never", 0)

    assert collector.metrics["tokens"] == 15
    assert "never" not in collector.metrics


def test_snapshot_is_a_detached_copy():
    """snapshot 必须拷贝：调用方拿它去聚合时，采集还在继续往里写"""
    with collect() as collector:
        with stage("a"):
            pass
        taken = collector.snapshot()
        with stage("b"):
            pass

    assert set(taken["durations"]) == {"a"}
    assert set(collector.durations) == {"a", "b"}


# --------------------------------------------------------------------------
# 关着的时候：真的没副作用
# --------------------------------------------------------------------------


def test_stage_outside_collect_is_a_noop():
    assert active() is False
    with stage("nobody-listening"):
        pass
    assert active() is False


def test_metric_outside_collect_is_a_noop():
    metric("tokens", 999)   # 不该抛，也不该留痕
    assert active() is False


# --------------------------------------------------------------------------
# 嵌套与并发
# --------------------------------------------------------------------------


def test_nested_collect_shadows_then_restores():
    """内层 collect 接管，退出后外层恢复——各记各的，不互相污染"""
    with collect() as outer:
        with stage("a"):
            pass
        with collect() as inner:
            with stage("b"):
                pass
        with stage("c"):
            pass

    assert "a" in outer.durations
    assert "c" in outer.durations
    assert "b" not in outer.durations      # b 记进了内层
    assert "b" in inner.durations


def test_collector_is_isolated_between_threads():
    """ContextVar 的核心承诺：并发下不串味

    Streamlit 的 rerun 可能在不同线程里跑。这里若用普通模块级全局变量，
    两个线程的阶段会写进同一本账，画出来的分位数就是两条链路混在一起的。
    """
    seen = {}

    def worker(name: str) -> None:
        with collect() as collector:
            with stage(f"only-in-{name}"):
                time.sleep(0.01)
            seen[name] = set(collector.durations)

    threads = [threading.Thread(target=worker, args=(n,)) for n in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert seen["a"] == {"only-in-a"}
    assert seen["b"] == {"only-in-b"}
