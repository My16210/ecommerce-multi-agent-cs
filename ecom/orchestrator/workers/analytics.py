"""
数据分析 worker：自然语言问销售数据（销量排名 / 趋势 / 退货率 …）。

**仍然走子进程——但理由变了，这个变化值得写清楚。**

合并**之前**，这里的第一个理由是硬约束：两个仓库都有顶层 `src` 包，同一进程里
先 import 谁，另一个的 `from src.x import y` 就静默解析到先来的那个。不报错，
只算错。所以不得不隔一个进程。

合并**之后**那个理由没了——`ecom.analytics` 就在同一个包里，进程内 import
毫无问题。（这正是知识库 worker 现在的样子。）但边界**留着，而且是故意的**：

1. **它执行模型生成的代码。** 这是本系统里唯一一处"别人的文本变成我们机器上
   跑起来的程序"。沙箱在 `ecom/analytics/core/sandbox.py`（AST 白名单 + 受限
   builtins），但沙箱**不做超时、也不做子进程隔离**——`while True: pass` 拦不住。
   真进程边界是补上这一层的唯一办法：`subprocess.run(timeout=…)` 到点是**真杀**，
   而进程内那个"超时"只是不再等它（见 README「已知限制」）。
2. **它可能跑几分钟。** 生成 → 执行 → 报错 → 改 → 再执行，循环几轮。进程内跑
   意味着这几分钟里 supervisor 那条线程被占着、还占着内存。
3. 顺带，崩了炸不到 supervisor。

代价是每次调用多付一次 Python 启动 + 建图的钱（秒级）。它本来就要几十秒，
这点开销不改变量级——**知识库那一路不一样**（热路径、秒级），所以那个走进程内。
两种接法不是不一致，是各自的账算下来不一样。

（合并前还有第 4 条"依赖各管各的，本仓库不用把 pandas 拖进来"。那条现在不成立了
——pandas 已经是核心依赖，见 pyproject。文件里留这句话是因为它曾经是个理由，
删掉会让人以为当时没想过。）
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from ecom.orchestrator.settings import WORKER_TIMEOUT
from ecom.orchestrator.utils.logger import setup_logger
from ecom.settings import ROOT

logger = setup_logger(__name__)

# 子进程的入口。合并前它是个文件路径（`../sales_agent/analyze_json.py`），
# 现在是个模块名——顺带解决了"哪个 python 起它"的问题：用 `sys.executable`，
# 也就是**跑了 supervisor 的那个解释器**。合并前这是个配置项
# （SALES_AGENT_PYTHON，因为两个仓库各管各的环境），现在同一个 venv，不需要了。
ENTRY_MODULE = "ecom.analytics.cli.analyze"


def _command(question: str) -> list:
    return [sys.executable, "-m", ENTRY_MODULE, question]


def run_analytics(question: str, chat_history: list = None) -> dict:
    root = ROOT
    # 子进程自己的超时设得比 supervisor 那层**略短**：让它先触发，
    # 这样我们拿到的是一条"数据分析跑超时了"的明确信息，
    # 而不是 supervisor 那边一句笼统的"worker 没返回"。
    limit = max(5, WORKER_TIMEOUT.get("analytics", 180) - 5)

    env = dict(os.environ)
    # 子进程的 PYTHONIOENCODING 固定 utf-8：我们要按 utf-8 解它的 stdout，
    # 但 Windows 默认控制台是 cp936，不固定下来这边会解出乱码。
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        proc = subprocess.run(
            _command(question),
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=limit,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"数据分析超过 {limit}s 未返回，已终止子进程") from exc

    payload = _parse_stdout(proc.stdout)
    if payload is None:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        raise RuntimeError(
            "数据分析子进程没有输出可解析的 JSON（返回码 "
            f"{proc.returncode}）。stderr 末尾：{' / '.join(tail) or '（空）'}"
        )

    if not payload.get("ok"):
        raise RuntimeError(payload.get("error") or "数据分析子进程返回了失败状态")

    report = payload.get("report") or ""
    chart_path = payload.get("chart_path") or ""
    if chart_path and not Path(chart_path).is_absolute():
        chart_path = str((root / chart_path).resolve())

    # 审查没通过时报告里已标注风险，这里也带出去——不能让它看起来像一次干净的成功
    if payload.get("review_passed") is False:
        logger.warning("数据分析的报告未通过代码审查，结果可能不可靠")

    return {
        "answer": report,
        "insufficient": not report.strip(),
        # 来源标签由子进程从**当前数据集自己的 NOTES.md** 里带出来，不在这里写死。
        # 写死的话，换数据集时这个回答底下会一直挂着旧数据集的名字，还不报错。
        "sources": [payload.get("source_label") or "数据分析"],
        "tokens": payload.get("tokens") or {"in": 0, "out": 0},
        "elapsed_ms": payload.get("elapsed_ms"),
        "data": {
            "chart_path": chart_path,
            "chart_generated": bool(payload.get("chart_generated")),
            "review_passed": payload.get("review_passed"),
        },
    }


def _parse_stdout(stdout: str):
    """
    取 stdout 里**最后一行**能解析成 JSON 的东西。

    不假定"整个 stdout 就是一行 JSON"——子进程里任何一处漏网的 print 都会插在前面。
    从后往前找第一个能解析的行，比"严格要求一行"稳。
    """
    for line in reversed((stdout or "").strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None
