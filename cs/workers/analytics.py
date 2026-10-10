"""
数据分析 worker：自然语言问销售数据（销量排名 / 趋势 / 退货率 …）。

**走子进程调 sibling 仓库 sales_agent，不进程内 import。** 三个理由，主次分明：

1. **非这样不可**：两个仓库都有顶层 `src` 包。同一进程里先 import 谁，
   另一个的 `from src.x import y` 就静默解析到先来的那个——不会报错，
   只会算错。这是硬约束，不是偏好。
2. **顺带赚到的隔离**：这是唯一带"生成代码 → 执行 → 报错改代码 → 再执行"
   纠错循环的 worker（最坏情况跑几分钟），也是唯一会执行**模型生成的代码**的。
   给它一个真进程边界，超时能真杀、崩了也炸不到 supervisor。
3. **依赖各管各的**：本仓库不需要把 pandas / matplotlib / deepseek 那套拖进来。

代价是每次调用多付一次 Python 启动 + 建图的钱（秒级）。它是最慢的 worker
（本来就要几十秒），这点开销不改变量级——**但知识库那一路就不一样了，
所以知识库走进程内**。两个 worker 用两种接法，是各自的账算下来不一样，不是不一致。
"""
import json
import os
import shlex
import subprocess
from pathlib import Path

from cs.settings import SALES_AGENT_PYTHON, SALES_AGENT_ROOT, WORKER_TIMEOUT
from cs.utils.logger import setup_logger

logger = setup_logger(__name__)

ENTRY = "analyze_json.py"


def _command(question: str) -> list:
    root = Path(SALES_AGENT_ROOT)
    script = root / ENTRY
    if not script.exists():
        raise RuntimeError(
            f"找不到数据分析仓库的入口：{script}。请把 sales_agent 和本项目放在同一层目录，"
            "或在 .env 里设 SALES_AGENT_ROOT=/绝对路径/sales_agent"
        )
    # SALES_AGENT_PYTHON 允许是 "uv run --project xxx python" 这种带参数的形式，
    # 所以按 shell 规则切词而不是当成单个可执行文件路径。
    return [*shlex.split(SALES_AGENT_PYTHON), str(script), question]


def run_analytics(question: str, chat_history: list = None) -> dict:
    root = Path(SALES_AGENT_ROOT)
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
        "sources": ["sales.csv（模拟销售数据）"],
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
