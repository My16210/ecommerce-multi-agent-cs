"""日志。按日期分文件，同时打到控制台。

不做彩色、不做 JSON 结构化——那是接 ELK 才需要的事。这里的日志是给人看的：
出了问题能照着复现一条请求。机器可读的那一份在 `trace.py`（JSONL）。
"""
import logging
import sys
from datetime import datetime

from ecom.orchestrator.settings import LOGS_DIR

_configured = False


def _configure_once() -> None:
    global _configured
    if _configured:
        return

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logfile = LOGS_DIR / f"{datetime.now():%Y-%m-%d}.log"

    handler_file = logging.FileHandler(logfile, encoding="utf-8")
    handler_file.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    )
    handler_console = logging.StreamHandler(sys.stderr)
    handler_console.setFormatter(logging.Formatter("%(levelname)-7s %(name)s | %(message)s"))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler_file)
    root.addHandler(handler_console)
    _configured = True


def setup_logger(name: str) -> logging.Logger:
    """取一个 logger。重复调用只配置一次，不会叠加 handler（否则日志会翻倍）。"""
    _configure_once()
    return logging.getLogger(name)
