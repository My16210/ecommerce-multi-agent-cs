"""
日志模块
功能：记录每次请求，方便排查问题
"""
import logging
from pathlib import Path

from ecom.knowledge.settings import LOGS_PATH

LOG_FILE = "rag.log"
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logger(name: str = "rag") -> logging.Logger:
    """
    配置日志，同时输出到控制台和 logs/rag.log
    重复调用同一个 name 不会叠加 handler（否则日志会打多份）
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(FORMAT)

    Path(LOGS_PATH).mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(Path(LOGS_PATH) / LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


if __name__ == "__main__":
    logger = setup_logger()
    logger.info("测试日志")
    print("日志已写入 logs/rag.log")
