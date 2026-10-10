import logging
from datetime import datetime

from ecom.settings import LOGS_DIR


def setup_logger(name: str = "analytics") -> logging.Logger:
    # 这里原来写的是 `Path(__file__).parent.parent.parent / "logs"`。
    # 那是在 <仓库根>/src/utils/logger.py 的深度上数的：utils → src → 仓库根。
    # 搬进 ecom/analytics/utils/ 之后，同样的三级数到的是 `ecom/`，
    # 日志会写进 ecom/logs/——**不报错，只是写错了地方**。所以这个算术删掉，
    # 换成从 ecom.settings 拿全项目唯一的 LOGS_DIR。
    log_dir = LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 文件输出
    log_file = log_dir / f"{datetime.now().strftime('%Y%m%d')}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    # 控制台输出
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(sh)

    return logger
