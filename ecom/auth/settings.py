"""身份层的开关与存储位置。"""
import os
from pathlib import Path

from ecom.settings import DATA_DIR

# 默认关闭：本地 CLI / Streamlit 演示不需要登录；对公网提供 API 时必须打开。
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}

# 和订单库、会话库分开一个文件：
#   · orders.sqlite 是派生数据，删了能重灌；
#   · sessions.db  是用户说过的话；
#   · auth.db      是凭证和审计，备份/保留策略和上面两个都不一样。
AUTH_DB_PATH = Path(
    os.getenv("AUTH_DB_PATH", str(DATA_DIR / "orchestrator" / "auth.db"))
)

# bearer token 的熵。32 字节 = 256 bit，token_urlsafe 出来约 43 个字符。
AUTH_TOKEN_BYTES = int(os.getenv("AUTH_TOKEN_BYTES", "32"))

# 登录会话的有效期（小时）。API token 可以不过期；登录会话必须有。
AUTH_SESSION_TTL_HOURS = float(os.getenv("AUTH_SESSION_TTL_HOURS", "12"))
