"""身份层的开关与存储位置。"""
import os
from pathlib import Path

from ecom.settings import DATA_DIR

# 部署环境。只有显式的 prod / production 才启用启动检查；默认 dev 保持
# "clone 下来就能跑"的体验。
APP_ENV = os.getenv("APP_ENV", "dev").strip().lower()
IS_PRODUCTION = APP_ENV in {"prod", "production"}

# 默认关闭：本地 CLI / Streamlit 演示不需要登录；对公网提供 API 时必须打开。
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}

# 是否信任反向代理写的 X-Forwarded-For / X-Real-IP。
# 默认关闭：只有当**确实**在可信代理后面时才打开；否则客户端可以自己伪造头，
# 把登录限流从"按 IP"变成"按我喜欢的任意字符串"。
TRUST_PROXY_HEADERS = os.getenv("TRUST_PROXY_HEADERS", "").strip().lower() in {
    "1", "true", "yes", "on"
}

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

# 审计保留期（天）。默认与会话保留期（SESSION_TTL_DAYS=30）的 3 倍，
# 给"事后追责"留出比用户内容更长的时间；<= 0 表示不自动清理。
AUDIT_RETENTION_DAYS = float(os.getenv("AUDIT_RETENTION_DAYS", "90"))


def production_problem(demo_mode: bool) -> str | None:
    """
    生产环境的启动检查：**默认配置必须是安全的那一边。**

    在 `APP_ENV=prod` 下，如果既没开身份层、也没开只读演示，就拒绝启动。
    否则一次"忘了配 AUTH_ENABLED"的部署会把 `/chat` 直接暴露给所有人——
    而它看起来和正常启动一模一样，没有任何症状。

    只读演示（`DEMO_MODE=1`）是显式选择：邀请码 fail closed、只读、预算账本，
    它接受"没有账号"这件事。所以 prod + demo 允许启动。
    """
    if not IS_PRODUCTION or AUTH_ENABLED or demo_mode:
        return None
    return (
        "APP_ENV=prod 但没有开启 AUTH_ENABLED，也没有 DEMO_MODE=1："
        "对外提供的 API / 界面将不需要任何身份。"
        "请设置 AUTH_ENABLED=1 并用 `python -m ecom.auth.cli set-password` 建账号；"
        "如果这是公网只读演示，请显式设置 DEMO_MODE=1 和 DEMO_INVITE_CODE。"
    )
