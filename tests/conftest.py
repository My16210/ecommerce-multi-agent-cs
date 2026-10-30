"""跨测试的默认值。

身份层默认关闭：本机 `.env` 里如果配了 `AUTH_ENABLED=1`，`tests/test_api.py`
那批"没配 API_KEY 就放行"的用例会以一种和接口无关的方式红掉。测试要断言的是
**代码行为**，不是"这台机器的 .env 长什么样"，所以在这里统一关掉；
`tests/test_api_auth.py` 会在自己的 fixture 里再打开。
"""
import pytest

from ecom.auth import settings as auth_settings


@pytest.fixture(autouse=True)
def _auth_disabled_by_default(monkeypatch):
    monkeypatch.setattr(auth_settings, "AUTH_ENABLED", False)
