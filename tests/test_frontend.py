"""
React 前端的**静态约定测试**。

这里不跑浏览器，也不跑 npm——那会把"离线、秒级"的测试套件变成需要下载
Chromium 的东西。能在这里钉住的是那些**错了不会报错、只会白屏或悄悄泄露**
的约定：

  · 构建产物的资源前缀必须和 FastAPI 的挂载路径一致（`/ui/`）
  · 密钥 / 会话号不许落 localStorage，也不许进 URL
  · 模型输出不许走 innerHTML
  · `api.py` 的挂载是"有就挂、没有就跳过"
"""
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="需要 uv sync --extra api")
pytest.importorskip("httpx", reason="TestClient 需要 httpx（dev 依赖组）")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
DIST = FRONTEND / "dist"


def _read(relative: str) -> str:
    return (FRONTEND / relative).read_text(encoding="utf-8")


def test_frontend_sources_are_present():
    for relative in (
        "index.html",
        "package.json",
        "vite.config.ts",
        "src/App.tsx",
        "src/api.ts",
        "src/main.tsx",
    ):
        assert (FRONTEND / relative).is_file(), f"缺少 frontend/{relative}"


def test_package_has_build_and_typecheck_scripts():
    package = json.loads(_read("package.json"))
    assert "build" in package["scripts"]
    assert "typecheck" in package["scripts"]
    assert "react" in package["dependencies"]


def test_vite_base_matches_fastapi_mount():
    config = _read("vite.config.ts")
    assert 'base: "/ui/"' in config
    # 开发时接口在 8000，前端在 5173：三条走代理，不靠给后端开 CORS。
    for path in ("/health", "/chat", "/trace"):
        assert f'"{path}"' in config


def test_index_is_chinese_and_has_root():
    html = _read("index.html")
    assert 'lang="zh-CN"' in html
    assert 'id="root"' in html


def test_session_and_credentials_stay_in_session_storage():
    app_source = _read("src/App.tsx")
    assert "sessionStorage" in app_source
    # 界面文案里会出现「不写 localStorage」这句话，所以查的是**调用**而不是字符串。
    assert "localStorage.setItem" not in app_source
    assert "localStorage.getItem" not in app_source
    assert "window.localStorage" not in app_source
    # 会话号是凭证：不许进 URL（query string / hash / history）。
    assert "location.search" not in app_source
    assert "URLSearchParams" not in app_source
    assert "pushState" not in app_source


def test_model_output_is_not_rendered_as_html():
    for relative in ("src/App.tsx", "src/api.ts"):
        source = _read(relative)
        assert "dangerouslySetInnerHTML" not in source
        assert "innerHTML" not in source


def test_api_client_uses_sse_endpoint_and_auth_headers():
    source = _read("src/api.ts")
    assert '"/chat/stream"' in source
    assert '"/health"' in source
    assert '"/auth/login"' in source
    assert "X-API-Key" in source
    assert "X-Invite-Code" in source


def test_login_form_uses_the_login_endpoint_and_keeps_password_out_of_storage():
    app_source = _read("src/App.tsx")
    assert "doLogin" in app_source
    assert "login(" in app_source
    # 密码只放在组件 state 里，不写 sessionStorage
    assert "password" in app_source
    assert "writeStored" in app_source


def test_mount_frontend_serves_index_from_temp_dist(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html><body>ok</body></html>", encoding="utf-8")

    app = FastAPI()
    assert api.mount_frontend(app, dist) is True

    response = TestClient(app).get("/ui/")
    assert response.status_code == 200
    assert "ok" in response.text


def test_mount_frontend_skips_when_not_built(tmp_path):
    app = FastAPI()
    assert api.mount_frontend(app, tmp_path / "missing") is False


@pytest.mark.skipif(not (DIST / "index.html").is_file(), reason="前端未构建（npm run build）")
def test_built_assets_use_the_ui_prefix():
    html = (DIST / "index.html").read_text(encoding="utf-8")
    # 页面能打开但白屏，通常就是这里和 base 不一致：资源在 /assets 而实际在 /ui/assets。
    assert 'src="/ui/assets/' in html
    assert 'href="/ui/assets/' in html
