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
        "src/components/AgentConsole.tsx",
        "src/components/Composer.tsx",
        "src/components/Icon.tsx",
        "src/components/Markdown.tsx",
        "src/components/SettingsDialog.tsx",
        "src/components/Sidebar.tsx",
        "src/components/TracePanel.tsx",
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


def test_agent_console_polls_the_queue_and_the_open_ticket():
    """客服台不能只靠"切 Tab"刷新：新工单和用户新消息都会自己出现。

    这不是 WebSocket，代价是最多晚 5 秒；但没有长连接、没有重连状态机。
    """
    source = _read("src/components/AgentConsole.tsx")
    assert "setInterval" in source
    assert "fetchTickets(token, tab)" in source
    assert "fetchTicketDetail(selectedId, token)" in source
    # 轮询只更新列表/详情，不能在刷新时把客服正在打的草稿清掉。
    assert "setDraft(\"\")" not in source.split("async function tick()")[1].split("}, [open")[0]


def test_model_output_is_not_rendered_as_html():
    # Markdown 渲染器是模型输出进 DOM 的唯一位置；它只生成 React 元素，
    # 不许有任何「拼 HTML 字符串再塞进节点」的路径。
    for relative in (
        "src/App.tsx",
        "src/api.ts",
        "src/components/Markdown.tsx",
    ):
        source = _read(relative)
        assert "dangerouslySetInnerHTML" not in source
        assert "innerHTML" not in source


def test_all_frontend_sources_avoid_html_injection_and_persistent_storage():
    # 组件拆分之后，安全约定不能只靠「记得检查 App.tsx」。
    checked = 0
    for path in sorted((FRONTEND / "src").rglob("*")):
        if path.suffix not in {".ts", ".tsx"}:
            continue
        source = path.read_text(encoding="utf-8")
        assert "dangerouslySetInnerHTML" not in source, path
        assert "innerHTML" not in source, path
        assert "localStorage.setItem" not in source, path
        assert "localStorage.getItem" not in source, path
        assert "window.localStorage" not in source, path
        checked += 1
    assert checked >= 9, "src/ 下的源码文件数不对，扫描可能没跑到"


def test_shell_wires_the_three_columns_and_session_store():
    # 三栏组件存在还不够：必须真的接到 App 上，会话列表键也必须落在
    # sessionStorage（按标签页），不能换成 localStorage。
    app_source = _read("src/App.tsx")
    for component in ("Sidebar", "Composer", "TracePanel", "Markdown", "SettingsDialog"):
        assert f'./components/{component}' in app_source
    assert '"ecom.conversations"' in app_source
    assert '"ecom.active_conversation"' in app_source
    # 思考过程单独一路：事件处理、展示组件、类型契约缺一不可。
    assert 'event.type === "thinking"' in app_source
    assert "ThinkingBlock" in app_source
    types_source = _read("src/types.ts")
    assert "ThinkingEvent" in types_source
    assert "thinking?" in types_source


def test_api_client_uses_sse_endpoint_and_auth_headers():
    source = _read("src/api.ts")
    assert '"/chat/stream"' in source
    assert '"/health"' in source
    assert '"/auth/login"' in source
    assert "X-API-Key" in source
    assert "X-Invite-Code" in source
    assert "Authorization" in source


def test_login_hydrates_history_from_the_server_not_only_session_storage():
    """关掉标签页后历史还在，靠的是登录后重新拉 `/sessions`。

    这几条是"少一行也不会报错"的接线：没有 `fetchSessions`，侧边栏就永远
    只有本标签页；没有 `loaded:false`，历史会话会被当成空对话显示欢迎页。
    """
    api_source = _read("src/api.ts")
    assert '"/sessions"' in api_source
    assert "fetchSessions" in api_source
    assert "fetchSession" in api_source
    assert "deleteSession" in api_source

    app_source = _read("src/App.tsx")
    assert "fetchSessions" in app_source
    assert "mergeServerSessions" in app_source
    assert "loaded: false" in app_source
    assert "IDENTITY_KEY" in app_source

    types_source = _read("src/types.ts")
    assert "SessionSummary" in types_source
    assert "SessionDetail" in types_source
    assert "loaded?" in types_source


def test_login_form_uses_the_login_endpoint_and_keeps_password_out_of_storage():
    app_source = _read("src/App.tsx")
    assert "doLogin" in app_source
    assert "login(" in app_source
    # 密码只放在组件 state 里，不写 sessionStorage
    assert "password" in app_source
    assert "writeStored" in app_source


def test_handoff_ui_is_wired_end_to_end():
    """转人工不只是一句文案：按钮、状态条、客服台、轮询缺一不可。

    少任何一段都不会报错——用户点了没反应、客服看不到队列、回复要手动刷新。
    """
    app_source = _read("src/App.tsx")
    assert "requestHandoff" in app_source
    assert "openHandoff" in app_source
    assert "syncTicket" in app_source
    assert "handoff-banner" in app_source
    assert "AgentConsole" in app_source
    # 角色单独存：它决定客服台按钮在不在。
    assert "ROLE_KEY" in app_source
    assert 'role === "agent"' in app_source
    # 人工消息要有独立样式，不能和机器人混在一起。
    assert 'message.role === "agent"' in app_source

    api_source = _read("src/api.ts")
    for path in ("/handoff", "/tickets", "/claim", "/reply", "/close"):
        assert path in api_source

    types_source = _read("src/types.ts")
    for name in ("HandoffInfo", "TicketInfo", "TicketDetail", "TicketSummary"):
        assert name in types_source
    assert '"agent"' in types_source

    console = _read("src/components/AgentConsole.tsx")
    assert "fetchTickets" in console
    assert "claimTicket" in console
    assert "replyTicket" in console
    assert "closeTicket" in console


def test_identity_changes_clear_the_local_conversation_cache():
    """换账号时本地缓存不能跟着走。

    服务端按 owner 过滤会话，但已经加载到内存里的正文在本地是无主的；
    退出 / 登录 / 手工换 token 这三条路少清一条，都会让上一个账号的对话
    留在下一个账号的侧边栏里。
    """
    source = _read("src/App.tsx")
    assert "function resetLocalHistory" in source
    assert source.count("resetLocalHistory();") >= 3

    reset_body = source.split("function resetLocalHistory", 1)[1].split("function ", 1)[0]
    for key in ("STORE_KEY", "ACTIVE_KEY", "LEGACY_SESSION_KEY"):
        assert f'writeStored({key}, "")' in reset_body


def test_logout_surfaces_that_a_long_lived_token_was_not_revoked():
    """`revoked: false` 不能翻译成"已安全退出"。

    长期 API token 没有登录会话，`/auth/logout` 吊销不了它；前端清掉本地
    凭证后仍必须告诉用户这枚 token 还有效，否则是一次假的安心。
    """
    api_source = _read("src/api.ts")
    app_source = _read("src/App.tsx")
    assert "interface LogoutResult" in api_source
    assert "alreadyInvalid" in api_source
    assert "长期 API token" in app_source
    assert "ecom.auth.cli rotate" in app_source


def test_caddy_sets_csp_and_denies_framing():
    """生产入口的安全头不能被改掉：CSP 防注入，frame-ancestors 防点击劫持。"""
    caddy = (FRONTEND.parent / "Caddyfile").read_text(encoding="utf-8")
    assert "Content-Security-Policy" in caddy
    assert "frame-ancestors 'none'" in caddy
    assert 'X-Frame-Options "DENY"' in caddy
    assert "Permissions-Policy" in caddy


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


def test_caddy_limits_request_body_size():
    """超大请求体应在边缘被挡住，而不是先读进应用再拒绝。"""
    caddy = (FRONTEND.parent / "Caddyfile").read_text(encoding="utf-8")
    assert "request_body" in caddy
    assert "max_size 8MB" in caddy
