"""统一界面 app.py 的冒烟测试：**塞一个假的 streamlit 进去，把它真跑一遍**。

为什么需要这个，以及它证明了什么、没证明什么：

    本机（以及任何没装 `--extra ui` 的环境）都没有 streamlit，`import app`
    直接 ImportError。于是 app.py 成了整个项目里**唯一一个连 import 都进不去**
    的源文件——它写错一个函数名、抄错一个 `ecom.*` 模块路径，跑遍全套测试
    也不会有任何东西报警，直到有人真的去起那个界面。

    做法：往 sys.modules 里放一个记账用的假 streamlit，让 app.py 的模块级代码
    和四个 Tab 的渲染函数真跑一遍，只把"画到浏览器上"换成 append 到一个列表。
    真的会执行的是：每一条 `from ecom...` import、每一次 `query_order()`、
    每一步 `_render_decision_chain` 的取值和格式化。

    它**不能**替代"装好 streamlit 点一遍"：假模块不校验参数名是否真是 streamlit
    的（`st.markdown(name="x")` 这种它也照收）、不校验页面布局、不校验
    `st.session_state` 的真实语义。所以这个测试守的是"调用写得对不对"，
    不是"界面好不好看"。

    另外：这里**不联网、不调模型**——chat_input 恒返回 None、button 恒返回
    False，所有"需要用户点一下"的路径都不进入。数据看板那个真的会起子进程的
    路径也不进入（它需要模型）。跑的只有"渲染已有状态"这一段。
"""
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# 渲染过程中被调用过的 streamlit 接口，记成 (名字, 位置参数)
CALLS: list = []


class _Element:
    """所有上下文管理器和占位元素的万用替身。

    真实 streamlit 里 `with st.columns(3)[0]:`、`st.empty().markdown(...)`、
    `with st.spinner():` 是同一种用法模式（拿到东西、进入它、往上画），
    所以一个类就能全部顶掉。
    """

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getattr__(self, name):
        def _any(*args, **kwargs):
            CALLS.append((name, args))
            return _Element()

        return _any


def _recorder(name):
    def _call(*args, **kwargs):
        CALLS.append((name, args))
        return _Element()

    return _call


def _install_fake_streamlit():
    """把假的 streamlit 装进 sys.modules。**无条件覆盖**。

    覆盖是刻意的：如果哪天在装了 `--extra ui` 的环境里跑这套测试，真的
    streamlit 会先在别处被 import；它照样拦不住在这里换成假的——因为下面
    的 `_import_app()` 每次都把 `app` 从 sys.modules 里摘掉再重新导入。
    """
    st = types.ModuleType("streamlit")
    st.session_state = {}
    st.set_page_config = lambda **kwargs: None
    # cache_resource 的语义是"按参数缓存一份、进程内共用"。测试里直接返回原函数
    # 就够了——这里要验的是"装饰器用对没"，不是"缓存命中没"。
    st.cache_resource = lambda **kwargs: (lambda fn: fn)
    st.cache_data = lambda **kwargs: (lambda fn: fn)

    for _name in (
        "caption markdown warning info error success title header subheader "
        "divider metric dataframe image code write text expander chat_message "
        # 界面重做后新增的：决策链是 status(compact) 外套 expander(type=step)，
        # 侧边栏的导航块是 container(border=True)，间距用 space。
        "status container space badge"
    ).split():
        setattr(st, _name, _recorder(_name))

    st.columns = lambda spec, **kwargs: [
        _Element() for _ in (spec if isinstance(spec, list) else range(int(spec)))
    ]
    st.tabs = lambda labels, **kwargs: [_Element() for _ in labels]
    st.sidebar = _Element()
    st.empty = lambda *args, **kwargs: _Element()
    st.spinner = lambda *args, **kwargs: _Element()

    # ——这几个决定了"不会走进需要用户操作的那条路"——
    st.button = lambda *args, **kwargs: False
    st.chat_input = lambda *args, **kwargs: None
    # 图片上传器：这一套测试不验上传路径（那需要真模型），恒返回"没选文件"。
    st.file_uploader = lambda *args, **kwargs: None
    # st.pills **必须返回固定的 None，不能进上面那个 _recorder 名单**。
    # `_Element` 没有定义 `__bool__`（而 dunder 查找也不走 `__getattr__`），
    # 所以 `bool(_Element())` 恒为 True——一旦它返回 _Element，空态里那句
    # `if picked:` 就会把一张"元素"当成"用户点了建议卡"，跑进一条根本不该
    # 走的分支（会去调模型）。返回 None = 这一轮没人点。
    def _pills(*args, **kwargs):
        # 记录下来但**仍然返回 None**：现有测试要能看见"传了哪些选项"，
        # 而返回一个元素会让空态误以为用户点了卡（见上面那段注释）。
        CALLS.append(("pills", args))
        return None

    st.pills = _pills
    st.text_input = lambda *args, **kwargs: ""
    st.toggle = lambda label, value=False, **kwargs: value
    st.selectbox = lambda label, options, **kwargs: (options[0] if options else None)
    st.rerun = lambda: None
    st.stop = lambda: None

    sys.modules["streamlit"] = st
    return st


def _import_app(st):
    """每次重新 import app，保证它拿到的是**当前**这个假 streamlit。"""
    sys.modules.pop("app", None)
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import app

        return app
    finally:
        sys.path.remove(str(REPO_ROOT))


@pytest.fixture
def app(monkeypatch, tmp_path):
    st = _install_fake_streamlit()
    CALLS.clear()
    module = _import_app(st)

    # 向量库换成假的：真实 ChromaStore 会去 data/knowledge/vector_store/ 建目录，
    # 而那是 .gitignore 里的东西——一个界面测试不该依赖"本机恰好跑过 ingest"。
    # 换掉它之后，知识库 Tab 那条路验的是 app.py 自己的代码，不含向量库实现。
    class _FakeStore:
        def count(self):
            return 3

    monkeypatch.setattr(module, "_store", lambda: _FakeStore())

    # 会话库也换到临时目录。**这是落盘之后新增的必要一步**：智能客服 Tab
    # 现在会真的去开会话库（`_restore_agent_session`），不换的话这个测试
    # 会往仓库的 data/orchestrator/ 里写东西，"测试不依赖本机状态"就破了。
    monkeypatch.setattr(module.session, "SESSION_DB_PATH", str(tmp_path / "sessions.db"))
    module.session.reset_connection()
    yield module
    module.session.reset_connection()


def _render_all_views(module):
    """四个视图各画一遍。**名字叫 view 不叫 tab**——主区域已经不是一个 Tab 集合了，
    留着旧名字会让人以为结构没变。"""
    module.render_agent_view()
    module.render_order_view()
    module.render_analytics_view()
    module.render_knowledge_view()


def _calls_named(name):
    return [c for c in CALLS if c[0] == name]


def test_the_module_imports_and_every_view_renders(app):
    """四个视图在**空状态**下都能渲染完，不抛异常。

    这一条抓的是最普通也最容易犯的错：视图之间互不认识、某个共享辅助函数
    签名改了而调用点没改、`st.session_state` 的键名打错（第一次跑是 KeyError，
    第二次跑才对——所以这里连跑两遍）。
    """
    _render_all_views(app)         # 第一遍：什么都没有
    app.main()                     # 第二遍：侧边栏 + 当前视图（走的是真入口）
    _render_all_views(app)         # 第三遍：session_state 已经有东西了

    assert _calls_named("title"), "连标题都没画出来，main() 大概没跑完"


def test_the_decision_chain_renders_a_failed_worker(app):
    """智能客服 Tab 的决策链：成功的、失败的、没答上的三种 worker 都要能显示。

    main.py -v 有测试钉着，界面这份是它的第二个视图——**两边必须显示同一件事**。
    这里用一个失败的 knowledge worker 把 `错误：...` 那条支路逼出来。
    """
    # 置上会话号，`_restore_agent_session` 才会走"已经有了，不重建"那条路
    # ——否则它会把下面这份手工搭的 turns 覆盖掉。
    app.st.session_state["agent_session_id"] = "test-session"
    app.st.session_state["agent_turns"] = [
        {
            "question": "订单 52644 到哪了",
            "elapsed_ms": 1234,
            "final": {
                "answer": "（假答复）",
                "sources": ["订单库"],
                "plan": ["order"],
                "plan_reason": "问的是订单号",
                "router_source": "rule",
                "plan_confidence": 0.9,
                "decision": "answer",
                "decision_reason": "只有一个结论",
                "tokens": {"in": 10, "out": 20},
                "results": [
                    {"worker": "order", "ok": True, "insufficient": False, "round": 1,
                     "elapsed_ms": 12, "tokens": {"in": 5, "out": 6}, "answer": "已签收"},
                    {"worker": "knowledge", "ok": False, "insufficient": True, "round": 1,
                     "elapsed_ms": 900, "tokens": {}, "error": "RuntimeError: 编的"},
                ],
            },
        }
    ]
    app.render_agent_view()

    # worker 那一行走 markdown（带加粗），错误原文走 caption。断言时不分流派，
    # 全捞进来看——这里要验的是"这些信息到底有没有被显示"，不是"用了哪个接口"。
    shown = " ".join(str(c[1]) for c in CALLS)
    assert "失败" in shown, "worker 失败没被显示出来"
    assert "未答上" in shown, "insufficient 和 ok=False 被混为一谈了"
    assert "RuntimeError" in shown, "错误原文没有展开（排查时看不到原因）"
    assert "已签收" in shown, "成功那个 worker 的结论没显示"


def test_the_order_tab_handles_an_empty_market(app, monkeypatch):
    """订单库为空时给的是"先去建库"的提示，不是一张空表。

    这不是洁癖：一个空的 selectbox 会让整个 Tab 静默什么都不显示，看起来像
    界面坏了；而真实原因是库里没数据、要去跑 --init-db。
    """
    class _EmptyDB:
        backend = "sqlite"

        def query(self, sql, params=()):
            return []

    monkeypatch.setattr(app, "_db", lambda: _EmptyDB())
    app.render_order_view()

    assert _calls_named("warning"), "空库时应当给出提示"
    assert not _calls_named("dataframe"), "空库时不该画表格"


def test_the_knowledge_tab_says_so_when_there_is_nothing_indexed(app, monkeypatch):
    """向量库为空时提示去跑 ingest，而不是安静地走生成那一步。

    空检索还去调模型，会得到一个"资料里没有提到"的答复——用户完全没法分辨
    这是"知识库里真的没有"还是"索引根本没建"。这两件事必须分开说。
    """

    class _EmptyStore:
        def count(self):
            return 0

    monkeypatch.setattr(app, "_store", lambda: _EmptyStore())
    app.render_knowledge_view()

    assert _calls_named("warning"), "空向量库时应当给出提示"


def test_a_new_browser_session_starts_a_new_conversation(app):
    """**这条钉的是一个边界，不是一个功能。**

    界面这边**刻意不把会话号放进 URL**（它是凭证，进了 URL 就会进浏览器历史、
    代理日志、Referer 头）。所以新开一个标签页 = session_state 空 = 新会话，
    上一次的上下文接不上——这是设计的一部分，不是漏做。

    写成测试是因为"没做的能力"最容易被下一个人当成 bug 顺手补上，而补法
    （`st.query_params["session"] = sid`）正好是那个不该做的。
    """
    old = app.session.ensure_session("")
    app.session.append_turn(old, "上次聊过的", "上次答过的")
    app.st.session_state["agent_turns"] = [
        {"question": "上次聊过的", "final": {"answer": "上次答过的"}, "elapsed_ms": 1}
    ]

    # 新标签页：session_state 是空的（假 streamlit 里就是个 dict，直接清掉）
    app.st.session_state.clear()
    CALLS.clear()
    app.render_agent_view()

    assert app.st.session_state["agent_session_id"] != old
    assert app.st.session_state["agent_turns"] == []
    shown = " ".join(str(c[1]) for c in CALLS)
    assert "上次答过的" not in shown, "旧会话的内容不该出现在新会话里"


def test_the_model_gets_history_from_the_store_not_from_the_screen(app):
    """喂给模型的历史来自存储，**不是**屏幕上那个 `agent_turns`。

    这两份东西必须只是"同一件事的两个视图"，不能各存各的——否则清空时忘了
    清另一份，模型就会收到一段界面上已经看不见的上文。
    这里用屏幕上的 turns 是空的、但存储里有数据的错位状态把它逼出来。
    """
    sid = app.session.ensure_session("")
    app.session.append_turn(sid, "上一轮的问题", "上一轮的回答")

    app.st.session_state["agent_session_id"] = sid
    app.st.session_state["agent_turns"] = []     # 屏幕上什么都没有

    seen = {}

    def _fake_answer_stream(question, history):  # noqa: ANN001
        seen["history"] = history
        yield {"type": "start", "request_id": "req-test"}
        yield {"type": "delta", "text": "（假答复）"}
        yield {"type": "final", "state": {"answer": "（假答复）"}, "elapsed_ms": 1}

    app.answer_stream = _fake_answer_stream
    app.st.chat_input = lambda *a, **k: "这一轮的问题"

    app.render_agent_view()

    assert [m["content"] for m in seen["history"]] == ["上一轮的问题", "上一轮的回答"]
    # 这一轮也要落盘（否则下一轮就接不上了）
    assert len(app.session.history(sid)) == 4


def test_a_reroute_does_not_concatenate_the_old_streamed_text(app):
    """单 worker 的 token 是真流之后，重路由必须清掉已经画出的那一轮正文。

    否则屏幕上会出现「第一轮中间答复 + 第二轮最终答复」拼成的一段，
    而服务端的 final 只认第二轮——用户看到的东西和落盘的不是一回事。
    """
    sid = app.session.ensure_session("")
    app.st.session_state["agent_session_id"] = sid
    app.st.session_state["agent_turns"] = []

    def _fake_stream(question, history):  # noqa: ANN001
        yield {"type": "start", "request_id": "req-test"}
        yield {"type": "delta", "text": "第一轮的中间答复"}
        yield {"type": "reset"}
        yield {"type": "delta", "text": "第二轮的最终答复"}
        yield {"type": "final", "state": {"answer": "第二轮的最终答复"}, "elapsed_ms": 1}

    app.answer_stream = _fake_stream
    app.st.chat_input = lambda *a, **k: "这一轮的问题"

    app.render_agent_view()

    rendered = [str(c[1][0]) for c in CALLS if c[0] == "markdown" and c[1]]
    assert "第二轮的最终答复" in rendered
    assert not any("第一轮的中间答复" in text and "第二轮的最终答复" in text for text in rendered)


def test_a_new_session_starts_blank_but_does_not_delete_the_old_one(app):
    """「新建会话」是"换一个"，不是"销毁上一个"。

    反过来的话，用户点了这个按钮就等于把记录删了——而按钮上写的是"新建"。
    旧会话留给保留期和 `cli.session delete` 管。
    """
    old = app.session.ensure_session("")
    app.session.append_turn(old, "老问题", "老回答")
    app.st.session_state["agent_session_id"] = old
    app.st.session_state["agent_turns"] = [{"question": "老问题", "final": {"answer": "老回答"},
                                            "elapsed_ms": 1}]

    app._start_new_session()

    assert app.st.session_state["agent_session_id"] != old
    assert app.st.session_state["agent_turns"] == []
    assert app.session.get_session(old) is not None, "旧会话被顺手删掉了"
    assert len(app.session.history(old)) == 2


# ——— 公网演示：入口门禁 / 隐藏后台视图 / 不诱导去问已停用的能力 ———

def _enable_demo(app, monkeypatch, code="let-me-in"):
    monkeypatch.setattr(app.demo, "DEMO_MODE", True)
    monkeypatch.setattr(app.demo, "DEMO_INVITE_CODE", code)


def test_demo_mode_shows_the_invite_gate_before_anything_else(app, monkeypatch):
    _enable_demo(app, monkeypatch)
    app.st.session_state.clear()
    CALLS.clear()

    app.main()

    shown = " ".join(str(c[1]) for c in CALLS)
    assert "只读演示环境" in shown
    assert not _calls_named("chat_input"), "没进邀请码就渲染了对话入口"


def test_demo_mode_without_a_configured_code_says_why(app, monkeypatch):
    _enable_demo(app, monkeypatch, code="")
    app.st.session_state.clear()
    CALLS.clear()

    app.main()

    shown = " ".join(str(c[1]) for c in CALLS)
    assert "DEMO_INVITE_CODE" in shown
    assert not _calls_named("chat_input")


def test_demo_mode_does_not_render_the_analytics_view(app, monkeypatch):
    """侧边栏已经不显示它了，但视图函数本身也要挡——那才是真正执行代码的地方。"""
    _enable_demo(app, monkeypatch)
    CALLS.clear()

    app.render_analytics_view()

    assert _calls_named("warning")
    assert not _calls_named("text_input"), "分析输入框不该被画出来"


def test_demo_sidebar_hides_the_direct_views_and_the_session_id(app, monkeypatch):
    """公网演示只留对话；会话号是凭证，不该挂在一个可截图的地方。"""
    _enable_demo(app, monkeypatch)
    app.st.session_state.clear()
    CALLS.clear()

    view = app.render_sidebar()

    assert view == app.DEFAULT_VIEW
    assert not _calls_named("pills"), "演示模式不该画视图选择器"
    assert not _calls_named("code"), "演示模式不该显示会话号"
    shown = " ".join(str(c[1]) for c in CALLS)
    assert "只读演示" in shown


def test_demo_suggestions_do_not_offer_sales_analytics(app, monkeypatch):
    """点了必然办不到的建议卡，比没有建议卡更糟——它让演示第一步就落空。"""
    _enable_demo(app, monkeypatch)
    app.st.session_state.clear()
    app.st.session_state["agent_turns"] = []
    CALLS.clear()

    app.render_agent_view()

    shown = " ".join(str(c[1]) for c in CALLS)
    assert "哪个地区卖得最好" not in shown
    assert "退款一般多久到账" in shown
