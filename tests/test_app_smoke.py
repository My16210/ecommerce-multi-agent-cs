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
        "divider metric dataframe image code write text expander chat_message"
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
def app(monkeypatch):
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
    return module


def _render_all_tabs(module):
    module.render_agent_tab()
    module.render_order_tab()
    module.render_analytics_tab()
    module.render_knowledge_tab()


def _calls_named(name):
    return [c for c in CALLS if c[0] == name]


def test_the_module_imports_and_every_tab_renders(app):
    """四个 Tab 在**空状态**下都能渲染完，不抛异常。

    这一条抓的是最普通也最容易犯的错：Tab 之间互不认识、某个共享辅助函数
    签名改了而调用点没改、`st.session_state` 的键名打错（第一次跑是 KeyError，
    第二次跑才对——所以这里连跑两遍）。
    """
    _render_all_tabs(app)          # 第一遍：什么都没有
    app.main()                     # 第二遍：四个 Tab 一起画（走的是真入口）
    _render_all_tabs(app)          # 第三遍：session_state 已经有东西了

    assert _calls_named("title"), "连标题都没画出来，main() 大概没跑完"


def test_the_decision_chain_renders_a_failed_worker(app):
    """智能客服 Tab 的决策链：成功的、失败的、没答上的三种 worker 都要能显示。

    main.py -v 有测试钉着，界面这份是它的第二个视图——**两边必须显示同一件事**。
    这里用一个失败的 knowledge worker 把 `错误：...` 那条支路逼出来。
    """
    app.st.session_state["agent_history"] = [{"role": "user", "content": "订单 52644 到哪了"}]
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
    app.render_agent_tab()

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
    app.render_order_tab()

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
    app.render_knowledge_tab()

    assert _calls_named("warning"), "空向量库时应当给出提示"
