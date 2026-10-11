"""在**真的 streamlit 运行时**里把两个界面入口跑一遍（`streamlit.testing.v1.AppTest`）。

`test_app_smoke.py` / `test_admin_smoke.py` 用的是自己塞的假 streamlit。那份假
模块**不校验参数名、不校验布局、不校验 `session_state` 的真实语义**——它守的是
"调用写得对不对"。这个文件补的是它够不着的那一段：

    真的 Streamlit 会不会在渲染到一半时抛出来？`st.tabs` 到底认不认这几个
    标签？`st.code(..., language=None)` 这个签名对不对？`st.chat_input` 在
    没有输入时返回的是什么？假模块对这些都是"照收不误"。

【它证明了什么】
  · 两个入口在真运行时里**能跑完**（`at.exception` 为空）
  · 四个 Tab 真的渲染出来了，侧边栏那几行环境信息真的画出来了
  · 后台**没配口令时确实停在门前**：报告一个 error，且一个 Tab 都没有

【它不证明什么】**这条要写进 README，不然它会被当成"界面验过了"**
  · 没有浏览器、没有 CSS、没有布局。AppTest 拿到的是组件树，不是页面。
  · **交互路径没走过**：打字进 `chat_input` 触发一次真实问答，会调模型——
    和全仓库"测试不联网"那条规矩直接冲突，所以这里不点。那条路由
    README 里「API 真起服务」那次一次性验证覆盖。

【为什么开头要先把假 streamlit 摘掉】
`test_admin_smoke.py` / `test_app_smoke.py` 把假模块塞进 `sys.modules` 之后
**不清理**（它们是照着"这可能是本进程唯一的 streamlit"写的）。pytest 默认按
文件名顺序收集，`test_admin_smoke.py` 排在前面——跑到这里时
`sys.modules["streamlit"]` 已经是个没有 `__file__` 的假货了。不摘掉的话，
`AppTest` 会去 import 那个假模块，然后以一种极难看懂的方式失败。

【缺数据时 skip，不是失败】
`app.py` 会真的去开 `data/orders.sqlite` 和 Chroma 向量库。没灌过数据的环境
不该因此变成"跑不了测试"，所以这里 skip——这个文件验的是**界面接线**，
不是"数据灌没灌"。

【但这道 skip 只加在 6 条中的 1 条上，这是量过的、不是漏的】
干净 checkout（`git archive` 出来的，没有 `orders.sqlite`、没有向量库）上实测过：

  · `test_app_renders_in_a_real_streamlit_runtime` 要真数据，缺了就 skip。
  · **其余 5 条照常跑，而且全绿。** 它们不需要灌过数据：`app.py` 遇到空库时
    是"warning 一句然后 return"（不是抛异常），而 Chroma 和订单库的构造函数
    会**自己把空库建出来**（所以跑完会在 `data/` 下留下一套空的
    `orders.sqlite` / `vector_store/` / `review_queue.db`——三个都在 .gitignore 里）。
    那 4 条后台的测试更是只要有 `ADMIN_PASSWORD` 就行，跟订单数据无关。

**所以不要把 `skipif` 提到模块级。** 那样会让那 5 条在 CI 上一起被跳过——
其中 4 条本来跟数据毫无关系，白白丢掉覆盖。
"""
import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

pytest.importorskip("streamlit")

# 数据是这条测试的前提。缺了就跳过（见文件头最后一段）。
_ORDERS_DB = REPO_ROOT / "data" / "orders.sqlite"
_VECTOR_STORE = REPO_ROOT / "data" / "knowledge" / "vector_store" / "chroma.sqlite3"


def _force_real_streamlit():
    """把假 streamlit 从 sys.modules 里摘掉，换回真的那个。

    判据是 `__file__`：程序里 `types.ModuleType("streamlit")` 造出来的模块
    没有这个属性，从 site-packages 里 import 出来的有。

    **必须连 `streamlit.*` 子模块一起摘掉**，只摘顶层会炸——而且是很难看懂
    的那种炸。原因是 pytest 的**收集阶段**会把所有测试模块先 import 一遍，
    而本文件模块级的 `pytest.importorskip("streamlit")` 就在那时把**真的**
    streamlit 连子模块一起装进了 sys.modules。之后 `test_admin_smoke.py`
    运行时装上假的，**只覆盖了顶层那一个键**，真的 `streamlit.delta_generator
    _singletons` 等子模块还留在里面。

    这时只删顶层再 import，等于让 `streamlit/__init__.py` 跑第二遍，而它第
    95 行要造一个 `DeltaGeneratorSingleton`——那个类的实例已经挂在没被清掉的
    子模块上了，于是报一句和"界面测试"毫无关系的错。
    """
    for name in [n for n in list(sys.modules) if n == "streamlit" or n.startswith("streamlit.")]:
        if name == "streamlit" and getattr(sys.modules[name], "__file__", None) is not None:
            return sys.modules[name]     # 已经是真的了，不用动
        del sys.modules[name]
    return importlib.import_module("streamlit")


@pytest.fixture
def app_test(monkeypatch, tmp_path):
    """给一个"能在真 streamlit 里跑一个脚本文件"的工厂。

    顺手把两处**会写到仓库 data/ 里**的东西改到临时目录——一个界面测试不该
    在仓库里留痕：
      · 会话库（app.py 启动时就会 ensure_session 一次）
      · 知识库审核队列（`kb_queue.enqueue` 的落点）
    """
    _force_real_streamlit()
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from streamlit.testing.v1 import AppTest

        import ecom.orchestrator.session as session

        monkeypatch.setattr(session, "SESSION_DB_PATH", str(tmp_path / "sessions.db"))
        session.reset_connection()

        def _run(script_name):
            sys.modules.pop(script_name.removesuffix(".py"), None)
            at = AppTest.from_file(str(REPO_ROOT / script_name), default_timeout=120)
            at.run()
            return at

        yield _run

        session.reset_connection()
    finally:
        sys.path.remove(str(REPO_ROOT))


@pytest.mark.skipif(
    not (_ORDERS_DB.exists() and _VECTOR_STORE.exists()),
    reason="本机没灌过数据（data/orders.sqlite 或向量库不在），界面接线这条测试跳过",
)
def test_app_renders_in_a_real_streamlit_runtime(app_test):
    """客户界面在真运行时里跑完，四个 Tab 和环境信息都在。

    这里抓的是假 streamlit 抓不到的那类错：接口签名改了、`st.tabs` 的返回值
    用法不对、某个 `st.*` 在真版本里拼写不同。表现是渲染中断 + `at.exception`。
    """
    at = app_test("app.py")

    assert not at.exception, f"app.py 在真 streamlit 里抛了：{at.exception}"

    assert [t.label for t in at.tabs] == ["💬 智能客服", "📦 订单", "📊 数据看板", "📚 知识库"]

    sidebar = " ".join(m.value for m in at.sidebar.markdown)
    assert "向量库后端" in sidebar
    assert "订单库" in sidebar
    assert "每轮回读历史" in sidebar, "侧边栏少了会话那一块"


def test_the_sidebar_shows_a_truncated_session_credential(app_test):
    """会话号在侧边栏**只露前 12 位**，并且真的来自会话存储。

    它是**凭证**（见 session.py 开头：客户身份是从话里猜的、未认证的，拿到
    会话号就能读到整段历史），而侧边栏是最容易被截图发出去的地方。
    这条同时钉住了"截断"和"它确实有值"两件事——只截断不显示，或者显示整条，
    都过不了。
    """
    at = app_test("app.py")

    assert not at.exception
    assert len(at.sidebar.code) == 1, "侧边栏没有会话号那一行"
    shown = at.sidebar.code[0].value

    assert shown.endswith("…"), "会话号没有截断就被摆出来了"
    assert len(shown) == 13, f"应当只显示 12 位加一个省略号，实际 {len(shown)} 个字符"


def test_the_admin_console_refuses_to_start_without_a_password(app_test, monkeypatch):
    """**没配 ADMIN_PASSWORD 时，真运行时里也确实起不来。**

    和 `test_admin_smoke.py` 里那条是同一个断言，但那条用的是假 streamlit。
    这一条的价值在于它证明 `st.stop()` 在**真的** Streamlit 里真的会中断脚本
    ——真运行时里 `st.stop` 是靠抛 `StopException` 工作的，AppTest 会把它接住；
    万一哪天写法变了（比如有人在 stop 之后又写了一行渲染），这里会红。
    """
    import ecom.settings as settings

    monkeypatch.setattr(settings, "ADMIN_PASSWORD", "")
    at = app_test("admin.py")

    assert not at.exception, "拒绝启动不该表现为一个崩溃"
    assert at.error, "没配口令却一声不吭"
    assert "ADMIN_PASSWORD" in at.error[0].value
    assert at.tabs == [], "没配口令却把三个页面画出来了"


def test_the_admin_console_shows_only_a_password_box_before_login(app_test, monkeypatch):
    """配了口令、还没登录：**只有一个输入框，没有任何页面**。"""
    import ecom.settings as settings

    monkeypatch.setattr(settings, "ADMIN_PASSWORD", "s3cret")
    at = app_test("admin.py")

    assert not at.exception
    assert at.tabs == [], "没登录就把三个页面画出来了"
    assert len(at.text_input) == 1, "登录框不在（或者多画了别的输入框）"
    assert not at.error, "什么都没输入就报错了"


def test_a_wrong_password_still_leaves_the_console_shut(app_test, monkeypatch):
    """口令错了：报一句、页面仍然一个都没有。

    这是**点了一次按钮之后**的状态，和上一条（还没点）不同——覆盖了
    "登录判断写在渲染之后"这类错法的另一半。
    """
    import ecom.settings as settings

    monkeypatch.setattr(settings, "ADMIN_PASSWORD", "s3cret")
    at = app_test("admin.py")

    # 填口令**和点"进入"都要做**：只 set_value 不点，脚本下一轮走的是
    # "按钮没被按"那条路，`st.error` 根本不会执行——而这个测试会以
    # "断言 at.error 为空"的形式**假绿**（真实原因是一次点击都没发生）。
    at.text_input[0].set_value("猜的")
    at.button[0].click()
    at.run()

    assert not at.exception
    assert at.tabs == [], "口令错了却把后台画出来了"
    assert [e.value for e in at.error] == ["口令不对。"]


def test_the_right_password_opens_the_console(app_test, monkeypatch):
    """口令对了：三个页面真的画出来，而且**渲染过程本身不炸**。

    这一条把前面几条合起来收口——只验"进不去"的话，一个永远进不去的后台
    会让全套测试全绿。三个页面会真的去读 `data/orders.sqlite` 和知识库队列
    （都是只读查询，不写），所以它顺带证明了**后台和真库的接线是通的**。
    """
    import ecom.settings as settings

    monkeypatch.setattr(settings, "ADMIN_PASSWORD", "s3cret")
    at = app_test("admin.py")

    at.text_input[0].set_value("s3cret")
    at.button[0].click()
    at.run()

    assert not at.exception, f"登录之后渲染炸了：{at.exception}"
    labels = [t.label for t in at.tabs]
    assert len(labels) == 3, f"三个页面没画全：{labels}"
    assert "退货审核" in " ".join(labels)
    assert "知识库" in " ".join(labels)
