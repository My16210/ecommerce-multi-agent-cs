"""后台管理端 admin.py 的冒烟测试：假 streamlit + **一个会抛异常的 st.stop**。

和 `test_app_smoke.py` 是同一套路子（那边开头讲了为什么需要、证明了什么、
没证明什么，这里不重复），但有三处**必须不一样**，每一处都是被真实语义逼出来的：

1. **`st.stop` 和 `st.rerun` 在这里必须抛异常**。真 streamlit 里这两个函数
   靠抛异常中断脚本；`test_app_smoke.py` 里那份假实现返回 None——对 app.py
   够用（它没有"先渲染一半再停"的地方），对 admin.py **不够**。后台的全部
   保证就是"没通过口令就停在门前"，如果 `st.stop` 返回 None，脚本会**继续
   往下画**，于是"未登录看不到审批按钮"这条断言会变成一句空话。
   → 这里把它们做成抛 `_ScriptAbort`，未登录的用例**必须 assert 到抛出**，
     抛不出来就说明门禁没挡住。

2. **口令比对这个判断单独测，不经过 streamlit**。理由写在 `admin.check_password`
   的 docstring 里：假 streamlit 的 `CALLS` **不记录关键字参数**，
   `st.error("口令不对。")` 和 `st.error(msg)` 在记录里长得一模一样——
   用"断言出现了 st.error"来证明"错误口令被挡住了"，那个断言证明不了什么。

3. **所有 `st.*` 断言的参数一律走位置参数**（同上：`CALLS` 只记 `args`）。

这里**不连数据库**：`returns.*` / `kb_queue.*` / `query_order` 全部换成假的。
一个界面测试不该要求"本机恰好灌过数据"，也不该往仓库的 data/ 里写真东西。
"""
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# 渲染过程中被调用过的 streamlit 接口，记成 (名字, 位置参数)
CALLS: list = []


class _ScriptAbort(Exception):
    """真 streamlit 用异常中断脚本（st.stop / st.rerun 都是）。"""


class _Element:
    """所有上下文管理器和占位元素的万用替身（和 test_app_smoke 里那份同源）。"""

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


def _install_fake_streamlit(pressed_keys=(), inputs=None):
    """
    装一个假的 streamlit。

    `pressed_keys` —— 哪些按钮算被点了。admin.py 里**每一个** st.button 都带
        `key`（`进入` / `通过` / `拒绝` / `批准进知识库` / `退出`），所以这里可以
        精确地只让某一个"被点"，不用担心连带触发别的分支。
    `inputs` —— text_input / text_area 的返回值，按 key 给。
    """
    st = types.ModuleType("streamlit")
    st.session_state = {}
    st.set_page_config = lambda **kwargs: None

    for _name in (
        "caption markdown warning info error success title header subheader "
        "divider metric dataframe json code write text expander container"
    ).split():
        setattr(st, _name, _recorder(_name))

    def _recording_return(record_name, value_fn):
        """**记账之后返回一个值**。

        和 `_recorder` 的区别就在这个返回值上：`st.button` / `st.text_input`
        这类"从用户那儿取东西"的接口必须返回真东西（True / 一段文字），
        否则脚本走不到分支里；但它们的调用同样要进 `CALLS`——不然
        「未登录时连输入框都没画」这种断言就没法写。
        """

        def _call(*args, **kwargs):
            CALLS.append((record_name, args))
            return value_fn(*args, **kwargs)

        return _call

    st.columns = _recording_return(
        "columns",
        lambda spec, **kw: [_Element() for _ in (spec if isinstance(spec, list) else range(int(spec)))],
    )
    st.tabs = _recording_return("tabs", lambda labels, **kw: [_Element() for _ in labels])
    st.sidebar = _Element()
    st.empty = lambda *args, **kwargs: _Element()
    st.spinner = lambda *args, **kwargs: _Element()

    st.button = _recording_return("button", lambda *a, **k: k.get("key") in pressed_keys)
    st.text_input = _recording_return(
        "text_input", lambda *a, **k: (inputs or {}).get(k.get("key"), "")
    )
    st.text_area = _recording_return(
        "text_area", lambda *a, **k: (inputs or {}).get(k.get("key"), "")
    )
    st.selectbox = _recording_return(
        "selectbox", lambda label, options, **kw: (options[0] if options else None)
    )

    def _stop(*args, **kwargs):
        raise _ScriptAbort("stop")

    def _rerun(*args, **kwargs):
        raise _ScriptAbort("rerun")

    st.stop = _stop
    st.rerun = _rerun

    sys.modules["streamlit"] = st
    return st


def _import_admin(st):
    """每次重新 import admin，保证它拿到的是**当前**这个假 streamlit。"""
    sys.modules.pop("admin", None)
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import admin

        return admin
    finally:
        sys.path.remove(str(REPO_ROOT))


def _build(monkeypatch, *, password="s3cret", logged_in=False, pressed=(), inputs=None):
    """装好假 streamlit、导进 admin、把三个数据源全换成假的。返回 (admin, st)。"""
    st = _install_fake_streamlit(pressed_keys=pressed, inputs=inputs)
    CALLS.clear()
    module = _import_admin(st)

    monkeypatch.setattr(module, "ADMIN_PASSWORD", password)
    if logged_in:
        st.session_state["admin_ok"] = True

    # —— 数据源全部换成假的：不连库、不写盘、不要求本机灌过数据 ——
    monkeypatch.setattr(
        module.returns, "stats",
        lambda db=None: {"待审核": 0, "已通过": 0, "已拒绝": 0},
    )
    monkeypatch.setattr(module.returns, "list_returns", lambda status=None, limit=50, db=None: [])
    monkeypatch.setattr(module.kb_queue, "stats", lambda: {})
    monkeypatch.setattr(module.kb_queue, "list_items", lambda **kw: [])
    return module, st


def _fake_order(order_id="52644"):
    """一笔订单快照。字段要**给全**——`_render_order_snapshot` 直接下标取值，
    少一个键就是一个 KeyError，而那不是这个测试想验的东西。"""
    return {
        "order": {"order_id": order_id, "status": "COMPLETE", "amount": "99.0",
                  "created_at": "2025-12-01", "region": "北京"},
        "logistics": None,
    }


def _calls_named(name):
    return [c for c in CALLS if c[0] == name]


def _shown():
    """所有被画出去的文本拼成一串。断言"这句话到底有没有出现"时用它。"""
    return " ".join(str(a) for c in CALLS for a in c[1])


# ═══════════════════════════════════════════════════════════════════════════
# 一、口令比对：**不经过 streamlit** 的那一半
# ═══════════════════════════════════════════════════════════════════════════
def test_the_configured_password_is_accepted(monkeypatch):
    module, _ = _build(monkeypatch, password="s3cret")
    assert module.check_password("s3cret", "s3cret") is True


def test_a_wrong_password_is_rejected(monkeypatch):
    module, _ = _build(monkeypatch, password="s3cret")
    assert module.check_password("s3cre", "s3cret") is False
    assert module.check_password("s3cret ", "s3cret") is False


def test_an_empty_password_never_lets_anyone_in(monkeypatch):
    """**fail closed 的那一条，写在最底层。**

    没配置口令时返回 False，而不是"没配置就等于没门槛"。这一行写反了，
    整个门禁就是一个摆设——而界面上的表现完全一样（有个输入框、点了没反应）。
    """
    module, _ = _build(monkeypatch, password="")
    assert module.check_password("", "") is False
    assert module.check_password("随便什么", "") is False


def test_a_non_ascii_password_does_not_blow_up(monkeypatch):
    """中文口令是最自然会被设出来的东西，而 `compare_digest` 碰上非 ASCII 的
    **str** 会直接抛 TypeError（它只吃 bytes）。两侧都先 encode 才躲得过。"""
    module, _ = _build(monkeypatch)
    assert module.check_password("后台口令", "后台口令") is True
    assert module.check_password("后台口另", "后台口令") is False


# ═══════════════════════════════════════════════════════════════════════════
# 二、门禁挡住的是"整个界面"，不只是那句提示
# ═══════════════════════════════════════════════════════════════════════════
def test_without_a_password_the_console_refuses_to_start(monkeypatch):
    """没配 ADMIN_PASSWORD → 报错并**停下来**，三个页面一个都不画。

    断言 `_ScriptAbort` 是必须的：假 st.stop 若返回 None，脚本会接着往下画完
    整个后台，而这个测试会**照样绿**——那就变成一个什么都没测的测试。
    """
    module, st = _build(monkeypatch, password="")

    with pytest.raises(_ScriptAbort):
        module.main()

    assert _calls_named("error"), "没给任何提示就停了，用户不知道要配什么"
    assert not _calls_named("tabs"), "没配口令却把后台画出来了"
    assert not st.session_state.get("admin_ok")


def test_the_password_screen_hides_everything_before_it_passes(monkeypatch):
    """配了口令、还没登录：只画登录框，不画数据。

    这条挡的是最容易被写错的一种：把登录判断放在页面渲染**之后**，
    界面上看着有登录框，实际上数据已经查过一遍、甚至已经画到 DOM 里了。
    """
    module, _ = _build(monkeypatch, password="s3cret")

    with pytest.raises(_ScriptAbort):
        module.main()

    assert _calls_named("text_input"), "登录框都没有"
    assert not _calls_named("tabs"), "没登录就把三个页面画出来了"
    assert not _calls_named("subheader"), "没登录就渲染了页面内容"


def test_a_wrong_password_does_not_let_you_in(monkeypatch):
    module, st = _build(
        monkeypatch, password="s3cret", pressed=("admin_login",),
        inputs={"admin_pw": "猜的"},
    )

    with pytest.raises(_ScriptAbort):
        module.main()

    # 位置参数 —— CALLS 不记录关键字参数，st.error("口令不对。") 和 st.error(msg)
    # 在记录里长得一样，所以这里断的是**内容**而不是"出现过 st.error"。
    assert ("口令不对。",) in [c[1] for c in _calls_named("error")]
    assert not st.session_state.get("admin_ok")
    assert not _calls_named("tabs")


def test_the_right_password_gets_in(monkeypatch):
    module, st = _build(
        monkeypatch, password="s3cret", pressed=("admin_login",),
        inputs={"admin_pw": "s3cret"},
    )

    with pytest.raises(_ScriptAbort):   # st.rerun / st.stop 都会中断脚本
        module.main()

    assert st.session_state.get("admin_ok") is True
    assert not _calls_named("error")


# ═══════════════════════════════════════════════════════════════════════════
# 三、登录之后：三个页面都在，且只有退货那一页能写
# ═══════════════════════════════════════════════════════════════════════════
def test_the_console_renders_all_three_pages(monkeypatch):
    module, _ = _build(monkeypatch, logged_in=True)

    module.main()      # 登录之后不该再中断

    labels = " ".join(str(c[1]) for c in _calls_named("tabs"))
    assert "退货审核" in labels and "订单与物流" in labels and "知识库队列" in labels


def test_the_returns_page_offers_a_decision_on_a_pending_application(monkeypatch):
    module, _ = _build(monkeypatch, logged_in=True)
    monkeypatch.setattr(
        module.returns, "list_returns",
        lambda status=None, limit=50, db=None: [
            {"return_id": "R1", "order_id": "52644", "reason": "收到是坏的",
             "status": "待审核", "created_at": "2026-01-01 00:00:00",
             "decision": None, "reviewer": None, "review_note": None, "reviewed_at": None},
        ],
    )
    monkeypatch.setattr(module, "query_order", _fake_order)

    module.render_returns_page()

    assert "收到是坏的" in _shown(), "审核人看不到申请理由，只能凭感觉点"
    pressed_labels = [c[1][0] for c in CALLS if c[0] == "button"]
    assert "通过" in pressed_labels and "拒绝" in pressed_labels


def test_an_already_reviewed_application_shows_the_trace_and_no_buttons(monkeypatch):
    """审过的申请在**界面上也不能再审一次**。

    和 `returns.review_return` 的幂等是同一条语义的两层：那一层是"不会覆盖"，
    这一层是"连按钮都不给"。只做前者的话，审核人会看到一个能点的按钮、
    点完弹一句"已经审过了"——那是把一次无效操作伪装成一次操作。
    """
    module, _ = _build(monkeypatch, logged_in=True)
    monkeypatch.setattr(
        module.returns, "list_returns",
        lambda status=None, limit=50, db=None: [
            {"return_id": "R9", "order_id": "52644", "reason": "不想要了",
             "status": "已通过", "created_at": "2026-01-01 00:00:00",
             "decision": "已通过", "reviewer": "张三", "review_note": "看过照片了",
             "reviewed_at": "2026-01-02 00:00:00"},
        ],
    )
    monkeypatch.setattr(module, "query_order", _fake_order)

    module.render_returns_page()

    assert "张三" in _shown() and "看过照片了" in _shown(), "审核痕迹没显示出来"
    pressed_labels = [c[1][0] for c in CALLS if c[0] == "button"]
    assert "通过" not in pressed_labels and "拒绝" not in pressed_labels


def test_approving_sends_a_human_decision_and_only_a_human_decision(monkeypatch):
    """点"通过"送下去的是 `已通过`——**不是"待审核"**。

    这条钉的是后台这半边的措辞来源：decision 取自 `RETURN_STATUSES[1]`。
    真写成字符串字面量也行，但一旦哪天有人把按钮接到"改回待审核"上，
    那条路是**改不动**的（`review_return` 会抛 ValueError），
    而错误会在界面上变成一个没人看得懂的报错。
    """
    module, _ = _build(monkeypatch, logged_in=True, pressed=("pass_R1",),
                       inputs={"note_R1": "客户提供了照片", "who_R1": "张三"})
    monkeypatch.setattr(
        module.returns, "list_returns",
        lambda status=None, limit=50, db=None: [
            {"return_id": "R1", "order_id": "52644", "reason": "坏的",
             "status": "待审核", "created_at": "2026-01-01 00:00:00",
             "decision": None, "reviewer": None, "review_note": None, "reviewed_at": None},
        ],
    )
    monkeypatch.setattr(module, "query_order", _fake_order)

    seen = {}

    def _fake_review(return_id, decision, note=None, reviewer=None, db=None):
        seen.update(return_id=return_id, decision=decision, note=note, reviewer=reviewer)
        return {"ok": True, "already": False, "return_id": return_id,
                "status": decision, "message": "已通过"}

    monkeypatch.setattr(module.returns, "review_return", _fake_review)

    with pytest.raises(_ScriptAbort):    # 落完结论会 st.rerun
        module.render_returns_page()

    assert seen["return_id"] == "R1"
    assert seen["decision"] == "已通过"
    assert seen["note"] == "客户提供了照片"
    assert seen["reviewer"] == "张三"


def test_a_rejected_application_offers_a_second_chance(monkeypatch):
    """**被拒绝过的申请不在"待审核"列表里，而这不影响它重新申请。**

    后台这一层只是按状态过滤显示；真正决定"能不能再申请"的是
    `create_return` 里的 `BLOCKING_RETURN_STATUSES`（那边有专门的测试）。
    这里只钉一件事：后台**不会**因为"这笔订单被拒过"就把它藏起来或挡住
    ——界面上没有这个判断，判断只有一处。
    """
    module, _ = _build(monkeypatch, logged_in=True)
    seen = {}

    def _fake_list(status=None, limit=50, db=None):
        seen["status"] = status
        return []

    monkeypatch.setattr(module.returns, "list_returns", _fake_list)

    module.render_returns_page()

    assert seen["status"] == "待审核", "列表页默认看的应当是待审核那一档"


def test_the_kb_page_reuses_the_command_line_function(monkeypatch):
    """知识库那一页**一行新逻辑都不写**，直接调 CLI 那个 `approve_item`。

    这不是省事：入库要落 md、要改状态、要幂等（审过的 md 不能被重写），
    三条都在那个函数里。在这里"顺手重写一遍"就等于给向量库开了第二个写入口
    ——而 `ingest.py` 是唯一写入口这条约定，正是知识库那一整套设计的前提。
    """
    module, _ = _build(monkeypatch, logged_in=True, pressed=("kb_ok_7",),
                       inputs={"kb_note_7": "已核对"})
    monkeypatch.setattr(
        module.kb_queue, "list_items",
        lambda **kw: [{"id": 7, "question": "保修多久", "answer": "一年", "category": "售后",
                       "score": 0.9, "ask_count": 3, "kb_hits": 2}],
    )
    seen = {}

    def _fake_approve(item_id, note=None):
        seen.update(item_id=item_id, note=note)
        return {"ok": True, "path": "x.md", "already": False, "message": "已通过 #7"}

    monkeypatch.setattr(module.kb_review, "approve_item", _fake_approve)

    with pytest.raises(_ScriptAbort):    # 通过之后会 st.rerun
        module.render_kb_queue_page()

    assert seen == {"item_id": 7, "note": "已核对"}


def test_the_orders_page_only_reads(monkeypatch):
    """订单页是**只读**的：查得到就显示，查不到就说查不到。"""
    module, _ = _build(monkeypatch, logged_in=True,
                       inputs={"admin_order_id": "52644", "admin_customer_id": ""})
    monkeypatch.setattr(
        module, "query_order",
        lambda order_id: {"order": {"order_id": order_id, "status": "COMPLETE"},
                          "logistics": {"status": "已签收", "to_city": "北京",
                                        "updated_at": "2025-12-02"}},
    )

    module.render_orders_page()

    assert "52644" in _shown()
    assert "已签收" in _shown(), "物流那一块没画出来"


def test_a_missing_order_is_reported_not_raised(monkeypatch):
    """订单号输错时给一句人话，不是一整段 traceback。

    后台是给人用的：查不到订单是**用户的输入问题**，不是程序故障。
    """
    module, _ = _build(monkeypatch, logged_in=True, inputs={"admin_order_id": "00000"})

    def _boom(order_id):
        raise module.OrderNotFound("查不到")

    monkeypatch.setattr(module, "query_order", _boom)

    module.render_orders_page()

    assert _calls_named("warning"), "查不到订单却一声不吭"


def test_logging_out_locks_the_console_again(monkeypatch):
    module, st = _build(monkeypatch, logged_in=True, pressed=("admin_logout",))

    with pytest.raises(_ScriptAbort):
        module.main()

    assert st.session_state["admin_ok"] is False
