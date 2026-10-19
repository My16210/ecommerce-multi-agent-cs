"""
会话内长期记忆的测试（`ecom/orchestrator/memory.py`）。

**整份文件不联网**：假模型走参数注入（`distill(sid, llm=FakeLLM(...))`），
embedding 用 `monkeypatch.setattr(memory, "get_embedding", fake)` 顶掉。

两个打桩落点值得说明：

1. **embedding 要打在 `memory` 模块上绑定的那个名字，不是 `embedder` 上的。**
   `memory.py` 是 `from ...embedder import get_embedding`，打 `embedder.get_embedding`
   改的是另一个名字，`memory` 里那份引用一点没动——**测试会安静地真的去联网**，
   而它看起来是绿的（因为网络通的时候结果也对）。
2. **统一用 `get_embedding`，不用 `get_query_embedding`。** 后者带 `lru_cache`，
   跨用例缓存不清会串——同一个问题在两条测试里拿到不同期望值时，红的会是
   第二条，而原因在第一条。
"""
import array

import pytest

from ecom.orchestrator import memory, session
from ecom.orchestrator.cli import memory as memory_cli
from tests.orchestrator.conftest import FakeLLM


@pytest.fixture
def store(tmp_path, monkeypatch):
    """每个用例一个独立的会话库文件（和 test_session.py 同一套隔离办法）。"""
    monkeypatch.setattr(session, "SESSION_DB_PATH", str(tmp_path / "sessions.db"))
    session.reset_connection()
    yield session
    session.reset_connection()


def _fill(turns: int, sid: str = "chat") -> str:
    """开一个会话并灌 `turns` 轮问答。返回会话号。"""
    sid = session.ensure_session(sid)
    for i in range(turns):
        session.append_turn(sid, f"问{i}", f"答{i}")
    return sid


def _spill(sid: str = "chat") -> tuple:
    """灌出一个**真有消息滑出窗口**的会话，返回 `(会话号, 水位线)`。

    水位线 = 最后一条滑出窗口的消息 id。按 `_window_size()` 现算，不写死——
    写死的话谁把 `HISTORY_TURNS` 调一下就红一片，而那种红说明不了代码有问题。
    """
    window = session._window_size()
    turns = window // 2 + 4
    return _fill(turns, sid), turns * 2 - window


def _fake_embedding(vector=None):
    """一个假的 `get_embedding`，顺带记调用次数。"""
    vector = [1.0, 0.0] if vector is None else vector

    def fake(texts):
        fake.calls += 1
        return [list(vector) for _ in texts]

    fake.calls = 0
    return fake


def _one_json(summary="用户为订单 52644 申请退货，尚未确认。", keywords=("退货", "52644")):
    import json

    return json.dumps({"summary": summary, "keywords": list(keywords)}, ensure_ascii=False)


# ——— 余弦：三个方向 ———

def test_cosine_measures_three_directions():
    """正交 0 / 同向 1 / 反向 -1。

    这三条钉的是**方向**，不是数值精度：把 `dot / (|a||b|)` 写成 `dot`（忘了归一化）
    时，正交和反向还能碰巧对，只有同向那条会红。
    """
    assert memory._cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert memory._cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert memory._cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_gives_zero_instead_of_raising_on_a_length_mismatch():
    """长度对不上只可能是"换过 embedding 模型"。让这一条排到最后，别把整次召回弄挂。"""
    assert memory._cosine([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0
    assert memory._cosine([], [1.0]) == 0.0
    assert memory._cosine([0.0, 0.0], [1.0, 0.0]) == 0.0  # 零向量没有方向


def test_pack_and_unpack_round_trip():
    """BLOB 存的是 float32，不是 Python float——这一条把"精度够用"这件事钉在测试里。"""
    vec = [0.5, -1.25, 3.0]
    assert memory._unpack(memory._pack(vec)) == pytest.approx(vec)
    assert isinstance(memory._pack(vec), bytes)  # 不引 numpy，走的是 stdlib array
    assert array.array("f", vec).tobytes() == memory._pack(vec)


# ——— 蒸馏 ———

def test_distill_writes_one_row_covering_everything_out_of_window(store, monkeypatch):
    sid, through = _spill()
    monkeypatch.setattr(memory, "get_embedding", _fake_embedding())

    result = memory.distill(sid, llm=FakeLLM(_one_json()))

    assert result["created"] is True
    assert result["through_message_id"] == through

    rows = store.memories_for(sid)
    assert len(rows) == 1
    assert "52644" in rows[0]["summary"]
    assert rows[0]["keywords"] == "退货，52644"
    assert rows[0]["embedding"]  # 向量真写进去了——NULL 向量等于"永远召不回"


def test_distill_does_not_summarize_what_is_still_in_the_window(store, monkeypatch):
    """窗口内的消息**不蒸馏**：下一轮 `history()` 会原样喂给模型，再压一遍是重复付费，
    而且模型会读到一段自己有完整原文的删节版。"""
    sid, through = _spill()
    monkeypatch.setattr(memory, "get_embedding", _fake_embedding())
    llm = FakeLLM(_one_json())

    memory.distill(sid, llm=llm)

    prompt = llm.calls[0][1][1]
    # 消息 id 奇偶成对：第 i 轮是 `问i`(id 2i+1) 和 `答i`(id 2i+2)。
    # 水位线 `through` 是偶数，所以它的正文是 `答{through//2 - 1}`。
    assert f"答{through // 2 - 1}" in prompt       # 最后一条滑出窗口的在
    assert f"问{through // 2}" not in prompt       # 第一条还在窗口里的不在
    assert f"答{through // 2}" not in prompt


def test_running_distill_twice_creates_only_one_memory(store, monkeypatch):
    """幂等。第二次跑的时候已经没有滑出窗口的消息了，报告"无可蒸馏"（**不是错误**）。"""
    sid, _ = _spill()
    monkeypatch.setattr(memory, "get_embedding", _fake_embedding())

    assert memory.distill(sid, llm=FakeLLM(_one_json()))["created"] is True
    second = memory.distill(sid, llm=FakeLLM(_one_json()))

    assert second["created"] is False
    assert second["reason"] == "无可蒸馏"
    assert len(store.memories_for(sid)) == 1


def test_two_distills_racing_on_the_same_watermark_keep_one_row(store, monkeypatch):
    """**并发蒸馏**：cron 撞上手动执行，两个进程读到同一个水位线、蒸馏同一批消息。

    它们各自看到的"滑出窗口的消息"是同一批——所以这里把 `out_of_window_messages`
    钉住返回同一批（模拟第二个进程在第一个写库之前就已经读完了），跑第二次蒸馏。
    `_lock` 是进程内的，挡不住跨进程；挡住它的是 `UNIQUE(session_id,
    through_message_id)` + `INSERT OR IGNORE`——**幂等由数据库保证，不是由锁保证**。
    """
    sid, through = _spill()
    rows = store.out_of_window_messages(sid, 10**6)
    monkeypatch.setattr(memory, "get_embedding", _fake_embedding())
    monkeypatch.setattr(session, "out_of_window_messages", lambda *a, **k: rows)

    first = memory.distill(sid, llm=FakeLLM(_one_json()))
    second = memory.distill(sid, llm=FakeLLM(_one_json("另一个进程的摘要")))

    assert first["created"] is True
    assert second["created"] is False
    assert second["reason"] == "已被蒸馏"
    memories = store.memories_for(sid)
    assert len(memories) == 1
    assert memories[0]["through_message_id"] == through


def test_a_short_session_distills_nothing(store, monkeypatch):
    """窗口都还没装满就没什么可蒸馏的。**这不是错误**——所以它也不该调模型。"""
    monkeypatch.setattr(memory, "get_embedding", _fake_embedding())
    llm = FakeLLM(_one_json())
    sid = _fill(session._window_size() // 2)

    result = memory.distill(sid, llm=llm)

    assert result == {"created": False, "through_message_id": 0, "messages": 0,
                      "reason": "无可蒸馏"}
    assert llm.call_count == 0


def test_a_model_that_returns_junk_does_not_advance_the_watermark(store, monkeypatch):
    """模型没给出可用摘要时**不写、水位线不推进**——下次修好还能重跑。

    要是写一条空记忆，它会永远占着水位线，把这段对话彻底吞掉：`out_of_window_messages`
    的下界是"已蒸馏到的最大 id"，推进了就再也读不出这批消息了。
    """
    sid, through = _spill()
    monkeypatch.setattr(memory, "get_embedding", _fake_embedding())

    result = memory.distill(sid, llm=FakeLLM("这个我总结不出来"))

    assert result["created"] is False
    assert result["reason"] == "模型没给出可用摘要"
    assert store.memories_for(sid) == []
    # 水位线没动：这批消息还在滑出窗口的那一段里，可重跑
    assert max(r["id"] for r in store.out_of_window_messages(sid, 10**6)) == through


def test_an_embedding_failure_is_raised_not_swallowed(store, monkeypatch):
    """embedding 挂了要**抛出去**，让 CLI 报失败、水位线不推进。

    绝不写一条没有向量的记忆：那条约等于"这条记忆存在但永远选不上"——`list` 里
    看得到它，用户会以为它在起作用。
    """
    sid, _ = _spill()

    def boom(_):
        raise RuntimeError("embedding 服务不可用")

    monkeypatch.setattr(memory, "get_embedding", boom)

    with pytest.raises(RuntimeError):
        memory.distill(sid, llm=FakeLLM(_one_json()))
    assert store.memories_for(sid) == []


def test_distill_without_a_session_id_is_a_no_op(store):
    assert memory.distill("")["reason"] == "没有会话号"


# ——— 召回 ———

def test_recall_on_a_session_without_memories_calls_no_embedding(store, monkeypatch):
    """**池子为空就一次 embedding 都不调。** 这条是硬约束，两个理由：

    老会话（没跑过 distill）每问一句都白花一次调用；以及**离线测试要能跑**——
    不早退的话，每个用真会话库的用例都会去连硅基流动。
    """
    sid = _fill(3)
    fake = _fake_embedding()
    monkeypatch.setattr(memory, "get_embedding", fake)

    assert memory.recall(sid, "订单 52644 到哪了") == []
    assert fake.calls == 0


def test_recall_ranks_by_cosine_and_honours_top_k(store, monkeypatch):
    """排序按余弦，条数按 `MEMORY_TOP_K`。用**构造好的向量**断言名次，不桩 `_cosine`。"""
    sid = _fill(3)
    store.add_memory(sid, "同向的", "", memory._pack([1.0, 0.0]), 1)
    store.add_memory(sid, "正交的", "", memory._pack([0.0, 1.0]), 2)
    store.add_memory(sid, "反向的", "", memory._pack([-1.0, 0.0]), 3)

    monkeypatch.setattr(memory, "get_embedding", _fake_embedding([1.0, 0.0]))
    monkeypatch.setattr(memory, "MEMORY_TOP_K", 2)

    hits = memory.recall(sid, "随便问问")

    assert [h["summary"] for h in hits] == ["同向的", "正交的"]
    assert hits[0]["score"] == pytest.approx(1.0)
    assert hits[1]["score"] == pytest.approx(0.0)


def test_recall_skips_a_memory_whose_vector_is_gone(store, monkeypatch):
    """没有向量的记忆**跳过**（不上榜，也不报错）。它会排到最后，用户看到的是列表里
    那条记不出来——但至少不会因为一条脏数据把整次召回弄挂。"""
    sid = _fill(3)
    store.add_memory(sid, "没有向量", "", None, 1)
    store.add_memory(sid, "有向量", "", memory._pack([1.0, 0.0]), 2)

    monkeypatch.setattr(memory, "get_embedding", _fake_embedding([1.0, 0.0]))

    assert [h["summary"] for h in memory.recall(sid, "随便问问")] == ["有向量"]


def test_recall_never_leaves_the_session(store, monkeypatch):
    """**会话内的记忆，不是用户画像。** 换个 session_id 就一条都召不回来。"""
    mine, _ = _spill("mine")
    other, _ = _spill("other")
    monkeypatch.setattr(memory, "get_embedding", _fake_embedding())
    memory.distill(mine, llm=FakeLLM(_one_json()))

    assert memory.recall(other, "退货") == []


# ——— 拼进历史 ———

def test_recalled_memories_are_prepended_to_the_real_history(store, monkeypatch):
    sid, _ = _spill()
    monkeypatch.setattr(memory, "get_embedding", _fake_embedding())
    memory.distill(sid, llm=FakeLLM(_one_json()))
    history = [{"role": "user", "content": "那能退吗"}]

    augmented = memory.augmented_history(sid, "那能退吗", history)

    assert [t["role"] for t in augmented] == ["memory", "user"]
    assert "52644" in augmented[0]["content"]
    assert augmented[1] is history[0]  # 真历史原样跟在后面，顺序没动


def test_augmenting_does_not_mutate_the_history_it_is_given(store, monkeypatch):
    """**每轮现拼、不落盘、不累积。**

    `main.py` 的交互模式自己维护一个内存 `history` 列表：要是这里往同一个列表
    前插，记忆就会**跨轮重复累积**（第 5 轮里有 5 份同样的摘要），而且看不出来。
    """
    sid, _ = _spill()
    monkeypatch.setattr(memory, "get_embedding", _fake_embedding())
    memory.distill(sid, llm=FakeLLM(_one_json()))
    history = [{"role": "user", "content": "那能退吗"}]

    memory.augmented_history(sid, "那能退吗", history)

    assert history == [{"role": "user", "content": "那能退吗"}]


def test_augmenting_without_memories_returns_the_history_unchanged(store):
    """没有记忆时原样交回去（**不是加一条空记忆，也不是加一条"暂无"**）。

    "没有"必须表现为**什么都没多**：多一条空内容进去，模型会看到一段
    `较早的对话摘要：（空）`，那比不带记忆更糟。
    """
    sid = _fill(3)
    history = [{"role": "user", "content": "你好"}]

    assert memory.augmented_history(sid, "你好", history) == history


def test_augmented_history_tolerates_an_empty_history(store, monkeypatch):
    sid, _ = _spill()
    monkeypatch.setattr(memory, "get_embedding", _fake_embedding())
    memory.distill(sid, llm=FakeLLM(_one_json()))

    assert [t["role"] for t in memory.augmented_history(sid, "退货", None)] == ["memory"]


# ——— 角色标签：记忆不能被当成"助手说过的话" ———

def test_the_memory_role_gets_its_own_label_not_the_assistant_one():
    """`ecom/knowledge/generator/answer.py` 里那句
    `"用户" if role == "user" else "助手"` 会把 `role="memory"` **静默当成助手**——
    等于告诉模型"这是你自己刚说过的话"，它会当成自己的承诺。三路映射必须显式。"""
    from ecom.knowledge.generator.answer import _format_history

    rendered = _format_history([
        {"role": "memory", "content": "用户要为订单 52644 退货，尚未确认"},
        {"role": "user", "content": "那能退吗"},
        {"role": "assistant", "content": "要看订单状态"},
    ])

    assert "较早的对话摘要：用户要为订单 52644 退货，尚未确认" in rendered
    assert "助手：要看订单状态" in rendered
    # 记忆那一段绝不能被印成"助手"
    memory_line = next(line for line in rendered.splitlines() if "52644" in line)
    assert not memory_line.startswith("助手")


def test_an_unknown_role_is_labelled_as_unknown_not_as_the_assistant():
    """认不出来就说认不出来，别替它编一个身份。"""
    from ecom.knowledge.generator.answer import _format_history

    assert "（未知角色）：嗯" in _format_history([{"role": "tool", "content": "嗯"}])


# ——— CLI ———

def test_distill_all_on_short_sessions_reports_nothing_and_exits_zero(store, capsys):
    """`distill --all` 把"无可蒸馏"当**正常**，退出码 0。

    要是它返回非 0，挂在 cron 上的蒸馏每天都会报警一次——报的还是"没有可蒸馏的"，
    这种警报的结局是被人忽略，然后真正的失败也一起被忽略。
    """
    _fill(session._window_size() // 2)

    args = memory_cli.build_parser().parse_args(["distill", "--all"])

    assert args.func(args) == 0
    assert "无可蒸馏" in capsys.readouterr().out


def test_distill_for_an_unknown_session_exits_non_zero(store, capsys):
    args = memory_cli.build_parser().parse_args(["distill", "--session", "查无此会话"])

    assert args.func(args) == 1
    assert "没有这个会话" in capsys.readouterr().err


def test_recall_dry_run_prints_the_pool_and_the_scores(store, monkeypatch, capsys):
    """`recall` 是 dry-run：它是唯一能看见"为什么没带记忆"的地方，
    所以它必须把池子大小、命中条数、每条余弦都打出来。"""
    sid, _ = _spill()
    monkeypatch.setattr(memory, "get_embedding", _fake_embedding())
    memory.distill(sid, llm=FakeLLM(_one_json()))

    args = memory_cli.build_parser().parse_args(["recall", "--session", sid, "退货"])

    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "池子里 1 条" in out
    assert "余弦" in out


def test_recall_dry_run_writes_nothing(store, monkeypatch, capsys):
    """dry-run 就是只读。它要是顺手蒸馏一下，这个命令就不能用来排查了。"""
    sid, _ = _spill()
    monkeypatch.setattr(memory, "get_embedding", _fake_embedding())

    args = memory_cli.build_parser().parse_args(["recall", "--session", sid, "退货"])
    args.func(args)

    assert store.memories_for(sid) == []
