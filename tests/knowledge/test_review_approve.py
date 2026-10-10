"""
approve 的落盘编排（review.approve_item）。

这段逻辑以前长在 cmd_approve 里，只有命令行能走；抽出来是为了让 Streamlit
界面能复用同一份。抽的时候最容易丢的是那个幂等分支——它看起来只是个
"已处理"的提示，实际守的是：**人工在落地 md 上做的修改不能被第二次 approve 盖掉**。
那份文件进库之后就是这个知识库的权威内容，悄悄重写等于让人白改一遍。
所以这一条必须有测试钉住，不能靠"UI 那边没人会点两次"。

全程离线：队列指到 tmp_path 的临时 SQLite，落地目录也指到 tmp_path。
"""
import pytest

from ecom.knowledge.cli import review
from ecom.knowledge.review import queue


@pytest.fixture
def env(monkeypatch, tmp_path):
    """隔离两头：队列库文件 + 落地目录，都不碰真实 data/"""
    monkeypatch.setattr(queue, "REVIEW_DB_PATH", str(tmp_path / "queue.db"))
    # 必须 patch review 模块自己的 REVIEWED_DOCS_PATH：
    # approve_item 读的是它所在模块的那个模块级名字，patch settings 没用。
    monkeypatch.setattr(review, "REVIEWED_DOCS_PATH", str(tmp_path / "reviewed"))
    queue.reset_connection()
    yield tmp_path / "reviewed"
    queue.reset_connection()


def _enqueue(question="Milvus 的 COSINE 和 Chroma 的距离有什么区别？"):
    result = {
        "answer": "Milvus 返回相似度，Chroma 返回距离。",
        "contexts": [{"text": "x", "source": "a.md", "chunk_id": 0, "score": 0.4}],
        "web_contexts": [],
        "sources": ["a.md"],
    }
    verdict = {"worthy": True, "category": "kb_gap", "score": 0.8, "reason": "理由"}
    row_id, _ = queue.enqueue(question, result, verdict)
    return row_id


def test_approve_writes_markdown_and_marks_approved(env):
    row_id = _enqueue()

    result = review.approve_item(row_id, note="已核对")

    assert result["ok"] is True
    assert result["already"] is False

    # 文件名里带当天日期，别去猜全名，直接按目录取
    files = list(env.glob("*.md"))
    assert len(files) == 1

    content = files[0].read_text(encoding="utf-8")
    assert "Milvus 的 COSINE" in content          # 问题进了标题
    assert "已核对" in content                     # 审核备注进了注释
    assert str(files[0]) == result["path"]

    item = queue.get_item(row_id)
    assert item["status"] == "approved"
    assert item["doc_path"] == str(files[0])
    assert item["review_note"] == "已核对"


def test_second_approve_returns_already_and_keeps_human_edit(env):
    """第二次 approve 必须原样返回、不重写——人工改过的内容不能被盖掉"""
    row_id = _enqueue()
    review.approve_item(row_id)
    path = list(env.glob("*.md"))[0]

    # 模拟"人工打开文件读一遍、该改的地方改掉"
    path.write_text("人工改过的权威内容", encoding="utf-8")

    again = review.approve_item(row_id, note="第二次")

    assert again["ok"] is True
    assert again["already"] is True
    assert again["path"] == str(path)
    assert path.read_text(encoding="utf-8") == "人工改过的权威内容"
    # 备注也不能被第二次调用改掉
    assert queue.get_item(row_id)["review_note"] is None


def test_approve_missing_id_fails_without_touching_disk(env):
    result = review.approve_item(99999)

    assert result["ok"] is False
    assert "99999" in result["message"]
    assert not env.exists()
