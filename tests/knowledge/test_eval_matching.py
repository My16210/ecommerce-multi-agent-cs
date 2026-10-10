"""
评测集的粒度与匹配规则。

source 级和 chunk 级的差别全在 `match_positions` 里：一个只要文件名对上，
一个要（文件名, 段号）都对上。这两种语义写错的症状是**数字静默失真**——
把 chunk 级误写成 source 级，Recall 会虚高到接近满分，看起来是"检索很好"，
实际上什么都没量到，而这正是做 chunk 级评测要解决的那个问题本身。
所以这条边界必须有测试守着。

不联网、不碰向量库：全是纯函数。
"""
from ecom.knowledge.utils.eval import load_eval_set, match_positions


def _hits(*pairs):
    return [
        {"text": "", "source": source, "chunk_id": chunk_id, "score": 0.0}
        for source, chunk_id in pairs
    ]


def test_source_level_matches_any_chunk_of_the_document():
    case = {"question": "q", "expected_source": "a.md"}
    assert match_positions(case, _hits(("b.md", 0), ("a.md", 7), ("a.md", 9))) == [2, 3]


def test_source_level_returns_every_position_for_the_precision_metric():
    """source 级故意返回**全部**命中位置，因为精度指标要数"召回的几条真相关\""""
    case = {"question": "q", "expected_source": "a.md"}
    assert len(match_positions(case, _hits(("a.md", 1), ("a.md", 2), ("a.md", 3)))) == 3


def test_chunk_level_requires_the_exact_chunk():
    case = {"question": "q", "expected_source": "a.md", "expected_chunk_id": 7}
    assert match_positions(case, _hits(("b.md", 0), ("a.md", 9), ("a.md", 7))) == [3]


def test_chunk_level_other_chunk_of_same_document_is_a_miss():
    """同文档的另一段**不算命中**——那正是 chunk 级存在的全部意义

    如果这里判成命中，chunk 级就退化回 source 级，候选又变回 3 份文档，
    Recall 再次虚高。
    """
    case = {"question": "q", "expected_source": "a.md", "expected_chunk_id": 7}
    assert match_positions(case, _hits(("a.md", 8), ("a.md", 6))) == []


def test_chunk_level_same_chunk_id_in_another_document_is_a_miss():
    """段号是各文档内部自增的，不同文档的同一段号毫无关系"""
    case = {"question": "q", "expected_source": "a.md", "expected_chunk_id": 7}
    assert match_positions(case, _hits(("b.md", 7))) == []


def test_chunk_level_can_only_hit_once():
    """一段只有一个标准答案，命中位置最多一个"""
    case = {"question": "q", "expected_source": "a.md", "expected_chunk_id": 7}
    assert match_positions(case, _hits(("a.md", 7), ("b.md", 7))) == [1]


def test_load_eval_set_returns_empty_when_the_file_is_missing(tmp_path):
    """没生成过评测集是**正常状态**，不该抛异常让人去猜（调用方自己给提示）"""
    assert load_eval_set(str(tmp_path / "还没生成.jsonl")) == []


def test_load_eval_set_reads_jsonl_and_skips_blank_lines(tmp_path):
    path = tmp_path / "eval_set.jsonl"
    path.write_text('{"question": "q1"}\n\n{"question": "q2"}\n', encoding="utf-8")
    assert [case["question"] for case in load_eval_set(str(path))] == ["q1", "q2"]
