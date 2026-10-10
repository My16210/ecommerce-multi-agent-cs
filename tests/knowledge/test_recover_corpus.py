"""
语料还原的核心逻辑。

这里测的是恢复事故数据的那个算法本身（recover_corpus.py）。它容易被当成
一次性的临时脚本而没人看，但两件事让它值得有测试：

  1. **它将来还会被再跑一次** —— 语料再次丢失、或换一台机器重建时。
  2. **算错了不会报错，只会静默产出坏语料** —— 重叠取短了会把同一段复制两遍，
     取长了会吞掉内容，两种都"看着像正常文本"。这正是当年那场事故的同类：
     坏数据不会喊疼。

不碰网络、不碰向量库：全部是纯函数。
"""
import pytest

from ecom.knowledge.cli.recover_corpus import (
    chunk_match_score,
    diagnose,
    group_by_source,
    merge_chunks,
    overlap_size,
    resolve_target,
)

# --------------------------------------------------------------------------
# overlap_size：重叠长度的判定
# --------------------------------------------------------------------------


def test_overlap_size_finds_exact_suffix_prefix():
    assert overlap_size("今天天气很好", "很好我们去公园") == 2


def test_overlap_size_returns_zero_when_nothing_matches():
    assert overlap_size("今天天气很好", "我们去公园") == 0


def test_overlap_size_takes_the_longest_match_not_the_shortest():
    """必须取最长

    "abab" 的后缀和 "abab..." 的前缀在长度 1、2、4 上都成立。取 1 或 2 都会
    让重叠部分被重复保留——症状是文本里凭空多出一截重复内容，肉眼很难发现。
    """
    assert overlap_size("xyabab", "ababzw") == 4


def test_overlap_size_handles_one_containing_the_other():
    """整块被包含时，上限是较短的那个，不能越界"""
    assert overlap_size("ab", "abcd") == 2


# --------------------------------------------------------------------------
# merge_chunks：拼接
# --------------------------------------------------------------------------


def _chunks(specs):
    """specs 形如 [(chunk_id, text), ...]"""
    return [
        {"text": text, "source": "a.md", "chunk_id": chunk_id}
        for chunk_id, text in specs
    ]


def test_merge_removes_the_overlap():
    chunks = _chunks([(0, "今天天气很好"), (1, "很好我们去公园")])
    text, overlaps, zeros = merge_chunks(chunks, {})
    assert text == "今天天气很好我们去公园"
    assert overlaps == [2]
    assert zeros == []


def test_merge_uses_separator_only_when_there_is_no_overlap():
    """有重叠的接缝不能被填东西，否则重叠区会被撑开"""
    chunks = _chunks([(0, "甲"), (1, "乙")])
    text, overlaps, zeros = merge_chunks(chunks, {1: "\n\n"})
    assert text == "甲\n\n乙"
    assert overlaps == [0]
    assert zeros == [1]


def test_merge_sorts_by_chunk_id_not_input_order():
    """all_chunks 不保证顺序（Milvus 尤其），拼接必须自己按 chunk_id 排"""
    shuffled = _chunks([(2, "丙"), (0, "甲"), (1, "乙")])
    text, _overlaps, _zeros = merge_chunks(shuffled, {})
    assert text == "甲乙丙"


def test_merge_of_a_single_chunk_is_itself():
    text, overlaps, zeros = merge_chunks(_chunks([(0, "只有一块")]), {})
    assert (text, overlaps, zeros) == ("只有一块", [], [])


# --------------------------------------------------------------------------
# chunk_match_score：还原保真度的判据
# --------------------------------------------------------------------------


def test_re_split_of_an_exact_document_reproduces_every_chunk():
    """正文就是原文时，重切应当逐块相同——判据本身的自检

    这条保证的是"重切复现率"这个指标有意义：它确实能在拼对时给满分。
    """
    from ecom.knowledge.document_loader.splitter import split_text

    body = "第一段内容。\n\n第二段内容。\n\n" * 60
    chunks = _chunks(list(enumerate(split_text(body))))
    matched, rechunked = chunk_match_score(body, chunks)

    assert matched == len(chunks)
    assert rechunked == len(chunks)


# --------------------------------------------------------------------------
# diagnose：拼接前的完整性检查
# --------------------------------------------------------------------------


def test_diagnose_flags_a_gap_in_chunk_ids():
    problems = diagnose(_chunks([(0, "甲"), (1, "乙"), (5, "丙")]))
    assert any("不连续" in problem for problem in problems)


def test_diagnose_flags_duplicate_chunk_ids():
    problems = diagnose(_chunks([(0, "甲"), (0, "乙")]))
    assert any("重复" in problem for problem in problems)


def test_diagnose_flags_a_missing_head():
    problems = diagnose(_chunks([(3, "甲"), (4, "乙")]))
    assert any("从 0 开始" in problem for problem in problems)


def test_diagnose_flags_blank_chunks():
    problems = diagnose(_chunks([(0, "甲"), (1, "   ")]))
    assert any("空白" in problem for problem in problems)


def test_diagnose_is_silent_on_a_clean_source():
    assert diagnose(_chunks([(0, "甲"), (1, "乙"), (2, "丙")])) == []


# --------------------------------------------------------------------------
# resolve_target：落盘路径
# --------------------------------------------------------------------------


def test_resolve_target_keeps_subdirectories():
    """reviewed/ 下的文档，source 就是 "reviewed/xxx.md" """
    target = resolve_target("reviewed/某文档.md")
    assert target.name.endswith("某文档.md")
    assert target.parent.name == "reviewed"


def test_resolve_target_rejects_path_traversal():
    """source 是库里的数据，不能当成可信路径直接拼

    没有这道关卡，一条 source = "../../../某文件" 的脏数据就能把文件写到
    documents 目录外面去。
    """
    with pytest.raises(ValueError):
        resolve_target("../../etc/passwd")


def test_resolve_target_rejects_a_bare_directory():
    with pytest.raises(ValueError):
        resolve_target(".")


# --------------------------------------------------------------------------
# group_by_source
# --------------------------------------------------------------------------


def test_group_by_source_keeps_every_chunk():
    chunks = _chunks([(0, "甲"), (1, "乙")]) + [
        {"text": "丙", "source": "b.md", "chunk_id": 0}
    ]
    grouped = group_by_source(chunks)
    assert set(grouped) == {"a.md", "b.md"}
    assert len(grouped["a.md"]) == 2
    assert len(grouped["b.md"]) == 1
