"""
`build_eval_set.py --sources` 的过滤。

它的作用只有一个：让不同的语料**各出一份题**，不要把难度不同的语料混进一份
评测集里——混在一起，两边的变化会互相抵消，看起来"什么都没变"。

不联网、不调模型：只测那个纯函数。
"""
from ecom.knowledge.cli.build_eval_set import filter_by_sources


def _chunk(source: str, chunk_id: int) -> dict:
    return {"text": f"{source} 的第 {chunk_id} 段", "source": source, "chunk_id": chunk_id}


CORPUS = [
    _chunk("LangGraph基础.md", 0),
    _chunk("售后政策.md", 0),
    _chunk("售后政策.md", 1),
    _chunk("物流与发票.md", 0),
]


def test_no_filter_keeps_everything():
    assert len(filter_by_sources(CORPUS, None)) == len(CORPUS)
    assert len(filter_by_sources(CORPUS, "")) == len(CORPUS)


def test_substring_matches_by_filename():
    kept = filter_by_sources(CORPUS, "售后政策")

    assert [c["chunk_id"] for c in kept] == [0, 1]
    assert {c["source"] for c in kept} == {"售后政策.md"}


def test_comma_separated_needles_are_any_of():
    """逗号是"或"：命中任一子串就算，方便一次圈住整套示例语料"""
    kept = filter_by_sources(CORPUS, "售后政策,物流与发票")

    assert {c["source"] for c in kept} == {"售后政策.md", "物流与发票.md"}


def test_full_filename_is_also_a_substring():
    """写全名和写片段都能用——不必让人记住"该写多长"""
    assert len(filter_by_sources(CORPUS, "物流与发票.md")) == 1


def test_blank_needles_are_ignored():
    """`, ,` 这种输入不该变成"全都不匹配"导致抽空"""
    assert len(filter_by_sources(CORPUS, " , , ")) == len(CORPUS)


def test_no_match_returns_empty():
    assert filter_by_sources(CORPUS, "不存在的文档") == []


def test_filter_does_not_mutate_the_input():
    before = list(CORPUS)
    filter_by_sources(CORPUS, "售后政策")

    assert CORPUS == before
