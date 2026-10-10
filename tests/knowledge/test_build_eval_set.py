"""
出题脚本的校验规则。

`reject_reason` 是这份评测集质量的**全部**防线，而它拦不住的坏数据不会报错——
只会让 A/B 出来的结论往"混合检索更好"的方向偏（照抄的改写型题目让 BM25 白赢），
而且偏得很好看。所以每条规则都要有用例钉住，**尤其是它会误伤什么**：
一条过严的规则比没有规则更糟，它会把好题当坏数据丢掉，还丢得毫无规律。

不联网、不花钱：这里只喂字符串给纯函数，不调任何模型。
"""
import pytest

from ecom.knowledge.cli.build_eval_set import (
    KIND_PARAPHRASE,
    KIND_TERM,
    MAX_QUESTION_CHARS,
    MAX_TERM_CHARS,
    CorpusTerms,
    has_verbatim_run,
    leaked_terms,
    normalize,
    parse_batch,
    reject_reason,
    sample_chunks,
)


def _corpus_docs():
    """一份受控的小语料：让"常见词"和"罕见词"的 df 都落在预期区间里

    "向量"故意出现在 6 段里（超过 MAX_TERM_DF=5，算常见），
    "SqliteSaver"/"BM25" 各只出现 1 次（算罕见）。
    """
    return [f"向量检索的第 {i} 种写法" for i in range(6)] + [
        "SqliteSaver 把对话状态存进 SQLite 文件。",
        "BM25 是词面匹配算法，靠精确词频打分。",
    ]


@pytest.fixture
def corpus():
    return CorpusTerms(_corpus_docs())


@pytest.fixture
def vector_chunk():
    return {"source": "a.md", "chunk_id": 0, "text": "向量检索的第 0 种写法"}


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------


def test_normalize_drops_case_space_and_punctuation():
    assert normalize("BGE-M3（模型）") == normalize("bge m3 模型")
    assert normalize("SqliteSaver 是啥") == "sqlitesaver是啥"


def test_verbatim_run_catches_a_copied_phrase():
    text = "检索增强生成（RAG）是把外部知识引入大模型的方法。"
    # 6 字短语"检索增强生成"原样重合
    assert has_verbatim_run("什么是检索增强生成", text)
    # 真改写：一个 6 字重合都没有
    assert not has_verbatim_run("外部资料怎么喂给模型", text)


def test_short_question_is_still_checked():
    """问题比阈值还短时，阈值要降到问题长度

    否则"什么是检索"这种整句照抄的短问题**根本无从比对**，直接漏过。
    """
    assert has_verbatim_run("什么是检索", "什么是检索增强生成？")


def test_df_counts_documents_not_occurrences():
    corpus = CorpusTerms(["向量检索", "向量库", "BM25 词面匹配", "向量"])
    assert corpus.df("向量") == 3      # 数的是"几段里有"，不是"出现几次"
    assert corpus.df("BM25") == 1
    assert corpus.df("压根不存在") == 0


# --------------------------------------------------------------------------
# 罕见词泄漏
# --------------------------------------------------------------------------


def test_leaked_terms_flags_a_rare_term_from_this_chunk(corpus):
    question = "SqliteSaver 是怎么把对话存下来的"
    assert leaked_terms(question, "SqliteSaver 把对话状态存进 SQLite 文件。", corpus)


def test_leaked_terms_ignores_common_words(corpus, vector_chunk):
    """常见词不拦。"向量"满语料都是，问题里带着它不算作弊。"""
    assert leaked_terms("向量检索是怎么实现的", vector_chunk["text"], corpus) == []


def test_leaked_terms_ignores_rare_words_not_in_this_chunk(corpus, vector_chunk):
    """罕见词必须**同时**出现在这一段里才算泄漏

    少了这个条件，任何生僻词都会触发拒绝——问题里写个语料里没有的词，
    跟这一段根本没关系，凭什么判它泄漏。
    """
    assert leaked_terms("SqliteSaver 是什么", vector_chunk["text"], corpus) == []


# --------------------------------------------------------------------------
# 改写型
# --------------------------------------------------------------------------


def test_paraphrase_accepts_a_natural_question(corpus, vector_chunk):
    assert reject_reason(
        "怎么把一段文字变成能比距离的数组", KIND_PARAPHRASE, None, vector_chunk, corpus
    ) is None


def test_paraphrase_rejects_a_verbatim_phrase(corpus):
    chunk = {"text": "检索增强生成（RAG）是把外部知识引入大模型的方法。"}
    reason = reject_reason("什么是检索增强生成", KIND_PARAPHRASE, None, chunk, corpus)
    assert reason[0] == "照抄原文"


def test_paraphrase_rejects_a_leaked_short_term(corpus):
    """4 个字的术语躲得过"连续 6 字重合"那条，得靠罕见词这条兜住

    这条是两条规则**互补**的证据：只有连续重合规则的话，"BM25 这种算法怎么打分"
    会一路放行，而 BM25 恰恰是 BM25 检索最吃香的那种查询。
    """
    chunk = {"text": "BM25 是词面匹配算法，靠精确词频打分。"}
    reason = reject_reason("BM25 这种算法是怎么打分的", KIND_PARAPHRASE, None, chunk, corpus)
    assert reason[0] == "改写型混进了术语"


# --------------------------------------------------------------------------
# 术语型
# --------------------------------------------------------------------------


def test_term_accepts_a_rare_term_in_question_and_chunk(corpus):
    chunk = {"text": "BM25 是词面匹配算法，靠精确词频打分。"}
    assert reject_reason("BM25 靠什么打分？", KIND_TERM, "BM25", chunk, corpus) is None


def test_term_rejects_a_term_that_is_not_rare(corpus, vector_chunk):
    reason = reject_reason("向量检索是什么？", KIND_TERM, "向量", vector_chunk, corpus)
    assert reason[0] == "术语不够罕见"


def test_term_rejects_a_term_missing_from_the_question(corpus):
    """模型挑了个术语，却把问题写得不需要这个术语——词面信号就没了"""
    chunk = {"text": "BM25 是词面匹配算法，靠精确词频打分。"}
    reason = reject_reason("这个算法靠什么打分？", KIND_TERM, "BM25", chunk, corpus)
    assert reason[0] == "术语没写进问题"


def test_term_rejects_a_term_not_present_in_the_chunk(corpus, vector_chunk):
    reason = reject_reason("BM25 是什么？", KIND_TERM, "BM25", vector_chunk, corpus)
    assert reason[0] == "术语不在本段里"


@pytest.mark.parametrize("term", [
    'filter="id in [463480757150366907, 463480757150366908]"',   # 真抽检抓到过的
    "a" * (MAX_TERM_CHARS + 1),
])
def test_term_rejects_a_code_expression_masquerading_as_a_term(corpus, term):
    """术语是**词汇**，不是从正文剪下来的一段代码

    拿一串 50 字的字面量去检索必然词面全中，测出来的不是"术语命中"而是
    "字符串相等"——这种题会让 BM25 白赢，正是要防的那类坏数据。
    """
    chunk = {"text": f"向量检索的第 0 种写法 {term} 结束。"}
    reason = reject_reason(f"{term} 是什么？", KIND_TERM, term, chunk, corpus)
    assert reason[0] == "术语是一段代码而不是词"


def test_term_without_a_term_is_rejected(corpus, vector_chunk):
    # 问题本身是合法的（长度够、无定位词），只有"没给术语"这一条不满足——
    # 否则先撞上的会是结构检查，测不到这条规则
    reason = reject_reason("随便问个什么问题呢？", KIND_TERM, None, vector_chunk, corpus)
    assert reason[0] == "术语型没给术语"


# --------------------------------------------------------------------------
# 结构：长度与定位信息
# --------------------------------------------------------------------------


def test_long_but_natural_question_is_accepted(corpus, vector_chunk):
    """钉住一次实测教训：60 字上限把一大堆真人问法当坏数据丢掉了

    中文里真实的用户提问动辄七八十字，而且**越像真人话的越容易被丢**。
    上限只该拦"模型把一整段话当问题交上来"。
    """
    question = (
        "我在电脑上装好了用来运行程序的工具，现在想确认它是不是真的在后台跑起来了，"
        "而且能随时听我指挥，我该敲哪两条命令来启动它并看看它当前有没有正常工作？"
    )
    assert len(question) > 60
    assert reject_reason(question, KIND_PARAPHRASE, None, vector_chunk, corpus) is None


def test_absurdly_long_question_is_rejected(corpus, vector_chunk):
    question = "问" * (MAX_QUESTION_CHARS + 1)
    assert reject_reason(question, KIND_PARAPHRASE, None, vector_chunk, corpus)[0] == "问题长度不合适"


def test_position_hint_is_rejected(corpus, vector_chunk):
    """喂给模型的资料带了编号，模型有时会把"第几段"抄进问题里

    这种题在评测里是作弊：它把定位信息泄进了问题，检索器不用理解内容也能命中。
    """
    for question in ["上面这段讲的算法是怎么打分的？", "第 3 段说的是什么？"]:
        assert reject_reason(question, KIND_PARAPHRASE, None, vector_chunk, corpus)[0] == "含定位信息"


def test_empty_question_is_rejected(corpus, vector_chunk):
    assert reject_reason("", KIND_PARAPHRASE, None, vector_chunk, corpus)[0] == "问题为空"
    assert reject_reason(None, KIND_PARAPHRASE, None, vector_chunk, corpus)[0] == "问题为空"


# --------------------------------------------------------------------------
# 解析与抽样
# --------------------------------------------------------------------------


@pytest.fixture
def two_chunks():
    return [
        {"source": "a.md", "chunk_id": 0, "text": "甲"},
        {"source": "a.md", "chunk_id": 1, "text": "乙"},
    ]


def test_parse_batch_maps_index_back_to_chunk(two_chunks):
    raw = '[{"index": 2, "paraphrase": "问乙", "term": null, "term_question": null}]'
    items = parse_batch(raw, two_chunks)
    # 一段出两道题：改写型 + 术语型，都挂在同一段上
    assert [item["kind"] for item in items] == [KIND_PARAPHRASE, KIND_TERM]
    assert [item["chunk"]["chunk_id"] for item in items] == [1, 1]
    assert items[0]["question"] == "问乙"
    assert items[1]["question"] is None


def test_parse_batch_strips_a_markdown_fence(two_chunks):
    raw = '```json\n[{"index": 1, "paraphrase": "问甲"}]\n```'
    assert parse_batch(raw, two_chunks)[0]["question"] == "问甲"


def test_parse_batch_drops_out_of_range_index(two_chunks):
    """模型会编出不存在的段号，这种条目整条丢掉，不能让后面的段对错位"""
    raw = '[{"index": 9, "paraphrase": "编的"}, {"index": 1, "paraphrase": "真的"}]'
    items = parse_batch(raw, two_chunks)
    assert [item["question"] for item in items] == ["真的", None]


def test_parse_batch_rejects_a_non_array(two_chunks):
    with pytest.raises(ValueError):
        parse_batch('{"index": 1, "paraphrase": "x"}', two_chunks)


def test_parse_batch_raises_when_there_is_no_json(two_chunks):
    with pytest.raises(ValueError):
        parse_batch("模型今天不想输出 JSON", two_chunks)


def _long_chunks(source, count):
    return [
        {"source": source, "chunk_id": i, "text": "长" * 100} for i in range(count)
    ]


def test_sample_chunks_is_stratified_per_document():
    """按文档分层，而不是全库随机抽

    三份文档分别 140/95/18 段，全库随机抽会让最短的那份几乎抽不到题，
    而它偏偏是唯一一份"介绍某个产品"的语料，题型和另外两份不一样。
    """
    chunks = _long_chunks("a.md", 5) + _long_chunks("b.md", 3)
    picked = sample_chunks(chunks, limit=2)
    assert len(picked) == 4
    assert {chunk["source"] for chunk in picked} == {"a.md", "b.md"}


def test_sample_chunks_skips_chunks_too_short_to_ask_about():
    chunks = _long_chunks("a.md", 3) + [{"source": "a.md", "chunk_id": 9, "text": "标题"}]
    picked = sample_chunks(chunks, limit=0)
    assert [chunk["chunk_id"] for chunk in picked] == [0, 1, 2]


def test_sample_chunks_is_reproducible():
    """同一种子必须抽出同一批：产物要能提交进仓库、能 diff、能复现"""
    chunks = _long_chunks("a.md", 20) + _long_chunks("b.md", 20)
    first = [c["chunk_id"] for c in sample_chunks(chunks, limit=5, seed=1)]
    second = [c["chunk_id"] for c in sample_chunks(chunks, limit=5, seed=1)]
    assert first == second
