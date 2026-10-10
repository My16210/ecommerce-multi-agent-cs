"""
答案质量评测的离线单测。

**不联网、不碰向量库、不需要任何 API key**：判定用的 `get_llm` 一律被换成假模型。

这个文件钉住的核心不变量只有一条：**分数必须能追到标签**。
一旦"分数"变成模型随口给的一个数，这个模块就退化成一个装饰品——
所以下面有一组用例专门盯着"模型给的 score 和它自己标的 label 打架时，听谁的"。
"""
import pytest

from ecom.knowledge.cli import answer_eval
from ecom.knowledge.cli.answer_eval import (
    SELFTEST_CASES,
    agreement,
    faithfulness_score,
    is_abstention,
    parse_faithfulness,
    parse_relevance,
    summarize,
)

# --------------------------------------------------------------------------
# 假模型：只回一段事先写好的文本，并按调用次数计数
# --------------------------------------------------------------------------


class _Response:
    def __init__(self, content):
        self.content = content
        the_usage = {"input_tokens": 111, "output_tokens": 22}
        self.usage_metadata = the_usage


class FakeLLM:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return _Response(self.content)


@pytest.fixture
def fake_llm(monkeypatch):
    """装上假模型。patched 的是 answer_eval 里的名字，不是 llm 模块里的"""
    def install(content):
        llm = FakeLLM(content)
        monkeypatch.setattr(answer_eval, "get_llm", lambda *args, **kwargs: llm)
        return llm
    return install


def _claim(text, label="supported", evidence="依据"):
    return {"claim": text, "label": label, "evidence": evidence}


# --------------------------------------------------------------------------
# 解析忠实性判定
# --------------------------------------------------------------------------


class TestParseFaithfulness:
    def test_normal_json_is_parsed(self):
        raw = ('{"claims": [{"claim": "退货 7 天", "label": "supported", '
               '"evidence": "自签收之日起 7 天内"}], "score": 1.0}')
        parsed = parse_faithfulness(raw)
        assert len(parsed["claims"]) == 1
        assert parsed["claims"][0]["label"] == "supported"
        assert parsed["claims"][0]["evidence"] == "自签收之日起 7 天内"
        assert parsed["parse_error"] is False

    def test_markdown_fence_is_stripped(self):
        """模型爱用 ```json 包裹——这不是失败，是常态"""
        raw = '```json\n{"claims": [], "score": 0.0}\n```'
        assert parse_faithfulness(raw)["parse_error"] is False

    def test_garbage_returns_empty_not_exception(self):
        """解析失败不能抛异常把整轮评测带崩；但也不能假装"没问题"——
        空 claims 会走到 score=None（未定义），不是 1.0"""
        parsed = parse_faithfulness("模型今天不想输出 JSON")
        assert parsed["claims"] == [] and parsed["parse_error"] is True

    def test_array_instead_of_object_is_rejected(self):
        """模型给数组也是坏输出——不能对列表调 .get()"""
        assert parse_faithfulness('[{"claim": "x"}]')["parse_error"] is True

    def test_unknown_label_falls_back_to_unsupported(self):
        """认不出的标签按**最严**处理：不能把来路不明的判定当成 supported 混进分子"""
        raw = '{"claims": [{"claim": "x", "label": "大概对吧", "evidence": ""}]}'
        assert parse_faithfulness(raw)["claims"][0]["label"] == "unsupported"

    def test_claim_without_text_is_dropped(self):
        """空声明会把分母灌水——一条都不该放进来"""
        raw = '{"claims": [{"claim": "  ", "label": "supported"}, {"claim": "有内容"}]}'
        claims = parse_faithfulness(raw)["claims"]
        assert len(claims) == 1 and claims[0]["claim"] == "有内容"


# --------------------------------------------------------------------------
# 忠实性分数：全部从标签算出来
# --------------------------------------------------------------------------


class TestFaithfulnessScore:
    def test_all_supported_is_one(self):
        assert faithfulness_score([_claim("a"), _claim("b")]) == 1.0

    def test_none_supported_is_zero(self):
        assert faithfulness_score([_claim("a", "unsupported")]) == 0.0

    def test_contradicted_counts_as_not_supported(self):
        """和资料打架比"资料没提"更严重，但这一条同样**没有**被支持"""
        assert faithfulness_score([_claim("a", "contradicted")]) == 0.0

    def test_mixed_is_ratio(self):
        claims = [_claim("a"), _claim("b", "unsupported"), _claim("c"), _claim("d")]
        assert faithfulness_score(claims) == pytest.approx(0.75)

    def test_no_claims_is_undefined_not_one(self):
        """**本文件最重要的一条。** 拒答/纯客套没有事实性声明，忠实性是**未定义**。
        算 1.0 会把"我不知道"捧成满分，算 0.0 会把它踩到底——两种都让均值静默失真。
        所以必须是 None，由汇总层单独计数、排除出去。"""
        assert faithfulness_score([]) is None


# --------------------------------------------------------------------------
# 解析相关性判定
# --------------------------------------------------------------------------


class TestParseRelevance:
    def test_normal(self):
        parsed = parse_relevance('{"label": "partial", "reason": "只答了一半"}')
        assert parsed["label"] == "partial" and parsed["reason"] == "只答了一半"

    def test_unknown_label_falls_back_to_refused(self):
        """认不出时**不能默认成 answered**——那等于白送分"""
        assert parse_relevance('{"label": "还行吧"}')["label"] == "refused"

    def test_garbage_is_refused_with_flag(self):
        parsed = parse_relevance("不是 JSON")
        assert parsed["label"] == "refused" and parsed["parse_error"] is True


# --------------------------------------------------------------------------
# 判定函数：分数必须来自标签，不来自模型随口给的数
# --------------------------------------------------------------------------


class TestGradingUsesLabels:
    def test_retries_until_the_output_parses(self, monkeypatch):
        """判定输出解析不出来要重试。**temperature=0 不等于确定性**——
        实测同一个答案重跑就正常了，把这种偶发当成"这题没有声明"是在自欺。"""
        class Flaky:
            def __init__(self):
                self.n = 0

            def invoke(self, messages):
                self.n += 1
                body = ("不是 JSON" if self.n < 3
                        else '{"claims": [{"claim": "a", "label": "supported"}]}')
                resp = _Response(body)
                return resp

        llm = Flaky()
        monkeypatch.setattr(answer_eval, "get_llm", lambda *a, **k: llm)
        result = answer_eval.grade_faithfulness({"contexts": [], "answer": "x"})
        assert llm.n == 3                      # 前两次失败，第三次成功
        assert result["score"] == 1.0
        assert result["parse_error"] is False
        # 重试的 token 必须累加，否则判定的实际成本被低报——试了 3 次只记 1 次
        assert result["usage"] == {"in": 333, "out": 66}

    def test_gives_up_after_the_retry_budget_and_keeps_the_evidence(self, monkeypatch):
        """重试到上限仍失败时，不能悄悄当成"零声明"混过去：既要标 parse_error，
        还要留下原始输出——上次就是因为没留证据，3 条解析失败被当成"拒答"
        静静踢出了分母，报告读起来一切正常。"""
        class Broken:
            def __init__(self):
                self.n = 0

            def invoke(self, messages):
                self.n += 1
                return _Response("模型今天就是不说 JSON")

        llm = Broken()
        monkeypatch.setattr(answer_eval, "get_llm", lambda *a, **k: llm)
        result = answer_eval.grade_faithfulness({"contexts": [], "answer": "x"})
        assert llm.n == answer_eval.PARSE_ATTEMPTS
        assert result["parse_error"] is True
        assert result["score"] is None
        assert "模型今天" in (result["raw"] or "")

    def test_score_ignores_the_models_own_number(self, fake_llm):
        """模型说 score=0.99，但它标的两条声明一条都没被支持——
        这时候必须听**标签**，算 0.0。听模型的数就等于把审计线索扔了。"""
        fake_llm('{"claims": [{"claim": "a", "label": "unsupported", "evidence": ""},'
                 ' {"claim": "b", "label": "contradicted", "evidence": ""}], "score": 0.99}')
        result = answer_eval.grade_faithfulness({"contexts": [], "answer": "x"})
        assert result["score"] == 0.0
        assert result["model_score"] == 0.99  # 记下来，但不采信

    def test_usage_comes_back_for_cost_accounting(self, fake_llm):
        """grade 的成本要靠这个数累加——判定跑在 collect() 外面，
        metric() 是空操作，所以必须一路带回调用方，不能丢在中间"""
        fake_llm('{"label": "answered", "reason": "ok"}')
        result = answer_eval.grade_relevance({"question": "q", "answer": "a"})
        assert result["usage"] == {"in": 111, "out": 22}

    def test_empty_contexts_are_shown_as_such(self, fake_llm):
        """没检索到资料时，提示词里必须写明，不能让模型面对一片空白自己猜"""
        llm = fake_llm('{"claims": [], "score": 0.0}')
        answer_eval.grade_faithfulness({"contexts": [], "answer": "x"})
        assert "没有检索到任何资料" in llm.calls[0][1][1]


# --------------------------------------------------------------------------
# 弃答
# --------------------------------------------------------------------------


class TestAbstention:
    def test_refused_label_is_abstention(self):
        assert is_abstention({"relevance": {"label": "refused"}}) is True

    def test_fallback_is_abstention(self):
        """兜底文案算弃答（正常情况下走不到，见 build 里"生成失败直接抛出去"）"""
        assert is_abstention({"fallback": True, "relevance": None}) is True

    def test_answered_is_not_abstention(self):
        assert is_abstention({"relevance": {"label": "answered"}}) is False


# --------------------------------------------------------------------------
# 汇总：分母全部要说清楚
# --------------------------------------------------------------------------


def _record(question, answerable, faith_score, claims, label,
            fallback=False, parse_error=False):
    return {
        "question": question,
        "answerable": answerable,
        "fallback": fallback,
        "status": "graded",
        "faithfulness": None if faith_score is None and claims is None else {
            "score": faith_score, "claims": claims or [], "parse_error": parse_error,
        },
        "relevance": {"label": label},
    }


class TestSummarize:
    def test_zero_claim_records_leave_the_denominator(self):
        """零声明的答案既不算命中也不算失手——它被**排除**并单独计数。
        混进分子会虚高，混进分母会虚低，两种都是编数。"""
        records = [
            _record("有声明且全支持", True, 1.0, [_claim("a"), _claim("b")], "answered"),
            _record("拒答，没有声明", True, None, [], "refused"),
        ]
        stats = summarize(records)
        assert stats["faithful_scored"] == 1
        assert stats["faithful_no_claims"] == 1
        assert stats["claim_total"] == 2  # 拒答那条一条声明都不贡献

    def test_unanswerable_excluded_from_faithfulness_and_relevance(self):
        """不可答题不该参与"忠实性/相关性"——它的正确答案是拒答，
        掺进去会让两边的分母都失去意义"""
        records = [
            _record("可答", True, 1.0, [_claim("a")], "answered"),
            _record("不可答", False, 0.0, [_claim("x", "unsupported")], "answered"),
        ]
        stats = summarize(records)
        assert stats["answerable"] == 1 and stats["unanswerable"] == 1
        assert stats["claim_total"] == 1
        assert stats["relevance"]["answered"] == 1

    def test_wrong_answers_are_those_that_did_not_abstain(self):
        """「误答」= 库里没有答案却给了实质回答。这是最贵的一类错误，
        报告要能指名道姓列出来，不能只给一个比例。"""
        records = [
            _record("客服电话是多少", False, None, [], "answered"),
            _record("退货地址在哪", False, None, [], "refused"),
        ]
        stats = summarize(records)
        assert stats["abstained"] == 1
        assert stats["wrong_answers"] == ["客服电话是多少"]

    def test_contradictions_are_counted_separately(self):
        records = [_record("q", True, 0.5, [_claim("a"), _claim("b", "contradicted")], "answered")]
        assert summarize(records)["claim_contradicted"] == 1

    def test_ungraded_records_are_ignored(self):
        """还没判定的条目一条都不该进统计"""
        records = [{"question": "q", "answerable": True, "status": "built",
                    "faithfulness": None, "relevance": None}]
        stats = summarize(records)
        assert stats["graded"] == 0 and stats["faithfulness"] is None

    def test_nothing_graded_does_not_crash(self):
        assert summarize([])["faithfulness"] is None

    def test_parse_failure_is_not_counted_as_a_zero_claim_answer(self):
        """**这条钉的是一个真踩过的坑。** 判定解析失败和"答案里没有声明"
        都会让 score 变成 None，但前者是**判定的故障**、后者是答案的性质。
        上一版把两者合并计数，报告写着"已排除 4 条零声明（拒答/纯客套）"，
        其中 3 条其实是解析失败被当成了拒答——数看着正常，坏掉的部分看不见。"""
        records = [
            _record("真的只是拒答", True, None, [], "refused"),
            _record("判定坏了", True, None, [], "answered", parse_error=True),
            _record("判定也坏了", True, None, [], "answered", parse_error=True),
        ]
        stats = summarize(records)
        assert stats["faithful_no_claims"] == 1
        assert stats["faithful_parse_failed"] == 2
        assert stats["faithful_scored"] == 0


# --------------------------------------------------------------------------
# 一致率：分母是"人真的标过的"
# --------------------------------------------------------------------------


def _calibrated(judge_faith, human_faith, judge_rel, human_rel):
    return {
        "faithfulness": {"score": judge_faith},
        "relevance": {"label": judge_rel},
        "human_faithfulness": human_faith,
        "human_relevance": human_rel,
    }


class TestAgreement:
    def test_unlabeled_records_leave_the_denominator(self):
        """没标的不能算"不一致"，也不能算"一致"——两种都是编数。
        所以分母是"两边都有的那些"。"""
        records = [
            _calibrated(1.0, 1.0, "answered", "answered"),
            _calibrated(0.5, None, "partial", None),  # 人没标 → 不进分母
        ]
        agreed, total = agreement(records, "relevance")
        assert (agreed, total) == (1, 1)

    def test_relevance_agreement_is_exact_label_match(self):
        records = [
            _calibrated(1.0, 1.0, "answered", "answered"),
            _calibrated(1.0, 1.0, "answered", "partial"),  # 不一致
        ]
        assert agreement(records, "relevance") == (1, 2)

    def test_faithfulness_agreement_uses_a_coarse_band(self):
        """0.83 vs 0.86 本来就没人分得清，硬比小数会把一致率压得毫无意义。
        所以只判"高 / 低"这一档。"""
        records = [
            _calibrated(0.83, 1.0, "answered", "answered"),  # 同档
            _calibrated(0.2, 1.0, "answered", "answered"),   # 跨档 → 不一致
        ]
        assert agreement(records, "faithfulness") == (1, 2)

    def test_missing_judge_score_leaves_the_denominator(self):
        """判定说"未定义"（零声明）时没法比，直接排除，不能当成不一致"""
        records = [_calibrated(None, 1.0, "refused", "refused")]
        assert agreement(records, "faithfulness") == (0, 0)
        assert agreement(records, "relevance") == (1, 1)


# --------------------------------------------------------------------------
# 不可答题自检
# --------------------------------------------------------------------------


class TestUnanswerableSelfCheck:
    def test_top_score_reads_the_best_cosine(self):
        record = {"contexts": [{"vector_score": 0.3}, {"vector_score": 0.8}]}
        assert answer_eval._top_score(record) == pytest.approx(0.8)

    def test_top_score_ignores_the_rrf_fused_score(self):
        """**这条钉的是一个真踩过的坑。** `score` 是 RRF 融合分，量纲 1/(K+rank)，
        实测恒在 0.03 上下——拿它比 0.65 的相似度阈值永远为假，自检就成了摆设，
        还让人以为"自检过了"。所以这里塞一个很高的 RRF 分进去，
        期望 `_top_score` **不**理它。"""
        record = {"contexts": [{"score": 0.99, "vector_score": 0.2}]}
        assert answer_eval._top_score(record) == pytest.approx(0.2)

    def test_no_contexts_is_zero_not_a_crash(self):
        assert answer_eval._top_score({"contexts": []}) == 0.0
        assert answer_eval._top_score({}) == 0.0

    def test_review_line_is_a_plausible_similarity(self):
        """复核提示线是**提示不是判据**，但它至少得落在余弦相似度的量纲里。
        定在 0.65 是拿实测的两条标错题（0.662 / 0.706）和六条真查不到的
        （0.530～0.642）标出来的——见常量上面那段。"""
        assert 0.5 < answer_eval.UNANSWERABLE_COSINE_REVIEW < 1.0


# --------------------------------------------------------------------------
# 金丝雀用例本身
# --------------------------------------------------------------------------


class TestSelftestCases:
    def test_covers_the_failure_modes_that_matter(self):
        """金丝雀要覆盖四类：忠实的、编造的、与资料矛盾的、跑题的，
        外加拒答（它检验的是"零声明"这条边界）。少哪一类，判定者在那类上
        失手都不会被发现。"""
        names = " ".join(case["name"] for case in SELFTEST_CASES)
        for keyword in ("忠实", "编造", "矛盾", "跑题", "拒答"):
            assert keyword in names, f"金丝雀缺少「{keyword}」这一类"

    def test_fabrication_case_expects_being_caught(self):
        """编造那条必须期望被判成"没被支持"——期望写成 True 就等于
        根本不检查判定者有没有抓住编造"""
        fabrication = next(c for c in SELFTEST_CASES if "编造" in c["name"])
        assert fabrication["expect_faithful"] is False

    def test_refusal_case_expects_zero_claims(self):
        """拒答句不是事实性声明。判定者在这里硬凑出一条声明，
        就会把未定义（None）算成 0%，把忠实性均值往下拽。"""
        refusal = next(c for c in SELFTEST_CASES if "拒答" in c["name"])
        assert refusal.get("expect_zero_claims") is True
        assert "expect_faithful" not in refusal
