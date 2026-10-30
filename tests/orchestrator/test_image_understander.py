"""图片理解：结构化契约 / 失败降级 / 不联网。

**一条都不调真视觉模型。** 所有模型调用都用 `FakeLLM` 顶掉——理由和
`conftest.py` 开头写的一样：真调模型会让测试随模型版本、网络、当天负载浮动，
那样的测试没法当回归用。

这组测试守三件事：
  1. 模型吐出来的 JSON 能稳定归一成 `ImageIntent`（包括围栏、字段污染、错号）。
  2. 任何失败都是 `ok=False` 的**返回值**，不是异常——界面靠它降级成"手动输入"。
  3. 图片块和用户正文分得开：块里的订单号不能混进正文被当成"用户当轮说的号"。
"""
import pytest

from ecom.orchestrator.tools.image_understander import understand_image
from ecom.orchestrator.utils.image_intent import (
    ImageIntent,
    augment_question_with_image,
    format_image_context,
    image_order_id,
    image_request,
    sanitize_image_text,
    strip_image_block,
)
from tests.orchestrator.conftest import ExplodingLLM, FakeLLM

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32


# ---------------------------------------------------------------------------
# 正常路径：JSON → ImageIntent
# ---------------------------------------------------------------------------
def test_a_valid_json_payload_is_normalized():
    llm = FakeLLM(
        '{"order_id":"52644","amount":"￥629.92","product":"人体工学椅",'
        '"request":"退货","confidence":"high","notes":""}'
    )
    intent = understand_image(PNG, "image/png", llm=llm)

    assert intent.ok is True
    assert intent.order_id == "52644"
    assert intent.amount == "￥629.92"
    assert intent.product == "人体工学椅"
    assert intent.request == "退货"
    assert intent.confidence == "high"
    assert intent.needs_manual_order_id is False
    assert intent.tokens == {"in": 10, "out": 5}


def test_a_markdown_fenced_payload_is_still_parsed():
    # 模型很爱在 JSON 外面裹一层 ```json；json_utils 已经会剥，这里钉住它
    # 在图片这条路上也生效。
    llm = FakeLLM('```json\n{"order_id":"52644","request":"查物流"}\n```')
    intent = understand_image(PNG, llm=llm)
    assert intent.ok is True
    assert intent.order_id == "52644"
    assert intent.request == "查物流"


def test_the_model_cannot_smuggle_a_customer_id_into_the_order_field():
    """`C9995` 不是订单号；`1234` 也不够 5 位。

    视觉模型完全可能把客户号、运单号、金额写进 order_id。校验不过就留空——
    空号会让主链路问用户要，错号会让它去查一笔不存在的订单。
    """
    for raw in ("C9995", "1234", "2025 年"):
        intent = understand_image(PNG, llm=FakeLLM(f'{{"order_id":"{raw}"}}'))
        assert intent.ok is True
        assert intent.order_id == "", f"{raw} 不该被当成订单号"


def test_an_order_id_embedded_in_a_sentence_is_extracted():
    # OCR 经常把"订单号：52644"整段塞进字段里；取其中的数字即可。
    intent = understand_image(PNG, llm=FakeLLM('{"order_id":"订单号：52644"}'))
    assert intent.order_id == "52644"


def test_an_unrecognized_confidence_is_downgraded_to_low():
    # 不确定就说不确定。把 "maybe" 原样显示出去，等于让界面替模型背书。
    intent = understand_image(PNG, llm=FakeLLM('{"request":"退货","confidence":"maybe"}'))
    assert intent.confidence == "low"


def test_the_image_is_sent_as_a_data_url():
    llm = FakeLLM('{"request":"查物流"}')
    understand_image(PNG, "image/png", llm=llm)

    messages = llm.calls[0]
    human_content = messages[1][1]
    image_block = human_content[1]
    assert image_block["type"] == "image_url"
    assert image_block["image_url"]["url"].startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# 失败路径：全部是 ok=False，不是异常
# ---------------------------------------------------------------------------
def test_an_empty_image_is_rejected_before_calling_the_model():
    llm = FakeLLM('{"request":"退货"}')
    intent = understand_image(b"", llm=llm)
    assert intent.ok is False
    assert "空" in intent.error
    assert llm.calls == []          # 连模型都不该碰


def test_an_oversized_image_is_rejected(monkeypatch):
    # 把上限改到极小，避免测试真去分配 5MB。
    monkeypatch.setattr(
        "ecom.orchestrator.tools.image_understander.VISION_MAX_IMAGE_MB", 0.000001
    )
    intent = understand_image(b"xx", llm=FakeLLM("{}"))
    assert intent.ok is False
    assert "上限" in intent.error


def test_a_non_image_mime_type_is_rejected():
    intent = understand_image(PNG, "application/pdf", llm=FakeLLM("{}"))
    assert intent.ok is False
    assert "不支持" in intent.error


def test_non_json_output_is_a_graceful_failure():
    intent = understand_image(PNG, llm=FakeLLM("我看不清这张图"))
    assert intent.ok is False
    assert "JSON" in intent.error
    assert intent.raw == "我看不清这张图"


def test_a_dead_vision_model_does_not_raise():
    """可选功能挂了，主链路还得能走——所以这里必须是返回值。"""
    intent = understand_image(PNG, llm=ExplodingLLM())
    assert intent.ok is False
    assert "模型不可用" in intent.error


def test_a_missing_key_is_a_readable_failure(monkeypatch):
    monkeypatch.setattr(
        "ecom.orchestrator.tools.image_understander.VISION_API_KEY", ""
    )
    # 清掉缓存，避免拿到别的测试建好的 client。
    monkeypatch.setattr("ecom.orchestrator.tools.image_understander._clients", {})
    intent = understand_image(PNG)
    assert intent.ok is False
    assert "VISION_API_KEY" in intent.error


def test_an_ok_result_without_an_order_id_is_not_a_failure():
    """识别成了，只是图里没有订单号——这是正常情况，不是"识别失败"。"""
    intent = understand_image(PNG, llm=FakeLLM('{"request":"退货","order_id":""}'))
    assert intent.ok is True
    assert intent.order_id == ""
    assert intent.needs_manual_order_id is True


# ---------------------------------------------------------------------------
# 文本块：拼接 / 解析 / 隔离
# ---------------------------------------------------------------------------
def test_the_block_round_trips():
    intent = ImageIntent(
        ok=True,
        order_id="52644",
        amount="629.92",
        product="人体工学椅",
        request="退货",
        confidence="high",
    )
    block = format_image_context(intent)

    assert block.startswith("【图片识别结果】")
    assert image_order_id(block) == "52644"
    assert image_request(block) == "退货"
    assert strip_image_block(f"我要退货\n\n{block}") == "我要退货"


def test_an_order_id_outside_the_block_is_not_an_image_order_id():
    """只有**块里**的号才算图片来的号。

    正文里的号是用户自己打的，`workers/order.py` 会把它标成 `question`，
    写操作允许——两者不能混。
    """
    assert image_order_id("订单 52644 到哪了") == ""


def test_a_malformed_block_is_left_alone_instead_of_guessed():
    """没有结束标记 = 不是一个完整的块。宁可当普通文本，也不猜。"""
    broken = "【图片识别结果】\n订单号：52644"
    assert strip_image_block(broken) == broken
    assert image_order_id(broken) == ""


def test_a_block_without_an_order_id_still_formats_the_rest():
    intent = ImageIntent(ok=True, product="杯子", request="查物流")
    block = format_image_context(intent)
    assert "订单号" not in block
    assert "杯子" in block
    assert image_order_id(block) == ""


def test_an_empty_intent_produces_no_block():
    assert format_image_context(ImageIntent(ok=True)) == ""
    assert format_image_context(ImageIntent(ok=False, request="退货")) == ""


def test_a_model_that_repeats_an_order_id_in_the_request_cannot_leak_it_into_the_body():
    """**这是图片功能最重要的一条安全测试。**

    视觉模型可能在 `request` 里复述「我要退货，订单号 52644」。如果原样拼进
    用户正文，规则层会把它标成 `source="question"`，从而允许写操作——等于
    绕过了"图片里的号只能读"那道闸。
    """
    intent = ImageIntent(ok=True, order_id="52644", request="我要退货，订单号 52644")
    question = augment_question_with_image("", intent)

    body, _, block = question.partition("【图片识别结果】")
    assert "52644" not in body, "订单号从 request 漏进了用户正文"
    assert "[订单号]" in body
    assert "52644" in block, "真正的订单号应当留在结构化块里"


def test_the_user_text_wins_over_the_image_request():
    intent = ImageIntent(ok=True, order_id="52644", request="查物流")
    question = augment_question_with_image("帮我看看这个", intent)
    assert question.startswith("帮我看看这个")
    assert "图片中的诉求" not in question.split("【图片识别结果】")[0]


def test_a_user_typed_order_id_is_not_sanitized():
    """清洗只针对**图片诉求**，不能连用户自己打的号一起洗掉。

    用户打的号是他当轮明确给出的，写操作可以用；洗掉它会把一笔正常退货
    变成"请提供订单号"。
    """
    intent = ImageIntent(ok=True, order_id="52644", request="退货")
    question = augment_question_with_image("订单 51592 我要退货", intent)
    body = question.split("【图片识别结果】")[0]
    assert "51592" in body


def test_an_image_only_turn_uses_the_request_as_the_question():
    intent = ImageIntent(ok=True, order_id="52644", request="查物流")
    question = augment_question_with_image("", intent)
    assert question.startswith("查物流")


def test_an_ok_intent_with_nothing_useful_produces_no_question():
    assert augment_question_with_image("", ImageIntent(ok=True)) == ""


def test_a_failed_intent_never_produces_a_question():
    assert augment_question_with_image("我还想问点别的", ImageIntent(ok=False, error="x")) == ""


@pytest.mark.parametrize(
    "text",
    [
        "订单 52644 我要退货",
        "客户 C9995 的订单 52644",
        "金额 62992 元",
    ],
)
def test_sanitize_removes_order_ids_but_keeps_the_words(text):
    cleaned = sanitize_image_text(text)
    assert "52644" not in cleaned
    assert "62992" not in cleaned
    assert "退货" in cleaned or "客户" in cleaned or "金额" in cleaned
