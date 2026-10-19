"""指代消解：什么时候该"接着上一轮说"，什么时候必须闭嘴。

这组测试守的是一条**不对称**：消解不了只是维持现状（多问一句），消解错了会
答非所问、甚至在写操作上替用户对错误订单办事。所以每条"该闭嘴"的情形都单独
钉一条——它们比"能消解"那边更容易被后来人放松。
"""
from ecom.orchestrator.settings import HISTORY_TURNS
from ecom.orchestrator.utils.entities import carried_order_id

# 一份最小的两轮历史：用户提了订单号，客服答了。
TWO_TURNS = [
    {"role": "user", "content": "订单 52644 到哪了"},
    {"role": "assistant", "content": "订单 52644 的物流：目的地 San Salvador，最近更新 2018-01-15。"},
]


def test_a_followup_picks_up_the_order_id_from_the_previous_turn():
    assert carried_order_id("那能退吗", TWO_TURNS) == "52644"


def test_the_id_is_looked_up_from_the_most_recent_mentioning_turn():
    """从后往前找**最近一条提过订单号的**用户发言，跳过中间那些没提号的。

    「订单 52644 到哪了」→「谢谢」→「它还能退吗」这一段里，最近一条**有号**的
    其实是两条之前那条；只看紧挨着的那一句会漏掉。
    """
    history = [
        {"role": "user", "content": "订单 52644 到哪了"},
        {"role": "assistant", "content": "已发货。"},
        {"role": "user", "content": "好的谢谢"},
        {"role": "assistant", "content": "不客气。"},
    ]
    assert carried_order_id("它还能退吗", history) == "52644"


# ---------------------------------------------------------------------------
# 该闭嘴的四种情形
# ---------------------------------------------------------------------------
def test_no_anaphora_word_means_no_carry():
    """没有指代词 = 在说一件新事，哪怕上一轮刚提过订单。

    「退货政策是什么」不是一个追问，它是一个新问题；把它和上一轮的订单绑在
    一起，会让一个纯政策问题变得像在针对某笔订单。
    """
    assert carried_order_id("退货政策是什么", TWO_TURNS) == ""


def test_an_order_id_in_the_current_turn_wins_over_the_carried_one():
    """"那 51592 呢"——当轮自己说了号，就不需要猜。返回空是**对的**：
    调用方（`_apply_carry`）看到当轮有号会直接用当轮那个，这里再返回一个
    就等于给了两个答案。"""
    assert carried_order_id("那 51592 呢", TWO_TURNS) == ""


def test_two_different_order_numbers_in_one_turn_are_not_guessed():
    """"对比 52644 和 33333"后面接「哪个更快」——两个候选，消解不了。

    歧义时**不猜**：猜错的那一半用户看不出来（答案确实提了一笔订单，只是
    不是他想问的那笔），所以宁可回空、让订单 worker 去要订单号。
    """
    history = [{"role": "user", "content": "对比一下 52644 和 33333 哪个更快"}]
    assert carried_order_id("那哪个更快", history) == ""


def test_numbers_in_the_assistant_answer_are_never_treated_as_order_ids():
    """只扫 user 消息。assistant 的回答里 5 位以上的数字太多了——
    销售额、订单条数、退货率，`ORDER_ID_RE` 会把它们全都当成订单号。

    （这条用一个**只有客服说过数字**的历史来钉住：库里真有 62992 这种数，
      它长得和订单号一模一样。）
    """
    history = [
        {"role": "user", "content": "我的订单到哪了"},
        {"role": "assistant", "content": "客户 C9995 名下共 6 笔订单，总金额 62992.10 元。"},
    ]
    assert carried_order_id("那能退吗", history) == ""


def test_an_order_id_from_far_back_is_not_picked_up():
    """窗口 = 最近 `HISTORY_TURNS` 条用户发言。再往前的是**另一件事**了，
    接着说不等于还在这件事上。"""
    filler = [{"role": "user", "content": f"顺便聊聊别的吧 {i}"} for i in range(HISTORY_TURNS)]
    history = [{"role": "user", "content": "订单 11111 到哪了"}] + filler
    assert carried_order_id("那能退吗", history) == ""


def test_an_empty_history_is_not_an_error():
    # 单轮提问是**常态**（CLI 的 main.py 就常常不带历史），不能因此抛异常。
    assert carried_order_id("那能退吗", None) == ""
    assert carried_order_id("那能退吗", []) == ""
