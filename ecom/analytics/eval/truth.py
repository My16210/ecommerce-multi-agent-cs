"""评测题的真值——**用标准库独立算出来**，不是抄在代码里的一串常数。

【为什么必须换一套工具算】

被测链路（analyst 写代码 → 沙箱执行）用的就是 pandas。如果真值也拿 pandas
算出来，那么"pandas 的理解出错"和"模型对数据的理解出错"会**同时**错在同一个
地方：真值跟着答案一起偏，评测照样判"答对"。

举例：假设 2024 年后的 pandas 改了 `groupby` 对空值的默认处理，而模型恰好
依赖这个行为。用 pandas 算真值，两边一起变，准确率依然是 100%——这个 bug
永远不会被发现。这不是假想的洁癖，是"用自己的输出验证自己"的经典失败模式。

所以这里只用标准库 `csv` + 手写循环。两条实现路径**互相独立**，它们给出同一个
答案，才说明答案是对的。代价是慢（30 行无所谓，DataCo 18 万行也就几秒）和代码
啰嗦，换来的是一份真正独立的口径。

【为什么真值不写死在 EXPECTED 里】

写死的常数没有来源。改一行数据、修一个 ETL bug，那串数字就悄悄过期了，
而评测会继续拿旧数字去判——判错的还是对的。真值必须能**跟着数据重算**。

本模块有一条自我约束：**不许 import pandas**。tests/analytics/test_eval_truth.py
里有一条测试守着它——不然"独立实现"这个前提会在某次顺手重构里悄悄消失，
而消失之后评测还是绿的，没人会注意到。
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


def load_rows(csv_path: Path) -> list[dict]:
    """用标准库读表。utf-8-sig 吃掉 BOM，否则首列名会带一个看不见的 \\ufeff。"""
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _argmax(counter: Counter, keys: list) -> object:
    """取最大值对应的键。

    并列时按 keys 里**先出现**的那个——顺序不能靠 dict 的插入序"看起来对"，
    所以显式排序，让并列的结果在两次运行之间稳定。
    """
    best = None
    for k in keys:
        if best is None or counter[k] > counter[best]:
            best = k
    return best


def sales_eval_set(csv_path: Path) -> dict:
    """30 行手写销售数据的评测题与真值。

    口径以 datasets/sales/NOTES.md 为准：销售额统计**全部订单**，不剔除退货。
    """
    rows = load_rows(csv_path)

    amount = Counter()   # 地区 -> 金额合计
    quantity = Counter() # 地区 -> 销量合计
    orders = Counter()   # 地区 -> 订单数
    refunded = Counter() # 地区 -> 退货订单数
    by_product = Counter()

    for r in rows:
        region = r["地区"].strip()
        amount[region] += float(r["金额"])
        quantity[region] += int(r["销量"])
        orders[region] += 1
        if r["是否退货"].strip() == "1":
            refunded[region] += 1
        by_product[r["商品名"].strip()] += int(r["销量"])

    # 地区首次出现的顺序，用来给并列情况一个确定的答案
    region_order = list(dict.fromkeys(r["地区"].strip() for r in rows))
    product_order = list(dict.fromkeys(r["商品名"].strip() for r in rows))

    top_region_by_amount = _argmax(amount, region_order)
    top_product = _argmax(by_product, product_order)
    top_refund_region = _argmax(
        Counter({k: refunded[k] / orders[k] for k in region_order}), region_order
    )

    # 「iPhone15 在哪个地区卖得最多」——只在含该商品的记录里按地区汇总，
    # 别的地区没有这个商品，不能当成"销量 0"参与比较（那样会引入一堆并列 0）。
    # 商品名从数据里取，不写死 "iPhone15"。
    biggest_product = top_product
    region_of_product = Counter()
    for r in rows:
        if r["商品名"].strip() == biggest_product:
            region_of_product[r["地区"].strip()] += int(r["销量"])
    top_region_of_product = _argmax(region_of_product, region_order)

    return {
        "哪个地区卖得最好？": {"kind": "text", "value": top_region_by_amount},
        "销量最高的商品是什么？": {"kind": "text", "value": top_product},
        "哪个地区的退货率最高？": {"kind": "text", "value": top_refund_region},
        "总销售额是多少？": {
            "kind": "number",
            "value": sum(amount.values()),
            "tol": 0.001,
        },
        f"{biggest_product}在哪个地区卖得最多？": {
            "kind": "text",
            "value": top_region_of_product,
        },
    }


# 数据集名 -> 真值构造器。每个构造器自己读那个数据集的 CSV、自己算。
# 加一个数据集 = 加一个函数 + 加一行注册；**没有**一个"通用真值引擎"，
# 因为五个问题各算各的，抽象成 DSL 只会让人看不懂这道题到底在比什么。
EVAL_SETS = {
    "sales": sales_eval_set,
    # "dataco": dataco_eval_set,  ← 等真实数据落地、看过实际取值域之后再写。
    #    提前照着记忆写一份"大概是这样"的真值，就等于把猜测写进了评测基准。
}


def get_eval_set(dataset_name: str, csv_path: Path) -> dict:
    """取某个数据集的评测题与真值。

    未注册的数据集**直接报错**，不返回空 dict：空 dict 会让评测跑出 0/0，
    报"准确率 0%"或者更糟，除以零崩在一个看不出原因的地方。
    """
    if dataset_name not in EVAL_SETS:
        known = "、".join(sorted(EVAL_SETS)) or "（无）"
        raise KeyError(
            f"数据集 {dataset_name!r} 还没有评测题真值。已注册的：{known}。"
            f"请在 ecom/analytics/eval/truth.py 里为它写一个构造器——"
            f"真值必须从数据算出来，不要手写常数。"
        )
    return EVAL_SETS[dataset_name](csv_path)
