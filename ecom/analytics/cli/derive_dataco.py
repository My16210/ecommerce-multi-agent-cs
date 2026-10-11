"""
DataCo 主表 → 分析用的窄表（派生产物，提交进仓库）

    uv run python -m ecom.analytics.cli.derive_dataco

【为什么要有这一步：原表不能直接当 data.csv】

三个理由，每一个单独都够：

1. **编码不对。** DataCo 主表不是 UTF-8，是 cp1252（西欧单字节）。直接读会在
   第 1709 字节抛 `UnicodeDecodeError`——不是"乱码能看"，是**连表头都读不出来**。
   而分析侧的 `data_loader` / `code_executor` 统一用 `encoding="utf-8-sig"` 读表，
   这是对的选择（它要能吃 BOM）。所以**在派生阶段把编码统一成 UTF-8**，
   而不是在下游为每一份数据各配一个编码——那样每加一个数据集就要改两处代码。

2. **体积。** 原表 95.9 MB / 180519 行。仓库背不动，也不需要背：真正用得上的
   是其中一部分列，而且分析结论在抽样上完全稳定（见下面的抽样说明）。

3. **PII 和垃圾列。** 原表里有 Customer Email / Customer Password / Customer
   Street / Customer Zipcode / 经纬度——这些和"地区销售分析"没有关系，没必要
   再从公开数据集里复刻一份进我们自己的仓库。另外 Product Description 这一列
   **180519 行全空**，Product Image 是一串指向外部站点的 URL，都不该跟着走。

【抽样：按订单号哈希，不是按行随机】

`order_id` 的 md5 前 4 字节对 9 取模，为 0 就整笔订单留下。

三个性质，缺一不可：

- **确定性**：同一份源数据、同一段代码，永远得到同一份结果。所以这个脚本可以
  跑第二遍、结果和仓库里那份逐字节相同——不用把"上次抽出来的是哪些"记在别处。
- **订单完整**：抽的是**订单**不是**行**。DataCo 一行 = 一个订单里的一个商品，
  一笔订单平均 2.7 行。按行抽会把一笔订单拆成"留下三件、丢掉两件"，那么
  「成交额」这类按订单汇总的口径会静默地算少——而且错得很隐蔽，因为表里每一行
  看起来都是完整的真实数据。
- **可复现的规则**：规则写在 NOTES.md 里（"md5 前 4 字节 % 9 == 0"），
  谁都可以拿原始数据自己验算，不需要信任这个脚本。

【这一步为什么不用 pandas】

不是洁癖：**这一步必须在宿主机上用标准库跑**（本机 PyPI 不通，装不了 pandas），
而且它做的事（选列、过滤、写盘）用 stdlib 反而更直白。pandas 留给真正需要它的
地方——被测的那条链路。
"""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

from ecom.analytics.settings import settings

# 源文件（不进仓库）。文件名与 Mendeley 上的一致，sha256 记在 NOTES.md 里。
SOURCE_CSV = Path(settings.SOURCE_DIR) / "dataco" / "DataCoSupplyChainDataset.csv"

# 源文件的编码。**不是 UTF-8**——写死在这里，是因为"猜编码"（chardet 那一类）
# 在一个 96MB、99% 是 ASCII 的文件上只会给出一个看起来合理但可能错一半的答案。
# 官方发布的就是 cp1252，那就写 cp1252。
SOURCE_ENCODING = "cp1252"

# 保留哪 18 列、按什么顺序。列名**照抄源表**，不翻译、不改成 snake_case：
# 这份数据是公开数据集，改列名等于我们自己造了一套字段名，而下游要引用的是
# "DataCo 的哪一列"这件事。含义写在 NOTES.md 里，那是给人读的。
KEEP_COLUMNS = [
    "order date (DateOrders)",      # 下单时间
    "Order Id",
    "Order Status",
    "Delivery Status",
    "Late_delivery_risk",
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Shipping Mode",
    "Market",
    "Order Region",
    "Order Country",
    "Category Name",
    "Product Name",
    "Order Item Quantity",
    "Sales",
    "Order Item Total",
    "Order Item Discount",
    "Customer Segment",
]

# 抽样：md5("dataco:<order_id>") 前 4 字节对 SAMPLE_MOD 取模，余数 < SAMPLE_KEEP 就留。
# 表达成"9 笔留 1 笔"比"11.1%"好——前者能被人在心里验算。
SAMPLE_MOD = 9
SAMPLE_KEEP = 1

OUTPUT_CSV = Path(settings.DATASET_DIR) / "data.csv"


def keep_order(order_id: str) -> bool:
    """这笔订单要不要抽中。判据只跟 order_id 有关，所以同订单的所有行同进同出。"""
    digest = hashlib.md5(f"dataco:{order_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % SAMPLE_MOD < SAMPLE_KEEP


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    if not SOURCE_CSV.exists():
        print(f"找不到源数据：{SOURCE_CSV}", file=sys.stderr)
        print("它是公开数据集，需要单独下载（不进仓库）。下载地址与 sha256 见 "
              f"{Path(settings.DATASET_DIR) / 'NOTES.md'}。", file=sys.stderr)
        return 1

    rows_in = rows_out = 0
    orders_in: set = set()
    orders_out: set = set()
    # 同一订单的多行之间，被抽中的判据是同一个 order_id，所以只会全进或全不进；
    # 这个集合用来把"订单数"和"行数"分开报——只报行数会让人以为抽样比例是按行的。
    seen_unkept: set = set()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SOURCE_CSV, encoding=SOURCE_ENCODING, newline="") as src, \
            open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=KEEP_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        for row in reader:
            rows_in += 1
            order_id = row["Order Id"].strip()
            orders_in.add(order_id)
            if not keep_order(order_id):
                seen_unkept.add(order_id)
                continue
            writer.writerow(row)
            rows_out += 1
            orders_out.add(order_id)

    print(f"源：{SOURCE_CSV}")
    print(f"    sha256 {sha256_of(SOURCE_CSV)}")
    print(f"    {rows_in} 行 / {len(orders_in)} 笔订单（{SOURCE_ENCODING}）")
    print(f"派生：{OUTPUT_CSV}")
    print(f"    {rows_out} 行 / {len(orders_out)} 笔订单，{OUTPUT_CSV.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"    抽样规则：md5('dataco:<Order Id>') 前 4 字节 % {SAMPLE_MOD} < {SAMPLE_KEEP}")
    print(f"    核对：留下 {len(orders_out)} + 丢掉 {len(seen_unkept)} = {len(orders_out) + len(seen_unkept)}")

    if rows_out == 0:
        print("抽出来是空的，检查抽样规则。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
