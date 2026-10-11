"""
从 DataCo 主表派生出订单库的种子数据

    uv run python -m ecom.orchestrator.db.derive_seed

产物（提交进仓库）：
    data/orchestrator/seed/orders.csv      订单表
    data/orchestrator/seed/logistics.csv   物流表

【为什么订单库的数据要从一份公开数据集派生，而不是继续手写】

原来那 20 条是手写的，好处是测试里能直接写 `assert amount == 299.0`。但它有个
说不出口的毛病：**它只证明了代码能处理我编出来的那种订单**。真实订单长什么样——
订单号几位、状态有几种、物流信息缺哪几项——手写数据一个都答不上来。换成派生之后，
`seed.py` 里那 20 行常量换成了一份可复现的 CSV，而这份 CSV 带着真实数据的
全部脾气（见下面【派生规则里必须解释的几件事】）。

【这一步和分析侧的派生脚本是两个脚本，不是重复】

`ecom/analytics/cli/derive_dataco.py` 派生的是**分析用的抽样表**（按订单号抽 1/9，
列挑的是给 pandas 用的那十几列）。这里是**订单库的全量种子**（一笔不抽，还要
客户姓名、城市、配送状态这些分析表里没有的列）。两个下游要的东西不一样，
硬合成一个脚本只会让两边都读不懂。同一个源，两份派生，各自一份说明——这是
"一份数据服务两个消费者"的正常样子，不是重复代码。

【派生规则里必须解释的几件事】

1. **订单号保持原样，不补零**（1~5 位）。补零看着更整齐，但那等于改掉了数据集的
   标识符，而这个项目的订单号是**给人念**的（「订单 77202 到哪了」）。代价是
   order worker 的正则只能认 5 位以上，1~4 位的订单号在自然语言里认不出来——
   这是一条已知边界，写在 README 里，不在这里偷偷补零绕过。

2. **客户号加 `C` 前缀**（`Order Customer Id` = 20755 → `C20755`）。DataCo 的订单号
   和客户号都是纯数字，不加前缀的话「订单 37570」和「客户 37570」是同一个字符串，
   规则层没法区分谁是谁。加一个字符就分开了，而且现有 `_guess_customer` 的
   `C\\d{3,}` 一行都不用改。前缀是**派生的展示编码**，不是原始字段——原始值
   就是纯数字，这一点写在这里免得以后有人以为 DataCo 里就带 C。

3. **金额 = 该订单所有行 `Sales` 求和**（未打折口径）。DataCo 一行是一个商品，
   一笔订单平均 2.8 行，所以订单金额必须按 Order Id 汇总，不能取第一行。
   选 `Sales` 而不是 `Order Item Total`（折后）是为了和分析侧的口径一致——
   那边 NOTES.md 里写死了默认用 `Sales`。两个子系统对同一个词用同一个口径，
   不然后面解释差异要花掉一整页。

4. **`paid_at` 一律为空**。DataCo 只有下单时间，没有支付时间。**不拿下单时间
   顶上**：那会让"已支付"这件事变成我们编的。代价是 `_fmt_status` 原来那句
   「（未支付）」必须改——空值现在有第二种含义（"数据里没有这个字段"），
   把"没记录"说成"未支付"就是编。

5. **没有运单的订单**：`Delivery Status == "Shipping canceled"` 的订单**不写物流行**。
   这不是数据缺失，是业务事实——被取消的配送本来就没有运单，和原来那句
   「未发货和已取消的订单本来就没有运单」是同一条。其余订单都有物流行。

6. **承运商、运单号一律为空，发货地一律为空**。公开数据集里没有这三项
   （承运商和运单号是快递公司的私有数据；`Order City` 是**目的地**，不是发货地）。
   留空 + 文档说明，**不合成假值充数**。

7. **状态用 DataCo 的英文原值**（9 种），不映射回中文。9 种映射到 5 种要靠猜
   （ON_HOLD 算"已下单"还是"已取消"？没有依据），而旧的那 5 个中文状态本来就是
   手写数据的产物。宁可回答里出现 `COMPLETE`，也不要一个编出来的映射。
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from ecom.analytics.settings import settings as analytics_settings
from ecom.orchestrator.settings import ORDER_SEED_DIR

SOURCE_DIR = analytics_settings.SOURCE_DIR

# 源文件。**跨子系统读分析侧的 source 目录**是有意的：这份下载下来的原始数据
# 只存一份，谁要用谁去读；再复制一份到 data/orchestrator/ 下就是两份真相。
SOURCE_CSV = Path(SOURCE_DIR) / "dataco" / "DataCoSupplyChainDataset.csv"
SOURCE_ENCODING = "cp1252"  # 源文件不是 UTF-8

ORDERS_CSV = Path(ORDER_SEED_DIR) / "orders.csv"
LOGISTICS_CSV = Path(ORDER_SEED_DIR) / "logistics.csv"

# 写进 CSV 的列顺序 = seed.py 里 INSERT 的列顺序，两边对着看就能核对
ORDER_COLUMNS = [
    "order_id", "customer_id", "customer_name", "status",
    "amount", "region", "created_at", "paid_at",
]
LOGISTICS_COLUMNS = [
    "order_id", "carrier", "tracking_no", "from_city", "to_city",
    "status", "updated_at",
]

# 配送被取消 = 这单没有运单。用 `Delivery Status` 而不是 `Order Status` 判断：
# "有没有发出去"是配送维度的事，不是订单维度的事。
NO_SHIPMENT_STATUS = "Shipping canceled"

# 源里是 `1/13/2018 12:06`（M/D/YYYY，**月在前**）。写成常量是为了让它显眼——
# 这个格式一旦被当成 D/M/YYYY，2018 年 1 月 13 日会变成"第 13 月"，
# 而 Python 的 strptime 会直接抛异常，所以只会炸不会错，这是好事。
SOURCE_DATETIME_FORMAT = "%m/%d/%Y %H:%M"
TARGET_DATETIME_FORMAT = "%Y-%m-%d %H:%M"


def parse_time(raw: str) -> str:
    """`1/13/2018 12:06` → `2018-01-13 12:06`。解析不了就原样返回。

    不抛异常是有意的：这条记录的时间格烂了，不该让整个派生脚本停下——
    但也不该悄悄变成空，那样看起来像"这单没有时间"。原样返回至少能查。
    """
    raw = (raw or "").strip()
    try:
        return datetime.strptime(raw, SOURCE_DATETIME_FORMAT).strftime(TARGET_DATETIME_FORMAT)
    except ValueError:
        return raw


def main() -> int:
    if not SOURCE_CSV.exists():
        print(f"找不到源数据：{SOURCE_CSV}")
        print("它是公开数据集，需要单独下载（不进仓库）。下载地址与 sha256 见 "
              "data/analytics/datasets/dataco/NOTES.md。")
        return 1

    # 按 Order Id 归拢。用 dict 保序，让输出文件的顺序只由源文件决定——
    # 这样重跑一次得到逐字节相同的文件，diff 才是干净的。
    orders: dict[str, dict] = {}
    logistics: dict[str, dict] = {}
    rows_read = 0

    with open(SOURCE_CSV, encoding=SOURCE_ENCODING, newline="") as f:
        for row in csv.DictReader(f):
            rows_read += 1
            order_id = row["Order Id"].strip()

            entry = orders.get(order_id)
            if entry is None:
                # 一笔订单的多行里，状态/客户/地区/下单时间都是一样的（派生时
                # 用 ORDER BY 之外的字段核过：0 笔订单行间不一致），所以取首行即可。
                name = f"{row['Customer Fname'].strip()} {row['Customer Lname'].strip()}".strip()
                entry = orders[order_id] = {
                    "order_id": order_id,
                    "customer_id": "C" + row["Order Customer Id"].strip(),
                    "customer_name": name,
                    "status": row["Order Status"].strip(),
                    "amount": 0.0,
                    "region": row["Order Region"].strip(),
                    "created_at": parse_time(row["order date (DateOrders)"]),
                    "paid_at": "",  # 见模块开头第 4 条
                }
                if row["Delivery Status"].strip() != NO_SHIPMENT_STATUS:
                    logistics[order_id] = {
                        "order_id": order_id,
                        "carrier": "",      # 公开数据里没有，见模块开头第 6 条
                        "tracking_no": "",
                        "from_city": "",
                        "to_city": row["Order City"].strip(),  # 官方字典：Destination city
                        "status": row["Delivery Status"].strip(),
                        "updated_at": parse_time(row["shipping date (DateOrders)"]),
                    }

            entry["amount"] += float(row["Sales"])

    ORDER_SEED_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(ORDERS_CSV, ORDER_COLUMNS, orders.values(), amount_decimals=2)
    write_csv(LOGISTICS_CSV, LOGISTICS_COLUMNS, logistics.values())

    print(f"源：{SOURCE_CSV}（{SOURCE_ENCODING}，{rows_read} 行）")
    print(f"→ {ORDERS_CSV}：{len(orders)} 笔订单，{ORDERS_CSV.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"→ {LOGISTICS_CSV}：{len(logistics)} 条物流，{LOGISTICS_CSV.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"   没有物流行的订单：{len(orders) - len(logistics)}"
          f"（配送状态为 {NO_SHIPMENT_STATUS!r}，见模块开头第 5 条）")
    print(f"   状态取值：{sorted({o['status'] for o in orders.values()})}")
    return 0


def write_csv(path: Path, columns: list, rows, amount_decimals: int = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            if amount_decimals is not None and "amount" in out:
                # 浮点求和出来的 314.6400146... 会原样写进去，看起来像精度问题。
                # 订单金额本来就是两位小数的东西，落盘时收干净。
                out["amount"] = f"{out['amount']:.{amount_decimals}f}"
            writer.writerow(out)


if __name__ == "__main__":
    raise SystemExit(main())
