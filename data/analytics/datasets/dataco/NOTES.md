---
label: DataCo 智能供应链（公开数据集抽样，订单项级）
---

这是一张**订单项（order item）级的表**：**一行 = 一笔订单里的一个商品**，不是一笔订单。
本表一笔订单平均 2.8 行。所以「行数」和「订单数」是两个不同的东西——
问「有多少订单」要先按 Order Id 去重；问「卖了多少件 / 多少钱」直接按行汇总。
把行数当订单数会多算近 3 倍，而表里每一行看上去都是完整数据，错得没有任何提示。

【来源与抽样】
- 原始数据集：DataCo Smart Supply Chain for Big Data Analysis（Mendeley Data）。
  95,910,149 字节，sha256 `fa6d022ed437155e1a2f0378710602848703c8a7f203f7ff5d77805bf8480aa6`。
- 本表是**抽样**：原始 180,519 行 / 65,752 笔订单 → 本表 20,242 行 / 7,336 笔订单。
- 抽样规则：`md5("dataco:" + Order Id)` 前 4 字节 % 9 == 0 的订单**整笔**留下。
  抽的是订单不是行，所以一笔订单要么整笔在、要么整笔不在，不会只剩一半商品。
- 派生脚本：`ecom/analytics/cli/derive_dataco.py`，可以用原始数据复现本表。
- 落盘是 utf-8；原始文件是 cp1252（不是 UTF-8），所以这中间必须有一次转码。
- **这是抽样，不是全量。** 下面所有口径和结论都只对本表成立。

【列的含义】
- `order date (DateOrders)`：下单时间，形如 `1/13/2018 12:06`（M/D/YYYY H:MM）。它是**文本**。
- `Order Id`：订单号，同一笔订单的多行共用同一个值。
- `Order Status`：订单状态，9 个取值——COMPLETE / CLOSED / PROCESSING / PENDING /
  PENDING_PAYMENT / ON_HOLD / PAYMENT_REVIEW / SUSPECTED_FRAUD / CANCELED。
- `Delivery Status`：配送状态，4 个取值——Advance shipping / Shipping on time /
  Late delivery / Shipping canceled。
- `Late_delivery_risk`：1 = 有延迟风险，0 = 没有。
- `Days for shipping (real)` / `Days for shipment (scheduled)`：实际 / 计划配送天数。
- `Shipping Mode`：配送方式——Standard Class / Second Class / First Class / Same Day。
- `Market`：大区，5 个取值——LATAM / Europe / Pacific Asia / USCA / Africa。
- `Order Region`：比大区细一级的地区，23 个取值；`Order Country` 再细一级，133 个取值。
  三级是 Market ⊃ Order Region ⊃ Order Country，别混着用。
- `Category Name` / `Product Name` / `Customer Segment`：类别 / 商品 / 客户类型。
- `Order Item Quantity`：这一行卖出的件数，取值 1~5。
- `Sales`：这一行的销售额（**未打折**）。
- `Order Item Total`：这一行的**折后**金额。
- `Order Item Discount`：折扣额，约等于 Sales − Order Item Total。

【默认口径】
- **销售额默认用 `Sales` 列**（未打折）。折后口径是 `Order Item Total`，两者差约 10%
  （本表分别约 4,113,939 和 3,696,872），**不能混用**：问了销售额就用 `Sales`。
  没有特别说明时，也**不剔除**任何订单状态（CANCELED 等照样计入）。
- 金额、销量直接按行汇总，不需要去重。
- 问「订单数」必须先按 `Order Id` 去重，再计数。
- 比率类（延迟率、占比）**以「行」为分母**，不按 `Order Item Quantity` 加权——
  风险标记本来就打在行上，按件数加权会让"买得多"的行更重，那不是这份数据在说的事。
- 商品名、大区名、类别名比较或分组前**先 strip**：源数据里商品名是定宽右填空格的，
  `'Smart watch '` 不 strip 就和 `'Smart watch'` 是两个商品。
- 金额字段带浮点尾数（如 `314.6400146`），判等要用容差，不要用 `==`。
- 按时间排序要先把 `order date (DateOrders)` 转成 datetime；按字符串排是错的
  （M/D/YYYY 下 `'9/9/2017'` 会排在 `'1/1/2018'` 前面）。

【边界】
- **没有退货数据。** 只有 `Order Status = CANCELED`（取消订单），那不是退货。不要用
  它算退货率。
- **没有承运商、没有运单号。** 公开数据里根本没有这两项（是快递公司的私有数据），
  结果里不许出现承运商名字。
- **没有发货地。** 只有目的地相关的字段，没有"从哪里发出"。
- **没有成本、利润列。** 原始表里的 `Benefit per order` / `Order Profit Per Order`
  本表没有保留。问利润只能回答算不了。
- 商品和类别对不上：销量最高的商品是 `Perfect Fitness Perfect Rip Deck`，
  但它的 `Category Name` 是 `Cleats`。原始数据就是这样，不是本表串了行。
- 类别取值里有脏值，例如 `'As Seen on  TV!'`（中间是两个空格）。
